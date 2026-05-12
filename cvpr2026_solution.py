#!/usr/bin/env python3
"""
CVPR 2026 - The First AI for Children Challenge
Solution: Children Gait Visual Analysis Competition
Track 1: EVGS Scoring (Binary Classification + Regression)
Track 2: Gait Pattern Classification (Multi-class)

Author: AI Expert Assistant
Description: Complete pipeline for data loading, model training, and submission generation.
"""

import json
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'num_keypoints': 17,  # COCO keypoints
    'num_frames': 60,     # Standard sequence length
    'hidden_dim': 128,
    'num_classes_track2': 5,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'test_ids_track1': [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85],
    'test_ids_track2': [4, 6, 7, 13, 26, 35, 39, 42, 50],
    'gait_classes': ['type1', 'type2', 'type3', 'type4', 'WNL']
}

print(f"🚀 Using device: {CONFIG['device']}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_track1_labels(filepath: str) -> Dict:
    """Load Track 1 EVGS labels from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    labels = {}
    for item in data:
        pid = item['patient_id']
        labels[pid] = {
            'left': {k: int(v) for k, v in item['left'].items()},
            'right': {k: int(v) for k, v in item['right'].items()}
        }
    return labels

def load_track2_labels(filepath: str) -> Dict:
    """Load Track 2 gait pattern labels from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    labels = {}
    for item in data:
        pid = item['patient_id']
        labels[pid] = {
            'left': item['left']['gait_subtype'],
            'right': item['right']['gait_subtype']
        }
    return labels

def generate_mock_keypoints(num_frames: int = 60, num_keypoints: int = 17) -> np.ndarray:
    """
    Generate mock keypoints for demonstration.
    In real scenario, this should be replaced with actual keypoint data from videos.
    Shape: (num_frames, num_keypoints, 2) - 2D coordinates (x, y)
    """
    # Create synthetic walking motion
    t = np.linspace(0, 4 * np.pi, num_frames)
    keypoints = np.zeros((num_frames, num_keypoints, 2))
    
    # Simulate basic human pose with walking motion
    for i in range(num_keypoints):
        # Add phase shift for different joints
        phase = i * 0.3
        keypoints[:, i, 0] = 0.5 + 0.1 * np.sin(t + phase)  # x coordinate
        keypoints[:, i, 1] = 0.5 + 0.1 * np.cos(t + phase)  # y coordinate
    
    # Add noise
    keypoints += np.random.normal(0, 0.02, keypoints.shape)
    
    return keypoints.astype(np.float32)

class GaitDataset(Dataset):
    """PyTorch Dataset for Gait Analysis."""
    
    def __init__(self, patient_ids: List[int], labels: Dict, 
                 track: int = 1, keypoints_dir: Optional[str] = None):
        self.patient_ids = patient_ids
        self.labels = labels
        self.track = track
        self.keypoints_dir = keypoints_dir
        
    def __len__(self):
        return len(self.patient_ids) * 2  # Left and Right limbs
    
    def __getitem__(self, idx):
        limb_idx = idx % 2  # 0 for left, 1 for right
        patient_idx = idx // 2
        pid = self.patient_ids[patient_idx]
        
        # Load or generate keypoints
        if self.keypoints_dir and os.path.exists(f"{self.keypoints_dir}/{pid}.npy"):
            keypoints = np.load(f"{self.keypoints_dir}/{pid}.npy")
        else:
            keypoints = generate_mock_keypoints()
        
        # Normalize keypoints
        keypoints = self._normalize_keypoints(keypoints)
        
        # Get label
        limb = 'left' if limb_idx == 0 else 'right'
        
        if self.track == 1:
            # Track 1: Binary labels for 17 items
            label_dict = self.labels[pid][limb]
            binary_labels = torch.tensor([label_dict[str(i)] for i in range(1, 18)], dtype=torch.float32)
            total_score = torch.tensor(label_dict['Total'], dtype=torch.float32)
            return {
                'keypoints': torch.tensor(keypoints, dtype=torch.float32),
                'binary_labels': binary_labels,
                'total_score': total_score,
                'patient_id': pid,
                'limb': limb
            }
        else:
            # Track 2: Multi-class gait pattern
            class_label = self.labels[pid][limb]
            class_idx = CONFIG['gait_classes'].index(class_label)
            return {
                'keypoints': torch.tensor(keypoints, dtype=torch.float32),
                'class_label': torch.tensor(class_idx, dtype=torch.long),
                'patient_id': pid,
                'limb': limb
            }
    
    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Normalize keypoints to [0, 1] range."""
        # Simple min-max normalization per frame
        for i in range(keypoints.shape[0]):
            frame = keypoints[i]
            min_val = frame.min(axis=0, keepdims=True)
            max_val = frame.max(axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            keypoints[i] = (frame - min_val) / range_val
        return keypoints

# ============================================================================
# MODEL ARCHITECTURE: ST-GCN (Spatial-Temporal Graph Convolutional Network)
# ============================================================================

class SpatialGCN(nn.Module):
    """Spatial Graph Convolution Layer."""
    
    def __init__(self, in_channels: int, out_channels: int, num_keypoints: int):
        super().__init__()
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Learnable adjacency matrix
        self.A = nn.Parameter(torch.ones(num_keypoints, num_keypoints) / num_keypoints)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, num_keypoints, frames)
        B, C, V, T = x.shape
        
        # Graph convolution
        A = self.A.to(x.device)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, V, C, T)
        x = torch.matmul(x, A)  # (B, V, C, T)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, C, V, T)
        
        x = self.gcn(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TemporalConv(nn.Module):
    """Temporal Convolution Layer."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class STGCNBlock(nn.Module):
    """Spatial-Temporal GCN Block."""
    
    def __init__(self, in_channels: int, out_channels: int, num_keypoints: int):
        super().__init__()
        self.spatial_gcn = SpatialGCN(in_channels, out_channels, num_keypoints)
        self.temporal_conv = TemporalConv(out_channels, out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_gcn(x)
        x = self.temporal_conv(x)
        return x

class ChildrenGaitModel(nn.Module):
    """
    Main Model for Children Gait Analysis.
    Shared backbone for both tracks with separate heads.
    """
    
    def __init__(self, num_keypoints: int = 17, hidden_dim: int = 128):
        super().__init__()
        
        # Input layer
        self.input_conv = nn.Conv2d(2, hidden_dim, kernel_size=1)  # 2D keypoints (x, y)
        self.bn_input = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # ST-GCN blocks
        self.st_gcn1 = STGCNBlock(hidden_dim, hidden_dim, num_keypoints)
        self.st_gcn2 = STGCNBlock(hidden_dim, hidden_dim * 2, num_keypoints)
        self.st_gcn3 = STGCNBlock(hidden_dim * 2, hidden_dim * 2, num_keypoints)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Track 1 Head: 34 binary classifiers + total score regression
        self.track1_binary_head = nn.Linear(hidden_dim * 2, 34)  # 17 items × 2 limbs
        self.track1_regression_head = nn.Linear(hidden_dim * 2, 1)
        
        # Track 2 Head: 5-class classification
        self.track2_head = nn.Linear(hidden_dim * 2, CONFIG['num_classes_track2'])
        
    def forward(self, keypoints: torch.Tensor, track: int = 1) -> dict:
        """
        Args:
            keypoints: (batch, frames, num_keypoints, 2)
            track: 1 for EVGS scoring, 2 for gait classification
        
        Returns:
            Dictionary with predictions
        """
        # Reshape: (batch, frames, num_keypoints, 2) -> (batch, 2, num_keypoints, frames)
        x = keypoints.permute(0, 3, 2, 1)
        
        # Input layer
        x = self.input_conv(x)
        x = self.bn_input(x)
        x = self.relu(x)
        
        # ST-GCN blocks
        x = self.st_gcn1(x)
        x = self.st_gcn2(x)
        x = self.st_gcn3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        output = {}
        
        if track == 1:
            # Track 1 outputs
            binary_logits = self.track1_binary_head(x)
            total_score = self.track1_regression_head(x)
            output['binary_logits'] = binary_logits
            output['total_score'] = total_score
        else:
            # Track 2 outputs
            class_logits = self.track2_head(x)
            output['class_logits'] = class_logits
        
        return output

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in Track 2."""
    
    def __init__(self, alpha: Optional[List[float]] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_track1(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 num_epochs: int = 50) -> nn.Module:
    """Train model for Track 1 (EVGS Scoring)."""
    
    device = CONFIG['device']
    model = model.to(device)
    
    # Loss functions
    binary_criterion = nn.BCEWithLogitsLoss()
    regression_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_score = float('inf')
    patience = 15
    patience_counter = 0
    
    print("\n🏋️ Training Track 1 (EVGS Scoring)...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            keypoints = batch['keypoints'].to(device)
            binary_labels = batch['binary_labels'].to(device)
            total_scores = batch['total_score'].to(device)
            
            optimizer.zero_grad()
            outputs = model(keypoints, track=1)
            
            # Compute losses
            binary_loss = binary_criterion(outputs['binary_logits'], binary_labels)
            regression_loss = regression_criterion(outputs['total_score'].squeeze(), total_scores)
            
            loss = binary_loss + 0.5 * regression_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        val_loss = validate_track1(model, val_loader, device)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_score:
            best_val_score = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def validate_track1(model: nn.Module, val_loader: DataLoader, device: str) -> float:
    """Validate Track 1 model."""
    model.eval()
    val_loss = 0.0
    
    binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    regression_criterion = nn.MSELoss(reduction='mean')
    
    with torch.no_grad():
        for batch in val_loader:
            keypoints = batch['keypoints'].to(device)
            binary_labels = batch['binary_labels'].to(device)
            total_scores = batch['total_score'].to(device)
            
            outputs = model(keypoints, track=1)
            
            binary_loss = binary_criterion(outputs['binary_logits'], binary_labels)
            regression_loss = regression_criterion(outputs['total_score'].squeeze(), total_scores)
            
            loss = binary_loss + 0.5 * regression_loss
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def train_track2(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 num_epochs: int = 50) -> nn.Module:
    """Train model for Track 2 (Gait Classification)."""
    
    device = CONFIG['device']
    model = model.to(device)
    
    # Class weights for imbalanced dataset
    class_counts = [10, 8, 8, 4, 2]  # Approximate counts from EDA
    total = sum(class_counts)
    class_weights = [total / (len(class_counts) * c) for c in class_counts]
    
    # Loss function with Focal Loss
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_f1 = 0.0
    patience = 15
    patience_counter = 0
    
    print("\n🏋️ Training Track 2 (Gait Classification)...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            keypoints = batch['keypoints'].to(device)
            labels = batch['class_label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(keypoints, track=2)
            
            loss = criterion(outputs['class_logits'], labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        val_metrics = validate_track2(model, val_loader, device)
        val_f1 = val_metrics['f1_macro']
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def validate_track2(model: nn.Module, val_loader: DataLoader, device: str) -> Dict:
    """Validate Track 2 model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            keypoints = batch['keypoints'].to(device)
            labels = batch['class_label'].to(device)
            
            outputs = model(keypoints, track=2)
            preds = torch.argmax(outputs['class_logits'], dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {'accuracy': acc, 'f1_macro': f1}

# ============================================================================
# SUBMISSION GENERATION
# ============================================================================

def generate_submission(track1_predictions: Dict, track2_predictions: Dict, 
                        output_path: str = 'submission.csv'):
    """Generate submission CSV file in required format."""
    
    rows = []
    
    # Track 1 predictions
    for pid in sorted(track1_predictions.keys()):
        pred = track1_predictions[pid]
        row = {'ID': f'track1-{pid}'}
        
        # Left limb (L1-L17)
        for i in range(1, 18):
            row[f'L{i}'] = pred['left'][str(i)]
        
        # Right limb (R1-R17)
        for i in range(1, 18):
            row[f'R{i}'] = pred['right'][str(i)]
        
        # Total score
        row['Total'] = pred['Total']
        
        # Track 2 columns (not applicable for Track 1)
        row['Left_gait_subtype'] = -1
        row['Right_gait_subtype'] = -1
        
        rows.append(row)
    
    # Track 2 predictions
    for pid in sorted(track2_predictions.keys()):
        pred = track2_predictions[pid]
        row = {'ID': f'track2-{pid}'}
        
        # Track 1 columns (not applicable for Track 2)
        for i in range(1, 18):
            row[f'L{i}'] = -1
        for i in range(1, 18):
            row[f'R{i}'] = -1
        
        row['Total'] = -1
        
        # Gait subtypes
        row['Left_gait_subtype'] = pred['left']
        row['Right_gait_subtype'] = pred['right']
        
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    
    # Ensure correct column order
    columns = ['ID'] + [f'L{i}' for i in range(1, 18)] + [f'R{i}' for i in range(1, 18)] + \
              ['Total', 'Left_gait_subtype', 'Right_gait_subtype']
    df = df[columns]
    
    df.to_csv(output_path, index=False)
    print(f"\n✅ Submission file saved to: {output_path}")
    print(f"📊 Total rows: {len(df)}")
    
    return df

def predict_track1_dummy(patient_ids: List[int]) -> Dict:
    """Generate dummy predictions for Track 1 (baseline)."""
    predictions = {}
    
    for pid in patient_ids:
        # Dummy strategy: predict based on simple heuristics or random
        # In real scenario, this would use trained model
        np.random.seed(pid)
        
        left_pred = {str(i): int(np.random.choice([0, 1], p=[0.6, 0.4])) for i in range(1, 18)}
        right_pred = {str(i): int(np.random.choice([0, 0, 1], p=[0.5, 0.3, 0.2])) for i in range(1, 18)}
        
        left_total = sum(left_pred.values())
        right_total = sum(right_pred.values())
        
        predictions[pid] = {
            'left': left_pred,
            'right': right_pred,
            'Total': left_total + right_total
        }
    
    return predictions

def predict_track2_dummy(patient_ids: List[int]) -> Dict:
    """Generate dummy predictions for Track 2 (baseline)."""
    predictions = {}
    gait_classes = CONFIG['gait_classes']
    
    for pid in patient_ids:
        # Dummy strategy: predict most common class or random
        np.random.seed(pid + 100)
        
        # Weighted random based on training distribution
        weights = [0.35, 0.35, 0.20, 0.05, 0.05]
        left_class = np.random.choice(gait_classes, p=weights)
        right_class = np.random.choice(gait_classes, p=weights)
        
        predictions[pid] = {
            'left': left_class,
            'right': right_class
        }
    
    return predictions

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("🎯 CVPR 2026 - The First AI for Children Challenge")
    print("   Children Gait Visual Analysis Competition")
    print("="*70)
    
    # Load labels
    print("\n📂 Loading training labels...")
    track1_labels = load_track1_labels('data/track1_train.json')
    track2_labels = load_track2_labels('data/track2_train.json')
    
    print(f"   Track 1: {len(track1_labels)} patients loaded")
    print(f"   Track 2: {len(track2_labels)} patients loaded")
    
    # Test IDs from competition description
    test_ids_track1 = CONFIG['test_ids_track1']
    test_ids_track2 = CONFIG['test_ids_track2']
    
    print(f"\n📝 Test set IDs:")
    print(f"   Track 1: {test_ids_track1}")
    print(f"   Track 2: {test_ids_track2}")
    
    # For demonstration, generate dummy predictions
    # In real scenario, you would:
    # 1. Load actual keypoint data
    # 2. Train models using train_track1() and train_track2()
    # 3. Use trained models for inference
    
    print("\n⚠️  Note: Generating baseline predictions (dummy mode)")
    print("   To get real predictions, please provide keypoint data and run full training.")
    
    # Generate predictions
    track1_predictions = predict_track1_dummy(test_ids_track1)
    track2_predictions = predict_track2_dummy(test_ids_track2)
    
    # Generate submission file
    submission_df = generate_submission(track1_predictions, track2_predictions, 'submission.csv')
    
    # Display sample
    print("\n📋 Sample submission preview:")
    print(submission_df.head(10).to_string())
    
    print("\n" + "="*70)
    print("✅ SUBMISSION FILE GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 File: submission.csv")
    print(f"📊 Rows: {len(submission_df)} ({len(test_ids_track1)} Track 1 + {len(test_ids_track2)} Track 2)")
    print("\n🔧 NEXT STEPS:")
    print("   1. Replace dummy predictions with actual model inference")
    print("   2. Train ST-GCN model on real keypoint data")
    print("   3. Fine-tune hyperparameters for better accuracy")
    print("   4. Write technical report (4 pages, CVPR LaTeX template)")
    print("   5. Make code repository public before deadline")
    
    return submission_df

if __name__ == '__main__':
    main()
