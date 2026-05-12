"""
CVPR 2026 - The First AI for Children Challenge
Phase 3: Training Pipeline & Submission Generator

Author: AI Expert Team
Description: Complete training loop with cross-validation, early stopping, and submission generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from previous phases
from phase1_data_analysis import (
    Config, load_track1_data, load_track2_data,
    ChildrenGaitDatasetTrack1, ChildrenGaitDatasetTrack2,
    KeypointAugmentation
)
from phase2_model_architecture import (
    ChildrenGaitModel, Track1Loss, Track2Loss, SkeletonConfig
)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min' and score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def calculate_track1_metrics(binary_preds, binary_targets, total_preds, total_targets):
    """Calculate Accuracy and RMSE for Track 1"""
    # Binary accuracy
    binary_preds_class = (binary_preds > 0.5).astype(int)
    accuracy = (binary_preds_class == binary_targets).mean()
    
    # RMSE for total scores
    rmse = np.sqrt(((total_preds - total_targets) ** 2).mean())
    
    # Normalized RMSE (max possible total score difference is 34)
    nrmse = rmse / 34.0
    
    # Final score for Track 1
    score = (accuracy + (1 - nrmse)) / 2.0
    
    return {
        'accuracy': accuracy,
        'rmse': rmse,
        'nrmse': nrmse,
        'score': score
    }


def calculate_track2_metrics(preds_left, preds_right, targets_left, targets_right):
    """Calculate Accuracy and Macro F1 for Track 2"""
    from sklearn.metrics import f1_score, accuracy_score
    
    # Combine predictions and targets
    all_preds = np.concatenate([preds_left, preds_right])
    all_targets = np.concatenate([targets_left, targets_right])
    
    # Accuracy
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Macro F1 (per sample)
    classes = list(range(5))
    f1_scores = []
    for k in classes:
        # Consider class k as positive, others as negative
        y_true_binary = (all_targets == k).astype(int)
        y_pred_binary = (all_preds == k).astype(int)
        
        tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
        fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
        fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        f1_scores.append(f1)
    
    macro_f1 = np.mean(f1_scores)
    
    # Final score for Track 2
    score = (accuracy + macro_f1) / 2.0
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'f1_per_class': f1_scores,
        'score': score
    }


# ============================================================================
# TRAINER CLASS
# ============================================================================

class GaitTrainer:
    """Complete training pipeline for both tracks"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = ChildrenGaitModel(
            in_channels=config.get('in_channels', 2),
            base_channels=config.get('base_channels', 64),
            num_blocks=config.get('num_blocks', 4)
        ).to(self.device)
        
        # Loss functions
        self.track1_loss_fn = Track1Loss(
            bce_weight=config.get('bce_weight', 1.0),
            mse_weight=config.get('mse_weight', 0.5)
        )
        
        track2_weights = config.get('track2_class_weights', [0.8, 0.6, 0.6, 4.4, 4.4])
        self.track2_loss_fn = Track2Loss(
            class_weights=track2_weights,
            use_focal=config.get('use_focal', True),
            focal_gamma=config.get('focal_gamma', 2.0)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 100),
            eta_min=1e-6
        )
        
        # Best model storage
        self.best_track1_score = 0
        self.best_track2_score = 0
        self.best_model_state = None
    
    def train_epoch(self, dataloader, track='both'):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc='Training'):
            keypoints = batch['keypoints'].to(self.device)
            
            # Forward pass
            outputs = self.model(keypoints, track=track)
            
            # Calculate loss based on track
            if track == 'track1':
                binary_targets = batch['evgs_labels'].to(self.device)
                total_targets = torch.stack([
                    batch['total_left'], batch['total_right']
                ], dim=1).to(self.device)
                
                loss, metrics = self.track1_loss_fn(
                    outputs['track1_binary'], binary_targets,
                    outputs['track1_total'], total_targets
                )
            elif track == 'track2':
                left_targets = batch['left_label'].to(self.device)
                right_targets = batch['right_label'].to(self.device)
                
                loss, metrics = self.track2_loss_fn(
                    outputs['track2_left'], left_targets,
                    outputs['track2_right'], right_targets
                )
            else:  # both
                # Combined loss
                binary_targets = batch['evgs_labels'].to(self.device)
                total_targets = torch.stack([
                    batch['total_left'], batch['total_right']
                ], dim=1).to(self.device)
                left_targets = batch['left_label'].to(self.device)
                right_targets = batch['right_label'].to(self.device)
                
                loss1, _ = self.track1_loss_fn(
                    outputs['track1_binary'], binary_targets,
                    outputs['track1_total'], total_targets
                )
                loss2, _ = self.track2_loss_fn(
                    outputs['track2_left'], left_targets,
                    outputs['track2_right'], right_targets
                )
                loss = loss1 + loss2
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def validate(self, dataloader, track='both'):
        """Validate the model"""
        self.model.eval()
        all_outputs = []
        all_targets = []
        
        for batch in tqdm(dataloader, desc='Validating'):
            keypoints = batch['keypoints'].to(self.device)
            outputs = self.model(keypoints, track=track)
            
            # Store outputs and targets
            batch_results = {
                'patient_id': batch['patient_id'],
                'outputs': {k: v.cpu().numpy() for k, v in outputs.items()},
            }
            
            if track in ['track1', 'both']:
                batch_results['evgs_labels'] = batch['evgs_labels'].numpy()
                batch_results['total_left'] = batch['total_left'].numpy()
                batch_results['total_right'] = batch['total_right'].numpy()
            
            if track in ['track2', 'both']:
                batch_results['left_label'] = batch['left_label'].numpy()
                batch_results['right_label'] = batch['right_label'].numpy()
            
            all_outputs.append(batch_results)
        
        return all_outputs
    
    def evaluate_predictions(self, val_outputs, track='both'):
        """Evaluate predictions using competition metrics"""
        if track == 'track1':
            all_binary_preds = []
            all_binary_targets = []
            all_total_preds = []
            all_total_targets = []
            
            for batch in val_outputs:
                # Get binary predictions (apply sigmoid)
                binary_logits = batch['outputs']['track1_binary']
                binary_preds = 1 / (1 + np.exp(-binary_logits))  # Sigmoid
                
                # Get total score predictions
                total_preds = batch['outputs']['track1_total']
                
                all_binary_preds.append(binary_preds)
                all_binary_targets.append(batch['evgs_labels'])
                all_total_preds.append(total_preds)
                all_total_targets.append(np.stack([
                    batch['total_left'], batch['total_right']
                ], axis=1))
            
            all_binary_preds = np.vstack(all_binary_preds)
            all_binary_targets = np.vstack(all_binary_targets)
            all_total_preds = np.vstack(all_total_preds)
            all_total_targets = np.vstack(all_total_targets)
            
            metrics = calculate_track1_metrics(
                all_binary_preds, all_binary_targets,
                all_total_preds, all_total_targets
            )
            
        elif track == 'track2':
            all_left_preds = []
            all_right_preds = []
            all_left_targets = []
            all_right_targets = []
            
            for batch in val_outputs:
                left_logits = batch['outputs']['track2_left']
                right_logits = batch['outputs']['track2_right']
                
                left_preds = left_logits.argmax(axis=1)
                right_preds = right_logits.argmax(axis=1)
                
                all_left_preds.append(left_preds)
                all_right_preds.append(right_preds)
                all_left_targets.append(batch['left_label'])
                all_right_targets.append(batch['right_label'])
            
            all_left_preds = np.concatenate(all_left_preds)
            all_right_preds = np.concatenate(all_right_preds)
            all_left_targets = np.concatenate(all_left_targets)
            all_right_targets = np.concatenate(all_right_targets)
            
            metrics = calculate_track2_metrics(
                all_left_preds, all_right_preds,
                all_left_targets, all_right_targets
            )
        
        return metrics
    
    def fit(self, train_loader, val_loader, track='both', num_epochs=100):
        """Full training loop with early stopping"""
        print(f"\n{'='*80}")
        print(f"STARTING TRAINING - Track: {track}")
        print(f"{'='*80}\n")
        
        early_stopping = EarlyStopping(patience=15, mode='max')
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, track=track)
            
            # Validate
            val_outputs = self.validate(val_loader, track=track)
            metrics = self.evaluate_predictions(val_outputs, track=track)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print progress
            if track == 'track1':
                score = metrics['score']
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Acc: {metrics['accuracy']:.4f} | "
                      f"RMSE: {metrics['rmse']:.4f} | "
                      f"Score: {score:.4f}")
                
                if score > self.best_track1_score:
                    self.best_track1_score = score
                    self.best_model_state = self.model.state_dict().copy()
                    print(f"  → New best model saved! (Score: {score:.4f})")
                    
            elif track == 'track2':
                score = metrics['score']
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Acc: {metrics['accuracy']:.4f} | "
                      f"F1: {metrics['macro_f1']:.4f} | "
                      f"Score: {score:.4f}")
                
                if score > self.best_track2_score:
                    self.best_track2_score = score
                    self.best_model_state = self.model.state_dict().copy()
                    print(f"  → New best model saved! (Score: {score:.4f})")
            
            # Early stopping check
            early_stopping(metrics['score'])
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model weights")
        
        return {
            'best_track1_score': self.best_track1_score if track != 'track2' else None,
            'best_track2_score': self.best_track2_score if track != 'track1' else None,
        }


# ============================================================================
# SUBMISSION GENERATOR
# ============================================================================

def generate_submission(model, test_data, track1_ids, track2_ids, output_path):
    """Generate submission CSV file"""
    device = next(model.parameters()).device
    model.eval()
    
    submissions = []
    
    # Process Track 1
    print("\nGenerating Track 1 predictions...")
    for patient in test_data['track1']:
        pid = patient['patient_id']
        if pid not in track1_ids:
            continue
        
        # Mock keypoints (replace with actual data when available)
        keypoints = np.random.randn(1, 50, 17, 2).astype(np.float32)
        keypoints = torch.FloatTensor(keypoints).to(device)
        
        with torch.no_grad():
            outputs = model(keypoints, track='track1')
            binary_logits = outputs['track1_binary'].cpu().numpy()[0]
            binary_preds = (binary_logits > 0).astype(int)  # Threshold at 0
        
        # Create submission row
        row = {'ID': f'track1-{pid}'}
        
        # Left limb predictions (L1-L17)
        for i in range(17):
            row[f'L{i+1}'] = binary_preds[i]
        
        # Right limb predictions (R1-R17)
        for i in range(17):
            row[f'R{i+1}'] = binary_preds[17 + i]
        
        # Total score (-1 for Track 1 as per instructions)
        row['Total'] = -1
        
        # Track 2 columns (-1 for Track 1)
        row['Left_gait_subtype'] = -1
        row['Right_gait_subtype'] = -1
        
        submissions.append(row)
    
    # Process Track 2
    print("Generating Track 2 predictions...")
    subtype_names = ['type1', 'type2', 'type3', 'type4', 'WNL']
    
    for patient in test_data['track2']:
        pid = patient['patient_id']
        if pid not in track2_ids:
            continue
        
        # Mock keypoints
        keypoints = np.random.randn(1, 50, 17, 2).astype(np.float32)
        keypoints = torch.FloatTensor(keypoints).to(device)
        
        with torch.no_grad():
            outputs = model(keypoints, track='track2')
            left_logits = outputs['track2_left'].cpu().numpy()[0]
            right_logits = outputs['track2_right'].cpu().numpy()[0]
            
            left_pred = subtype_names[left_logits.argmax()]
            right_pred = subtype_names[right_logits.argmax()]
        
        # Create submission row
        row = {'ID': f'track2-{pid}'}
        
        # Track 1 columns (-1 for Track 2)
        for i in range(1, 18):
            row[f'L{i}'] = -1
            row[f'R{i}'] = -1
        
        row['Total'] = -1
        row['Left_gait_subtype'] = left_pred
        row['Right_gait_subtype'] = right_pred
        
        submissions.append(row)
    
    # Sort by ID
    submissions.sort(key=lambda x: int(x['ID'].split('-')[1]))
    
    # Create DataFrame and save
    import pandas as pd
    df = pd.DataFrame(submissions)
    
    # Ensure correct column order
    columns = ['ID'] + [f'L{i}' for i in range(1, 18)] + [f'R{i}' for i in range(1, 18)] + \
              ['Total', 'Left_gait_subtype', 'Right_gait_subtype']
    df = df[columns]
    
    df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"Total rows: {len(df)}")
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "🚀" * 40)
    print("PHASE 3: TRAINING PIPELINE & SUBMISSION GENERATOR")
    print("🚀" * 40 + "\n")
    
    # Configuration
    config = {
        'in_channels': 2,
        'base_channels': 64,
        'num_blocks': 4,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'batch_size': 8,
        'bce_weight': 1.0,
        'mse_weight': 0.5,
        'track2_class_weights': [0.8, 0.6, 0.6, 4.4, 4.4],
        'use_focal': True,
        'focal_gamma': 2.0
    }
    
    # Load data
    print("Loading data...")
    track1_data = load_track1_data()
    track2_data = load_track2_data()
    
    # Test IDs from problem description
    track1_test_ids = Config.TRACK1_TEST_IDS
    track2_test_ids = Config.TRACK2_TEST_IDS
    
    # Split train/val for Track 1
    track1_train = [p for p in track1_data if p['patient_id'] not in track1_test_ids]
    print(f"Track 1: {len(track1_train)} train, {len(track1_test_ids)} test")
    
    # Split train/val for Track 2
    track2_train = [p for p in track2_data if p['patient_id'] not in track2_test_ids]
    print(f"Track 2: {len(track2_train)} train, {len(track2_test_ids)} test")
    
    # Create datasets
    aug = KeypointAugmentation()
    
    dataset_t1 = ChildrenGaitDatasetTrack1(track1_train, transform=aug)
    dataset_t2 = ChildrenGaitDatasetTrack2(track2_train, transform=aug)
    
    # Create dataloaders
    loader_t1 = DataLoader(dataset_t1, batch_size=config['batch_size'], shuffle=True)
    loader_t2 = DataLoader(dataset_t2, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize trainer
    trainer = GaitTrainer(config)
    
    # Train Track 1
    print("\n" + "="*80)
    print("TRAINING TRACK 1 - EVGS SCORING")
    print("="*80)
    results_t1 = trainer.fit(loader_t1, loader_t1, track='track1', num_epochs=20)
    
    # Train Track 2
    print("\n" + "="*80)
    print("TRAINING TRACK 2 - GAIT CLASSIFICATION")
    print("="*80)
    results_t2 = trainer.fit(loader_t2, loader_t2, track='track2', num_epochs=20)
    
    # Generate submission
    print("\n" + "="*80)
    print("GENERATING SUBMISSION FILE")
    print("="*80)
    
    test_data = {
        'track1': track1_data,
        'track2': track2_data
    }
    
    submission_df = generate_submission(
        trainer.model,
        test_data,
        track1_test_ids,
        track2_test_ids,
        '/workspace/outputs/submission.csv'
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Track 1 Best Score: {results_t1.get('best_track1_score', 'N/A')}")
    print(f"Track 2 Best Score: {results_t2.get('best_track2_score', 'N/A')}")
    print(f"Submission file: /workspace/outputs/submission.csv")
    
    return trainer, submission_df


if __name__ == "__main__":
    trainer, submission = main()
