"""
CVPR 2026 - The First AI for Children Challenge
Comprehensive Data Analysis & Solution Framework

Author: AI Expert Team
Description: Complete solution framework for Children Gait Visual Analysis
"""

import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the entire pipeline"""
    
    # Paths
    DATA_DIR = Path('/workspace/data')
    OUTPUT_DIR = Path('/workspace/outputs')
    
    # Track 1 Test IDs (from problem description)
    TRACK1_TEST_IDS = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
    
    # Track 2 Test IDs (from problem description)
    TRACK2_TEST_IDS = [4, 6, 7, 13, 26, 35, 39, 42, 50]
    
    # EVGS Items (17 items per limb)
    EVGS_ITEMS = [str(i) for i in range(1, 18)]
    
    # Gait Subtypes (Track 2)
    GAIT_SUBTYPES = ['type1', 'type2', 'type3', 'type4', 'WNL']
    
    # Model hyperparameters
    NUM_JOINTS = 17  # Number of keypoints
    NUM_CHANNELS = 2  # x, y coordinates (or 3 with confidence)
    SEQUENCE_LENGTH = 50  # Frames per sequence
    
    # Training params
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Device
    DEVICE = 'cuda' if True else 'cpu'  # Will check actual availability


# ============================================================================
# DATA LOADING & ANALYSIS
# ============================================================================

def load_track1_data(json_path='/workspace/track1_train.json'):
    """Load and parse Track 1 training data"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_track2_data(json_path='/workspace/track2_train.json'):
    """Load and parse Track 2 training data"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def analyze_track1_labels(data):
    """Comprehensive analysis of Track 1 labels"""
    print("=" * 80)
    print("TRACK 1 - EVGS SCORING ANALYSIS")
    print("=" * 80)
    
    n_patients = len(data)
    print(f"\nTotal patients: {n_patients}")
    
    # Initialize counters for each item
    left_counters = {item: Counter() for item in Config.EVGS_ITEMS}
    right_counters = {item: Counter() for item in Config.EVGS_ITEMS}
    total_scores_left = []
    total_scores_right = []
    
    for patient in data:
        pid = patient['patient_id']
        
        # Left limb
        left_data = patient['left']
        for item in Config.EVGS_ITEMS:
            left_counters[item][left_data[item]] += 1
        total_scores_left.append(left_data['Total'])
        
        # Right limb
        right_data = patient['right']
        for item in Config.EVGS_ITEMS:
            right_counters[item][right_data[item]] += 1
        total_scores_right.append(right_data['Total'])
    
    # Calculate statistics
    print("\n" + "-" * 80)
    print("LEFT LIMB - Class Distribution per EVGS Item")
    print("-" * 80)
    print(f"{'Item':<6} {'Negative (0)':<15} {'Positive (1)':<15} {'Pos Rate (%)':<15}")
    
    left_pos_rates = {}
    for item in Config.EVGS_ITEMS:
        neg = left_counters[item].get(0, 0)
        pos = left_counters[item].get(1, 0)
        total = neg + pos
        pos_rate = (pos / total * 100) if total > 0 else 0
        left_pos_rates[item] = pos_rate
        print(f"{item:<6} {neg:<15} {pos:<15} {pos_rate:<15.2f}")
    
    print("\n" + "-" * 80)
    print("RIGHT LIMB - Class Distribution per EVGS Item")
    print("-" * 80)
    print(f"{'Item':<6} {'Negative (0)':<15} {'Positive (1)':<15} {'Pos Rate (%)':<15}")
    
    right_pos_rates = {}
    for item in Config.EVGS_ITEMS:
        neg = right_counters[item].get(0, 0)
        pos = right_counters[item].get(1, 0)
        total = neg + pos
        pos_rate = (pos / total * 100) if total > 0 else 0
        right_pos_rates[item] = pos_rate
        print(f"{item:<6} {neg:<15} {pos:<15} {pos_rate:<15.2f}")
    
    # Total score statistics
    print("\n" + "-" * 80)
    print("TOTAL SCORE STATISTICS")
    print("-" * 80)
    print(f"Left  - Mean: {np.mean(total_scores_left):.2f}, Std: {np.std(total_scores_left):.2f}, Min: {min(total_scores_left)}, Max: {max(total_scores_left)}")
    print(f"Right - Mean: {np.mean(total_scores_right):.2f}, Std: {np.std(total_scores_right):.2f}, Min: {min(total_scores_right)}, Max: {max(total_scores_right)}")
    
    # Identify imbalanced items
    print("\n" + "-" * 80)
    print("IMBALANCED ITEMS (Positive Rate < 20% or > 80%)")
    print("-" * 80)
    imbalanced_left = [(k, v) for k, v in left_pos_rates.items() if v < 20 or v > 80]
    imbalanced_right = [(k, v) for k, v in right_pos_rates.items() if v < 20 or v > 80]
    
    if imbalanced_left:
        print("Left limb:", imbalanced_left)
    if imbalanced_right:
        print("Right limb:", imbalanced_right)
    if not imbalanced_left and not imbalanced_right:
        print("No severely imbalanced items found.")
    
    return {
        'n_patients': n_patients,
        'left_pos_rates': left_pos_rates,
        'right_pos_rates': right_pos_rates,
        'total_scores_left': total_scores_left,
        'total_scores_right': total_scores_right,
        'imbalanced_items': imbalanced_left + imbalanced_right
    }


def analyze_track2_labels(data):
    """Comprehensive analysis of Track 2 labels"""
    print("\n" + "=" * 80)
    print("TRACK 2 - GAIT CLASSIFICATION ANALYSIS")
    print("=" * 80)
    
    n_patients = len(data)
    print(f"\nTotal patients: {n_patients}")
    
    # Count gait subtypes
    left_counter = Counter()
    right_counter = Counter()
    
    for patient in data:
        left_subtype = patient['left']['gait_subtype']
        right_subtype = patient['right']['gait_subtype']
        left_counter[left_subtype] += 1
        right_counter[right_subtype] += 1
    
    # Combined counter
    combined_counter = left_counter + right_counter
    
    print("\n" + "-" * 80)
    print("LEFT LIMB - Gait Subtype Distribution")
    print("-" * 80)
    print(f"{'Subtype':<10} {'Count':<10} {'Percentage (%)':<15}")
    for subtype in Config.GAIT_SUBTYPES:
        count = left_counter.get(subtype, 0)
        pct = count / n_patients * 100
        print(f"{subtype:<10} {count:<10} {pct:<15.2f}")
    
    print("\n" + "-" * 80)
    print("RIGHT LIMB - Gait Subtype Distribution")
    print("-" * 80)
    print(f"{'Subtype':<10} {'Count':<10} {'Percentage (%)':<15}")
    for subtype in Config.GAIT_SUBTYPES:
        count = right_counter.get(subtype, 0)
        pct = count / n_patients * 100
        print(f"{subtype:<10} {count:<10} {pct:<15.2f}")
    
    print("\n" + "-" * 80)
    print("COMBINED DISTRIBUTION (Both Limbs)")
    print("-" * 80)
    total_samples = n_patients * 2
    print(f"{'Subtype':<10} {'Count':<10} {'Percentage (%)':<15}")
    for subtype in Config.GAIT_SUBTYPES:
        count = combined_counter.get(subtype, 0)
        pct = count / total_samples * 100
        print(f"{subtype:<10} {count:<10} {pct:<15.2f}")
    
    # Calculate class weights for imbalanced dataset
    print("\n" + "-" * 80)
    print("CLASS WEIGHTS FOR TRAINING (Inverse Frequency)")
    print("-" * 80)
    class_weights = {}
    for subtype in Config.GAIT_SUBTYPES:
        count = combined_counter.get(subtype, 0)
        if count > 0:
            weight = total_samples / (len(Config.GAIT_SUBTYPES) * count)
        else:
            weight = 10.0  # High weight for missing classes
        class_weights[subtype] = weight
        print(f"{subtype:<10} Weight: {weight:.4f}")
    
    # Check for same-side consistency
    same_side_count = sum(1 for p in data if p['left']['gait_subtype'] == p['right']['gait_subtype'])
    print(f"\nPatients with same gait subtype on both sides: {same_side_count}/{n_patients} ({same_side_count/n_patients*100:.1f}%)")
    
    return {
        'n_patients': n_patients,
        'left_counter': dict(left_counter),
        'right_counter': dict(right_counter),
        'combined_counter': dict(combined_counter),
        'class_weights': class_weights,
        'same_side_ratio': same_side_count / n_patients
    }


# ============================================================================
# DATASET CLASSES FOR PYTORCH
# ============================================================================

class ChildrenGaitDatasetTrack1:
    """
    PyTorch Dataset for Track 1 - EVGS Scoring
    
    Expected keypoint format: (seq_len, num_joints, num_channels)
    Labels: 34 binary values (17 items × 2 limbs) + 2 total scores
    """
    
    def __init__(self, data_json, keypoints_dict=None, transform=None):
        """
        Args:
            data_json: Path to JSON file or list of patient data
            keypoints_dict: Dictionary mapping patient_id to keypoint sequences
            transform: Optional transform to apply to keypoints
        """
        if isinstance(data_json, str):
            with open(data_json, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = data_json
        
        self.keypoints_dict = keypoints_dict or {}
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        patient = self.data[idx]
        pid = patient['patient_id']
        
        # Get keypoints (mock if not available)
        if pid in self.keypoints_dict:
            keypoints = self.keypoints_dict[pid]
        else:
            # Mock keypoints for development
            seq_len = Config.SEQUENCE_LENGTH
            keypoints = np.random.randn(seq_len, Config.NUM_JOINTS, Config.NUM_CHANNELS).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            keypoints = self.transform(keypoints)
        
        # Prepare labels
        left_labels = [patient['left'][str(i)] for i in range(1, 18)]
        right_labels = [patient['right'][str(i)] for i in range(1, 18)]
        all_labels = left_labels + right_labels  # 34 binary labels
        
        # Total scores
        total_left = patient['left']['Total']
        total_right = patient['right']['Total']
        
        sample = {
            'keypoints': keypoints,
            'patient_id': pid,
            'evgs_labels': np.array(all_labels, dtype=np.float32),
            'total_left': np.array(total_left, dtype=np.float32),
            'total_right': np.array(total_right, dtype=np.float32)
        }
        
        return sample


class ChildrenGaitDatasetTrack2:
    """
    PyTorch Dataset for Track 2 - Gait Classification
    
    Labels: 2 class indices (left and right limb)
    """
    
    def __init__(self, data_json, keypoints_dict=None, transform=None):
        if isinstance(data_json, str):
            with open(data_json, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = data_json
        
        self.keypoints_dict = keypoints_dict or {}
        self.transform = transform
        self.subtype_to_idx = {s: i for i, s in enumerate(Config.GAIT_SUBTYPES)}
        self.idx_to_subtype = {i: s for i, s in enumerate(Config.GAIT_SUBTYPES)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        patient = self.data[idx]
        pid = patient['patient_id']
        
        # Get keypoints (mock if not available)
        if pid in self.keypoints_dict:
            keypoints = self.keypoints_dict[pid]
        else:
            seq_len = Config.SEQUENCE_LENGTH
            keypoints = np.random.randn(seq_len, Config.NUM_JOINTS, Config.NUM_CHANNELS).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            keypoints = self.transform(keypoints)
        
        # Prepare labels
        left_label = self.subtype_to_idx[patient['left']['gait_subtype']]
        right_label = self.subtype_to_idx[patient['right']['gait_subtype']]
        
        sample = {
            'keypoints': keypoints,
            'patient_id': pid,
            'left_label': np.array(left_label, dtype=np.int64),
            'right_label': np.array(right_label, dtype=np.int64)
        }
        
        return sample


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class KeypointAugmentation:
    """Data augmentation for keypoint sequences"""
    
    def __init__(self, 
                 noise_std=0.01,
                 scale_range=(0.9, 1.1),
                 rotate_range=(-10, 10),
                 time_mask_prob=0.1,
                 joint_mask_prob=0.05):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.rotate_range = rotate_range
        self.time_mask_prob = time_mask_prob
        self.joint_mask_prob = joint_mask_prob
    
    def __call__(self, keypoints):
        """
        Args:
            keypoints: (seq_len, num_joints, num_channels)
        """
        keypoints = keypoints.copy()
        
        # Add Gaussian noise
        noise = np.random.randn(*keypoints.shape) * self.noise_std
        keypoints += noise
        
        # Random scaling
        scale = np.random.uniform(*self.scale_range)
        keypoints *= scale
        
        # Random rotation (2D only)
        if keypoints.shape[-1] >= 2:
            angle = np.random.uniform(*self.rotate_range)
            angle_rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            x, y = keypoints[..., 0], keypoints[..., 1]
            keypoints[..., 0] = x * cos_a - y * sin_a
            keypoints[..., 1] = x * sin_a + y * cos_a
        
        # Time masking
        if np.random.rand() < self.time_mask_prob:
            seq_len = keypoints.shape[0]
            mask_len = np.random.randint(1, max(2, seq_len // 10))
            start = np.random.randint(0, seq_len - mask_len + 1)
            keypoints[start:start + mask_len] = 0
        
        # Joint masking
        if np.random.rand() < self.joint_mask_prob:
            num_joints = keypoints.shape[1]
            mask_joints = np.random.choice(num_joints, size=max(1, num_joints // 10), replace=False)
            keypoints[:, mask_joints, :] = 0
        
        return keypoints


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "🚀" * 40)
    print("CVPR 2026 - CHILDREN GAIT VISUAL ANALYSIS")
    print("Comprehensive Data Analysis & Solution Framework")
    print("🚀" * 40 + "\n")
    
    # Load data
    print("Loading Track 1 data...")
    track1_data = load_track1_data()
    
    print("Loading Track 2 data...")
    track2_data = load_track2_data()
    
    # Analyze Track 1
    track1_stats = analyze_track1_labels(track1_data)
    
    # Analyze Track 2
    track2_stats = analyze_track2_labels(track2_data)
    
    # Create datasets (with mock keypoints for now)
    print("\n" + "=" * 80)
    print("CREATING PYTORCH DATASETS")
    print("=" * 80)
    
    train_dataset_t1 = ChildrenGaitDatasetTrack1(track1_data)
    print(f"Track 1 Dataset: {len(train_dataset_t1)} samples")
    
    train_dataset_t2 = ChildrenGaitDatasetTrack2(track2_data)
    print(f"Track 2 Dataset: {len(train_dataset_t2)} samples")
    
    # Test augmentation
    print("\nTesting augmentation pipeline...")
    aug = KeypointAugmentation()
    mock_keypoints = np.random.randn(50, 17, 2)
    augmented = aug(mock_keypoints)
    print(f"Original shape: {mock_keypoints.shape}, Augmented shape: {augmented.shape}")
    print("✅ Augmentation pipeline working!")
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Track 1: {track1_stats['n_patients']} patients, 34 binary labels per patient")
    print(f"Track 2: {track2_stats['n_patients']} patients, 5-class classification")
    print(f"Track 2 Class Weights: {track2_stats['class_weights']}")
    print(f"Track 2 Same-side consistency: {track2_stats['same_side_ratio']*100:.1f}%")
    
    print("\n✅ Data analysis complete!")
    print("\nNext steps:")
    print("1. Implement ST-GCN/PoseFormer model architecture")
    print("2. Set up training loop with appropriate loss functions")
    print("3. Implement cross-validation strategy")
    print("4. Create submission generator")
    
    return track1_stats, track2_stats


if __name__ == "__main__":
    stats_t1, stats_t2 = main()
