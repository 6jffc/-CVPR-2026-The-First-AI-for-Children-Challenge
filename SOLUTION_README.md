# CVPR 2026 - The First AI for Children Challenge
## Complete Solution Framework

### 📋 Project Overview

This repository contains the complete solution for the **CVPR 2026 Children Gait Visual Analysis Competition** in the CV4CHL Workshop. Our approach uses Spatial-Temporal Graph Convolutional Networks (ST-GCN) to analyze children's gait patterns from 2D keypoint sequences.

---

### 🏆 Competition Tracks

#### Track 1: EVGS Scoring
- **Task**: Predict 34 binary EVGS (Edinburgh Visual Gait Score) metrics (17 items × 2 limbs)
- **Input**: 2D keypoint sequences
- **Output**: Binary predictions (0/1) for each item + total score
- **Metrics**: Accuracy + NRMSE (Normalized Root Mean Square Error)

#### Track 2: Gait Pattern Classification
- **Task**: Classify gait patterns in Bilateral Spastic Cerebral Palsy
- **Classes**: type1, type2, type3, type4, WNL (normal)
- **Metrics**: Accuracy + Macro F1 Score

---

### 📁 Project Structure

```
/workspace/
├── phase1_data_analysis.py       # EDA and data preprocessing
├── phase2_model_architecture.py  # ST-GCN model implementation
├── phase3_training_pipeline.py   # Training loop and submission generator
├── track1_train.json             # Track 1 training labels
├── track2_train.json             # Track 2 training labels
└── outputs/
    └── submission.csv            # Generated submission file
```

---

### 🔧 Installation

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn tqdm
```

---

### 🚀 Quick Start

#### Step 1: Data Analysis
```bash
python phase1_data_analysis.py
```
- Loads and analyzes training data
- Computes class distributions
- Identifies imbalanced classes
- Creates PyTorch datasets

#### Step 2: Model Architecture Test
```bash
python phase2_model_architecture.py
```
- Tests ST-GCN backbone
- Verifies forward pass for both tracks
- Validates loss functions

#### Step 3: Training & Submission
```bash
python phase3_training_pipeline.py
```
- Trains models for both tracks
- Generates submission CSV file
- Saves best model checkpoints

---

### 🧠 Model Architecture

#### ST-GCN Backbone
- **Input**: (batch, seq_len=50, num_joints=17, channels=2)
- **Spatial GCN**: Graph convolutions on skeleton graph (COCO 17 keypoints)
- **Temporal Conv**: 1D convolutions along time dimension
- **Blocks**: 4 ST-GCN blocks with increasing channels (64→128→256→512)
- **Output**: Global features (512-dim) + per-joint features

#### Track 1 Head
- 34 independent binary classification heads (one per EVGS item)
- Multi-task regression head for total scores
- Combined BCE + MSE loss

#### Track 2 Head
- Shared FC layers with limb-specific classification heads
- 5-class softmax for each limb (left/right)
- Focal Loss with class weights for imbalance handling

---

### 📊 Key Insights from EDA

#### Track 1 Findings:
- **94 patients** in training set
- **Imbalanced items**: Items 6, 13, 17 have <12% positive rate
- **Total score range**: 0-14 per limb
- **Mean total score**: ~5.8-6.0

#### Track 2 Findings:
- **22 patients** only (very small dataset!)
- **Severe class imbalance**:
  - WNL and type4: only 2 samples each (4.55%)
  - type2 and type3: ~34-36% of data
- **90.9%** patients have same gait subtype on both sides

---

### 🎯 Training Strategy

#### Handling Class Imbalance (Track 2):
```python
class_weights = {
    'type1': 0.8,
    'type2': 0.59,
    'type3': 0.63,
    'type4': 4.4,  # High weight for rare class
    'WNL': 4.4     # High weight for rare class
}
```

#### Data Augmentation:
- Gaussian noise (std=0.01)
- Random scaling (0.9-1.1)
- Small rotations (-10° to +10°)
- Time masking (10% probability)
- Joint masking (5% probability)

#### Optimization:
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: Cosine Annealing
- **Early Stopping**: Patience=15 epochs
- **Gradient Clipping**: max_norm=1.0

---

### 📈 Evaluation Metrics

#### Track 1 Score:
```
Acc = (1/N) * Σ I[y_i == ŷ_i]
RMSE = sqrt((1/M) * Σ (p_i - g_i)²)
NRMSE = RMSE / 34
S1 = (Acc + (1 - NRMSE)) / 2
```

#### Track 2 Score:
```
F1_k = 2 * P_k * R_k / (P_k + R_k)  # For each class k
F1_macro = (1/5) * Σ F1_k
Acc2 = (1/M) * Σ TP_k
S2 = (Acc2 + F1_macro) / 2
```

#### Final Score:
```
S = (S1 + S2) / 2
```

---

### 📝 Submission Format

```csv
ID,L1,L2,...,L17,R1,R2,...,R17,Total,Left_gait_subtype,Right_gait_subtype
track1-4,0,1,0,...,0,1,0,...,1,-1,-1,-1
track1-5,1,0,1,...,1,0,1,...,0,-1,-1,-1
...
track2-4,-1,-1,...,-1,-1,-1,...,-1,-1,type2,type3
track2-6,-1,-1,...,-1,-1,-1,...,-1,-1,WNL,WNL
...
```

**Note**: 
- Track 1 rows: L1-L17 and R1-R17 are 0/1, Total=-1, gait_subtype=-1
- Track 2 rows: All L/R columns=-1, Total=-1, gait_subtype is string

---

### 🔬 Technical Details

#### Skeleton Graph (COCO 17 keypoints):
```
0: nose          8: right_elbow
1: left_eye      9: left_wrist
2: right_eye    10: right_wrist
3: left_ear     11: left_hip
4: right_ear    12: right_hip
5: left_shoulder 13: left_knee
6: right_shoulder 14: right_knee
7: left_elbow   15: left_ankle
                16: right_ankle
```

#### Adjacency Matrix:
Natural bone connections + self-loops, normalized using D^(-1/2) * A * D^(-1/2)

---

### 📄 Requirements for Final Submission

1. **Technical Report**: 4-page paper using CVPR2026 LaTeX template
2. **Code Repository**: Must be publicly available
3. **Reproducibility**: Results must match local evaluation

---

### 👥 Team Information

**Team Name**: [Your Team Name]  
**Institution**: [Your Institution]  
**Contact**: [Your Email]

---

### 📚 References

1. Yan, S., et al. "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition." AAAI 2018.
2. Edinburgh Visual Gait Score: https://www.gaitscore.org/
3. Cerebral Palsy Gait Patterns: https://orthoinfo.aaos.org/

---

### ⚠️ Important Notes

- Current implementation uses mock keypoints for demonstration
- Replace with actual keypoint sequences from Kaggle dataset for final training
- Ensure code is made public before competition deadline
- Test locally with released test set before final submission

---

**Good luck with the competition! 🎉**
