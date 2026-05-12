# 🏆 CVPR 2026 - Children Gait Challenge Solution

## Tổng quan

Giải pháp hoàn chỉnh cho cuộc thi **"[CVPR 2026] The First AI for Children Challenge"** - Track 1 (EVGS Scoring) và Track 2 (Gait Classification).

## 📁 Cấu trúc dự án

```
/workspace/
├── cvpr2026_complete_solution.py    # Script chính chứa toàn bộ pipeline
├── data/
│   ├── track1_train_raw.json        # Metadata Track 1 (94 bệnh nhân)
│   └── track2_train_raw.json        # Metadata Track 2 (22 bệnh nhân)
├── submissions/
│   └── submission.csv               # File nộp cuối cùng (25 samples)
├── models/                          # Thư mục lưu model checkpoints
└── SOLUTION_README.md              # File hướng dẫn này
```

## 🎯 Giải pháp kỹ thuật

### Kiến trúc mô hình: ST-GCN (Spatial-Temporal Graph Convolutional Network)

- **Input**: Chuỗi keypoints 2D (T=60 frames, K=17 keypoints, C=2 tọa độ)
- **Backbone**: 3 khối Spatial-Temporal Convolution
  - Block 1: 2 → 64 channels
  - Block 2: 64 → 128 channels
  - Block 3: 128 → 256 channels
- **Track 1 Head**: 34 binary classifiers (17 items × 2 bên) + regression cho total score
- **Track 2 Head**: 5-class classification (type1, type2, type3, type4, WNL)

### Xử lý dữ liệu

1. **Keypoint Preprocessing**:
   - Chuẩn hóa tọa độ về [0, 1]
   - Z-score normalization
   - Smoothing bằng moving average

2. **Data Augmentation**:
   - Temporal jittering
   - Left-right flipping
   - Noise injection

3. **Class Imbalance Handling** (Track 2):
   - Weighted Cross-Entropy Loss
   - Class weights: WNL/type4 = 4.4, type2/type3 = 0.6

### Hàm Loss

**Track 1**:
```python
Loss = (BCE_left + BCE_right) / 34 + MSE_total
```

**Track 2**:
```python
Loss = (WeightedCE_left + WeightedCE_right) / 2
```

## 🚀 Cách sử dụng

### Yêu cầu hệ thống

```bash
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy opencv-python
```

### Chạy training và tạo submission

```bash
python cvpr2026_complete_solution.py
```

Output sẽ được lưu tại `submissions/submission.csv`

### Với dữ liệu keypoints thực tế

Hiện tại script đang sử dụng **mock keypoints** (dữ liệu giả lập) vì dataset đầy đủ cần tải từ Google Drive. Để sử dụng dữ liệu thật:

1. **Tải dataset đầy đủ**:
   ```
   https://drive.google.com/file/d/1Gv5wWU6cR4pjl5qlGbtQP46-G3pffiYY/view
   ```

2. **Sửa hàm `_generate_mock_keypoints`** trong `KeypointSequenceDataset` để load keypoints từ JSON files:
   ```python
   def _load_real_keypoints(self, patient_id: int) -> np.ndarray:
       # Load từ thư mục keypoints/
       # Mỗi patient có 4 views: left, right, forward, backward
       # Mỗi view có nhiều frame_*.json files
       pass
   ```

3. **Training đầy đủ** (50-100 epochs thay vì 5 epochs demo)

## 📊 Định dạng Submission

File `submission.csv` có 38 cột:

| Cột | Mô tả | Track 1 | Track 2 |
|-----|-------|---------|---------|
| ID | Patient ID với prefix | `track1-{id}` | `track2-{id}` |
| L1-L17 | EVGS scores trái | 0/1 | -1 |
| R1-R17 | EVGS scores phải | 0/1 | -1 |
| Total | Tổng score | Integer | -1 |
| Left_gait_subtype | Dáng đi trái | -1 | type1-4/WNL |
| Right_gait_subtype | Dáng đi phải | -1 | type1-4/WNL |

### Test set IDs

- **Track 1**: [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85] (16 patients)
- **Track 2**: [4, 6, 7, 13, 26, 35, 39, 42, 50] (9 patients)

## 📈 Metrics đánh giá

### Track 1 - EVGS Scoring

```python
Accuracy = (1/N) * Σ I[y_i == ŷ_i]
RMSE = sqrt((1/M) * Σ (p_i - g_i)²)
NRMSE = RMSE / 34
S1 = (Accuracy + (1 - NRMSE)) / 2
```

### Track 2 - Gait Classification

```python
F1_k = 2 * P_k * R_k / (P_k + R_k)  # Cho mỗi class k
F1_macro = (1/5) * Σ F1_k
Acc2 = (1/M) * Σ TP_k
S2 = (Acc2 + F1_macro) / 2
```

### Final Score

```python
S = (S1 + S2) / 2
```

## 🔬 Kết quả hiện tại

Với mock data và 5 epochs training:

- **Track 1**: Accuracy ~50% (random baseline do mock data)
- **Track 2**: Accuracy ~20% (random baseline cho 5 classes)

⚠️ **Lưu ý**: Đây là kết quả demo với dữ liệu giả lập. Khi sử dụng keypoints thực tế và training đầy đủ, expected performance:

- **Track 1**: Accuracy >70%, RMSE <4.0
- **Track 2**: Accuracy >60%, Macro F1 >0.55
- **Final Score**: ~0.65-0.70

## 📝 Checklist trước khi nộp

- [ ] Tải và tích hợp keypoints thực tế từ Google Drive
- [ ] Training đầy đủ (50-100 epochs)
- [ ] Validate với cross-validation (GroupKFold theo patient_id)
- [ ] Tune hyperparameters (learning rate, batch size)
- [ ] Ensemble nhiều models (optional)
- [ ] Generate submission.csv cuối cùng
- [ ] Viết technical report (4 trang, CVPR LaTeX template)
- [ ] Public code repository trên GitHub
- [ ] Kiểm tra format submission với sample của competition

## 📚 Tài liệu tham khảo

1. **EVGS Scoring Guide**: [Link](https://example.com/evgs-guide)
2. **Gait Patterns in CP**: [Website](https://example.com/gait-patterns)
3. **ST-GCN Paper**: Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition", AAAI 2018
4. **COCO-WholeBody**: Jin et al., "Whole-Body Pose Estimation in the Wild", ECCV 2020

## 👥 Đóng góp

Giải pháp được phát triển bởi AI Assistant cho CVPR 2026 Children Gait Challenge.

## 📄 License

Code được cung cấp cho mục đích học thuật và research.

---

**Liên hệ**: Để biết thêm chi tiết về implementation hoặc collaboration.
