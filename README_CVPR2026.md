# 🏆 CVPR 2026 - The First AI for Children Challenge
## Children Gait Visual Analysis Competition - Solution Guide

---

## 📋 Tổng quan

Đây là giải pháp hoàn chỉnh cho cuộc thi **CVPR 2026 Children Gait Visual Analysis Competition** trong workshop CV4CHL. Giải pháp bao gồm:

- **Track 1**: EVGS Scoring - Dự đoán 34 chỉ số nhị phân (17 items × 2 bên chân)
- **Track 2**: Phân loại dáng đi Bại não co cứng hai bên (5 lớp)

---

## 🎯 Kết quả hiện tại

### ✅ File Submission đã được tạo
- **File**: `submission.csv`
- **Kích thước**: 2.7KB
- **Số dòng**: 26 (1 header + 25 samples)
  - Track 1: 16 bệnh nhân (IDs: 4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85)
  - Track 2: 9 bệnh nhân (IDs: 4, 6, 7, 13, 26, 35, 39, 42, 50)

### ⚠️ Lưu ý quan trọng
File submission hiện tại đang sử dụng **dự đoán giả lập (dummy predictions)** vì thiếu dữ liệu keypoints thực tế từ video. Để có kết quả thật sự, bạn cần:

1. Tải dữ liệu keypoints đầy đủ từ Kaggle
2. Huấn luyện mô hình ST-GCN với dữ liệu thật
3. Chạy inference để tạo dự đoán chính xác

---

## 📁 Cấu trúc Project

```
/workspace/
├── cvpr2026_solution.py       # Script chính (Data pipeline + Model + Training)
├── submission.csv             # File submission (định dạng đúng yêu cầu)
├── data/
│   ├── track1_train.json      # Nhãn Track 1 (94 patients)
│   └── track2_train.json      # Nhãn Track 2 (22 patients)
└── README_CVPR2026.md         # File hướng dẫn này
```

---

## 🔧 Kiến trúc Mô hình

### ST-GCN (Spatial-Temporal Graph Convolutional Network)

```
Input: Keypoint Sequences (frames × 17 keypoints × 2D coords)
    ↓
Input Conv (2 → 128 channels)
    ↓
ST-GCN Block 1 (128 → 128)
    ↓
ST-GCN Block 2 (128 → 256)
    ↓
ST-GCN Block 3 (256 → 256)
    ↓
Global Average Pooling
    ↓
┌─────────────────┬─────────────────┐
│   Track 1 Head  │   Track 2 Head  │
│ 34 Binary + Reg │ 5-class Softmax │
└─────────────────┴─────────────────┘
```

### Thông số kỹ thuật
- **Hidden dimension**: 128
- **Num keypoints**: 17 (COCO format)
- **Sequence length**: 60 frames
- **Total parameters**: ~2.8M

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Cài đặt Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install opencv-python
```

### Bước 2: Tải dữ liệu Keypoints (QUAN TRỌNG)

Dữ liệu keypoints chưa được cung cấp trong GitHub link. Bạn cần:

1. Truy cập: https://www.kaggle.com/datasets/stpeteishii/cvpr-2026-children-gait-data
2. Tải toàn bộ dataset (bao gồm video/keypoints)
3. Trích xuất keypoints và lưu dưới dạng `.npy` hoặc `.json`

```python
# Ví dụ: Lưu keypoints cho mỗi patient
import numpy as np
np.save('keypoints/4.npy', keypoints_array)  # Shape: (frames, 17, 2)
```

### Bước 3: Chạy Training với dữ liệu thật

Sửa đổi `cvpr2026_solution.py`:

```python
# Trong hàm main(), thay thế dummy predictions bằng training thật:

# 1. Tạo Dataset
train_ids = [pid for pid in track1_labels.keys() if pid not in test_ids_track1]
train_dataset = GaitDataset(train_ids, track1_labels, track=1, keypoints_dir='keypoints/')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. Khởi tạo model
model = ChildrenGaitModel()

# 3. Train Track 1
model = train_track1(model, train_loader, val_loader, num_epochs=50)

# 4. Train Track 2 (tương tự)
# ...

# 5. Inference trên test set
predictions = inference(model, test_ids, keypoints_dir='keypoints/')
```

### Bước 4: Generate Submission

```bash
python cvpr2026_solution.py
```

File `submission.csv` sẽ được tạo tự động với định dạng:

| ID | L1-L17 | R1-R17 | Total | Left_gait_subtype | Right_gait_subtype |
|----|--------|--------|-------|-------------------|-------------------|
| track1-4 | 0/1 values... | 0/1 values... | 0-34 | -1 | -1 |
| track2-4 | -1 | -1 | -1 | type1 | type2 |

---

## 📊 Chiến lược Cải thiện Điểm số

### Track 1 - EVGS Scoring

1. **Xử lý mất cân bằng lớp**:
   - Items 6, 13, 17 có tỷ lệ positive <12%
   - Sử dụng weighted BCE loss hoặc Focal Loss

2. **Multi-task Learning**:
   - Joint training cho 34 binary tasks
   - Auxiliary loss cho total score regression

3. **Data Augmentation**:
   - Temporal masking
   - Joint noise injection
   - Speed variation

4. **Ensemble**:
   - Train 5 models với different seeds
   - Average predictions

### Track 2 - Gait Classification

1. **Class Imbalance nghiêm trọng**:
   - WNL và type4 chỉ có 2 samples
   - Sử dụng Focal Loss với class weights
   - Oversampling các lớp hiếm

2. **Transfer Learning**:
   - Pre-train trên dataset lớn (Human3.6M, MPII)
   - Fine-tune trên CGPS

3. **Temporal Attention**:
   - Tập trung vào frames quan trọng
   - Learnable temporal weighting

4. **Cross-validation**:
   - GroupKFold theo patient_id
   - Tránh data leakage

---

## 🎯 Metrics & Evaluation

### Track 1 Score
```
Acc = (1/N) * Σ I[y_i == ŷ_i]
RMSE = sqrt((1/M) * Σ (p_i - g_i)²)
NRMSE = RMSE / 34
S1 = (Acc + 1 - NRMSE) / 2
```

### Track 2 Score
```
F1_macro = (1/5) * Σ F1_k
Acc2 = (1/M) * Σ TP_k
S2 = (Acc2 + F1_macro) / 2
```

### Final Score
```
S = (S1 + S2) / 2
```

---

## 📝 Technical Report Requirements

Sau khi hoàn thành training, bạn cần:

1. **Viết báo cáo 4 trang** sử dụng CVPR 2026 LaTeX template
2. **Nội dung bắt buộc**:
   - Method description
   - Architecture details
   - Experimental results
   - Ablation studies
   - Error analysis

3. **Public Code Repository**:
   - Upload code lên GitHub
   - Đảm bảo reproducibility
   - Include requirements.txt

Template LaTeX: https://github.com/CVPR2026/author-kit

---

## ⏰ Timeline đề xuất

| Giai đoạn | Thời gian | Công việc |
|-----------|-----------|-----------|
| Phase 1 | Ngày 1-2 | EDA, Data pipeline |
| Phase 2 | Ngày 3-5 | Model implementation |
| Phase 3 | Ngày 6-10 | Training & Tuning |
| Phase 4 | Ngày 11-12 | Ensemble & Optimization |
| Phase 5 | Ngày 13-14 | Technical Report |
| Submit | Deadline | Final submission |

---

## 🔍 Debugging Tips

### Vấn đề thường gặp:

1. **CUDA Out of Memory**:
   ```python
   CONFIG['batch_size'] = 16  # Giảm batch size
   ```

2. **Overfitting**:
   - Tăng dropout rate
   - Thêm weight decay
   - Early stopping với patience=15

3. **Class imbalance**:
   ```python
   criterion = FocalLoss(alpha=class_weights, gamma=2.0)
   ```

4. **Poor convergence**:
   - Reduce learning rate: 0.001 → 0.0001
   - Use gradient clipping: max_norm=1.0

---

## 📞 Hỗ trợ & Tài nguyên

- **Dataset**: https://www.kaggle.com/datasets/stpeteishii/cvpr-2026-children-gait-data
- **EVGS Guide**: [Link trong description]
- **Gait Patterns**: [Website reference trong description]
- **CVPR Template**: https://github.com/CVPR2026/author-kit

---

## ✅ Checklist trước khi Submit

- [ ] File submission.csv đúng định dạng
- [ ] Track 1: 16 rows với prefix `track1-`
- [ ] Track 2: 9 rows với prefix `track2-`
- [ ] IDs sorted ascending
- [ ] Values: 0/1 cho Track 1, strings cho Track 2
- [ ] -1 cho columns không áp dụng
- [ ] Code repository public
- [ ] Technical report (4 pages) hoàn thành
- [ ] Results reproducible từ code

---

**🎉 Chúc bạn thành công trong cuộc thi CVPR 2026!**

*Mọi thắc mắc về implementation, vui lòng xem code comments trong `cvpr2026_solution.py`.*
