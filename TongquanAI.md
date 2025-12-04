# Tổng quan mô hình AI đánh giá bài tập guitar

## 1. Mục tiêu & Kiến trúc tổng quan
- Dự án `my-guitar-ai-service` cung cấp hai mô hình clip-level để chấm chất lượng bài tập guitar: RandomForest đa đầu ra cho các tiêu chí hồi quy (pitch, timing, tempo, chord, overall) và RandomForest phân loại `level_class` (0: cần luyện thêm, 1: đạt cơ bản, 2: tốt).
- Toàn bộ pipeline gồm các script:
  - `train_clip_quality_model.py`: tải cấu hình, xử lý dữ liệu, huấn luyện, ghi model + metrics.
  - `extract_features.py`: trích xuất đặc trưng từ file audio thật.
  - `infer_clip_quality.py`: inference nhận JSON features và trả về điểm số/level.
- Artifacts (model `.joblib`, `metrics.json`) lưu tại `my-guitar-ai-service/artifacts/`. Dataset CSV ở `data/`.

## 2. Dataset & đặc trưng
- Cấu hình trong `config/training_config.yaml` xác định:
  - `dataset_path`: `data/clip_level_metrics.csv`.
  - `feature_columns`: 12 đặc trưng từ audio (pitch error mean/std, timing offset mean/std, onset density, tempo variation, buzz ratio, missing strings ratio, extra noise level, mean SNR dB, attack smoothness, sustain consistency).
  - `regression_targets`: `pitch_accuracy`, `timing_accuracy`, `timing_stability`, `tempo_deviation_percent`, `chord_cleanliness_score`, `overall_score`.
  - `classification_target`: `level_class`.
- Nếu CSV chưa tồn tại, pipeline tự sinh synthetic dataset (~800 mẫu) dựa trên các phân phối vật lý hợp lý (ví dụ pitch_accuracy giảm khi pitch error tăng, tempo deviation bám sát `tempo_variation_pct`). Điều này cho phép bootstrapping khi dữ liệu annotate thật còn hạn chế.
- README mô tả chi tiết schema và định hướng tích hợp sequence-level labels trong giai đoạn tiếp theo (per-note -> aggregate).

## 3. Quy trình trích xuất đặc trưng (audio thật)
- `extract_features.py` sử dụng `librosa` kết hợp FFmpeg (khi cần) để đọc chuẩn hóa audio.
- Các bước chính:
  - Pitch: dùng `librosa.pyin` để tính mean/std sai lệch cao độ (đơn vị bán cung).
  - Timing: onset strength + onset detect để lấy inter-onset intervals, từ đó tính mean/std timing offset, onset density, tempo variation phần trăm.
  - Spectral/noise: STFT phân tách năng lượng cao tần để ước lượng `buzz_ratio`, `extra_noise_level`.
  - Dynamics: RMS frame-based để tính `missing_strings_ratio`, `mean_snr_db`, `attack_smoothness`, `sustain_consistency`.
- Kết quả được reorder theo `feature_columns` trong cấu hình và xuất JSON, đảm bảo tương thích với pipeline huấn luyện/inference.

## 4. Pipeline huấn luyện
1. **Nạp cấu hình** (`training_config.yaml`) và đảm bảo thư mục artifact.
2. **Tải/sinh dataset**: đọc CSV nếu có, nếu không kích hoạt synthetic generator (seed 42) để tạo dữ liệu mô phỏng hợp lý với giới hạn vật lý (tempo 60–160 BPM, noise floor -55 đến -20 dB, v.v.).
3. **Chia tập dữ liệu**: train/val/test lần lượt 70%/10%/20%, dùng stratify theo `level_class` khi có thể.
4. **Huấn luyện mô hình**:
   - Regressor: `StandardScaler` + `RandomForestRegressor` (400 cây, `min_samples_leaf=2`, `n_jobs=-1`).
   - Classifier: `StandardScaler` + `RandomForestClassifier` (300 cây, `class_weight="balanced"`).
5. **Đánh giá**: tính MAE/RMSE/R² cho từng mục tiêu hồi quy và Accuracy/macro-F1 cho classification trên cả val/test.
6. **Lưu kết quả**: ghi mô hình (`clip_regressor.joblib`, `level_classifier.joblib`) và `metrics.json` (bao gồm snapshot cấu hình) vào `artifacts/`.

## 5. Kết quả hiện tại (dataset synthetic 800 mẫu)
- Regression (test set):
  - `overall_score`: MAE 1.77, RMSE 2.26, R² 0.74.
  - `pitch_accuracy`: MAE ~2.68, R² 0.68.
  - `timing_accuracy`: MAE ~4.47, R² 0.79.
  - `timing_stability`: MAE ~2.50, R² 0.93.
  - `tempo_deviation_percent`: MAE ~1.29, R² 0.15 (khó do phân bố hẹp).
  - `chord_cleanliness_score`: MAE ~3.90, R² 0.49.
- Classification:
  - Accuracy 0.894, macro-F1 0.819 trên tập test.
- Metrics chi tiết nằm trong `artifacts/metrics.json` kèm snapshot cấu hình.

## 6. Inference & tích hợp
- `infer_clip_quality.py` nhận payload JSON dạng:
  ```json
  {
    "features": {
      "mean_pitch_error_semitones": ...,
      "...": ...
    },
    "metadata": { "user_id": "...", "clip_id": "..." }
  }
  ```
- Script nạp cấu hình để đảm bảo đủ cột đặc trưng, gọi các pipeline joblib và trả về:
  ```json
  {
    "success": true,
    "scores": {
      "regression": {
        "pitch_accuracy": ...,
        "...": ...
      },
      "classification": {
        "level_class": 2,
        "probabilities": [0.05, 0.32, 0.63]
      }
    },
    "metadata": { ... }
  }
  ```
- Có thể chạy qua CLI (`python infer_clip_quality.py --config ... --input sample.json`) hoặc đọc STDIN để kết nối với backend Node/Go.

## 7. Hướng phát triển
- Thu thập dữ liệu thật: pipeline annotate sequence-level (per-note pitch_correct, timing_class) và aggregate lên clip-level để thay dataset synthetic, giảm bias mô phỏng.
- Mở rộng mô hình:
  - Thử Gradient Boosting, XGBoost, LightGBM hoặc kiến trúc sâu (Temporal CNN, Audio Spectrogram Transformer) khi đã có đặc trưng trực tiếp từ audio.
  - Triển khai mô hình sequence-to-sequence dự đoán lỗi từng note -> aggregation để đối chiếu với clip-level RandomForest.
- Bổ sung explainability: SHAP/feature importance để giải thích điểm số, giúp giáo viên/learner hiểu yếu tố ảnh hưởng.
- Tích hợp monitoring: log drift của đặc trưng và phân phối nhãn để biết khi nào cần tái huấn luyện.

---

**Cách chạy nhanh**
```bash
cd my-guitar-ai-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_clip_quality_model.py --config config/training_config.yaml
python infer_clip_quality.py --config config/training_config.yaml --input sample.json
```

