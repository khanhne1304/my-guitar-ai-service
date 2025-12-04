# my-guitar-ai-service

Pipeline huấn luyện mô hình đánh giá chất lượng bài tập guitar dựa trên các tiêu chí clip-level và sequence-level do giáo viên định nghĩa.

## Cấu trúc

- `config/training_config.yaml`: tham số huấn luyện (đường dẫn dữ liệu, cột đặc trưng, cột nhãn, seed, tỉ lệ train/val/test…).
- `data/`: chứa dataset ở dạng bảng (CSV). Script huấn luyện tự sinh synthetic dataset mẫu nếu file chưa tồn tại.
- `artifacts/`: nơi lưu model đã train (`regressor.joblib`, `classifier.joblib`) và báo cáo `metrics.json`.
- `train_clip_quality_model.py`: entry point cho huấn luyện.
- `infer_clip_quality.py`: script inference đọc payload JSON (STDIN hoặc `--input`) và trả về điểm số cho từng tiêu chí.
- `requirements.txt`: thư viện Python cần thiết.

## Lược đồ dataset (clip-level)

| Nhóm | Cột | Mô tả |
| --- | --- | --- |
| Đặc trưng tổng hợp | `mean_pitch_error_semitones`, `std_pitch_error_semitones`, `mean_timing_offset_ms`, `std_timing_offset_ms`, `onset_density`, `tempo_variation_pct`, `buzz_ratio`, `missing_strings_ratio`, `extra_noise_level`, `mean_snr_db`, `attack_smoothness`, `sustain_consistency` | Các thống kê rút ra từ phân tích note/chord cấp thấp hoặc đặc trưng phổ âm thanh. |
| Nhãn regression | `pitch_accuracy`, `timing_accuracy`, `timing_stability`, `tempo_deviation_percent`, `chord_cleanliness_score`, `overall_score` | Bám theo các tiêu chí 2.1–2.5 của yêu cầu. |
| Nhãn classification | `level_class` | 0 = cần luyện thêm, 1 = đạt cơ bản, 2 = tốt. |

Sequence-level labels (per-note/per-chord) vẫn được lưu trong hệ thống annotate và có thể được bổ sung khi cần training nâng cao (ví dụ model Transformer dự đoán pitch_correct hay timing_class cho từng note). Trong giai đoạn 1, script này tập trung vào clip-level để dễ gán nhãn và phản hồi nhanh cho người dùng cuối.

## Cách chạy

```bash
cd my-guitar-ai-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_clip_quality_model.py --config config/training_config.yaml
python infer_clip_quality.py --config config/training_config.yaml --input sample.json
```

> **Lưu ý decoder audio**: `extract_features.py` sử dụng `librosa`/`audioread`. Với các định dạng nén như `.webm`/`.weba`, bạn cần cài [FFmpeg](https://ffmpeg.org/download.html) và thêm `ffmpeg.exe` vào `PATH` (hoặc đặt biến môi trường `FFMPEG_BIN`/`FFMPEG_PATH` trỏ tới file thực thi). Script sẽ tự động tìm trong các vị trí phổ biến (ví dụ winget) và báo lỗi hướng dẫn cài đặt nếu không tìm thấy.

Script sẽ:

1. Tải CSV từ `config.dataset_path`. Nếu chưa có, nó sinh synthetic dataset tôn trọng các giới hạn vật lý (ví dụ pitch_accuracy tỉ lệ nghịch với mean_pitch_error).
2. Chia dữ liệu thành train/val/test, chuẩn hóa đặc trưng.
3. Train mô hình regression đa đầu ra (RandomForestRegressor) cho các điểm số clip-level.
4. Train mô hình classification (RandomForestClassifier) cho `level_class`.
5. Ghi metrics (MAE, RMSE, R², Accuracy, macro F1) và snapshot cấu hình vào `artifacts/metrics.json`.
6. Lưu models dùng `joblib` để phục vụ inference.

## Bước tiếp theo

- Thu thập thêm dữ liệu thật bằng pipeline annotate sequence-level -> aggregate.
- Thay RandomForest bằng kiến trúc sâu hơn (Temporal CNN, AST) khi có feature vector trực tiếp từ audio.
- Thêm mô hình sequence-to-sequence để dự đoán `pitch_correct`, `timing_class` cho từng note, sau đó aggregate thành clip-level score để so sánh với model clip-level.