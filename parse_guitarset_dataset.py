"""
Script để parse dataset Guitarset và tạo dataset CSV cho training.
Đọc annotations từ file .jams và audio từ file .wav, tính toán các metrics.
"""
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Import extract_features từ cùng thư mục
sys.path.insert(0, str(Path(__file__).parent))
from extract_features import extract_features

warnings.filterwarnings("ignore")


def load_jams_file(jams_path: Path) -> Dict:
    """Đọc file JAMS annotation."""
    with jams_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_annotation_by_namespace(jams_data: Dict, namespace: str) -> Optional[Dict]:
    """Lấy annotation theo namespace."""
    for ann in jams_data.get("annotations", []):
        if ann.get("namespace") == namespace:
            return ann
    return None


def parse_pitch_contour(ann: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Parse pitch contour từ annotation."""
    data = ann.get("data", {})
    times = np.array(data.get("time", []))
    
    # Lấy frequency từ value
    values = data.get("value", [])
    if not values:
        return np.array([]), np.array([])
    
    if isinstance(values[0], dict):
        # Nếu value là dict với frequency field (format Guitarset)
        freqs = []
        times_voiced = []
        for i, v in enumerate(values):
            if isinstance(v, dict) and v.get("voiced", False):
                freq = v.get("frequency", 0)
                if freq > 0 and i < len(times):
                    freqs.append(freq)
                    times_voiced.append(times[i])
        if times_voiced:
            return np.array(times_voiced), np.array(freqs)
    elif isinstance(values[0], (int, float)):
        # Nếu value là frequency trực tiếp
        freqs = np.array(values)
        if len(times) == len(freqs):
            return times, freqs
    
    return np.array([]), np.array([])


def parse_notes(ann: Dict) -> List[Dict]:
    """Parse notes từ note_midi annotation."""
    notes = []
    for item in ann.get("data", []):
        notes.append({
            "time": item.get("time", 0),
            "duration": item.get("duration", 0),
            "midi": item.get("value", 0),
        })
    return sorted(notes, key=lambda x: x["time"])


def parse_tempo(ann: Dict) -> float:
    """Parse tempo từ tempo annotation."""
    data = ann.get("data", [])
    if data:
        return float(data[0].get("value", 120.0))
    return 120.0


def parse_chords(ann: Dict) -> List[Dict]:
    """Parse chords từ chord annotation."""
    chords = []
    for item in ann.get("data", []):
        chords.append({
            "time": item.get("time", 0),
            "duration": item.get("duration", 0),
            "chord": item.get("value", ""),
        })
    return sorted(chords, key=lambda x: x["time"])


def compute_pitch_metrics_from_annotations(
    notes: List[Dict], pitch_contour_times: np.ndarray, pitch_contour_freqs: np.ndarray
) -> Tuple[float, float]:
    """Tính pitch error từ annotations."""
    if len(notes) == 0 or len(pitch_contour_times) == 0:
        return 1.2, 0.8
    
    errors = []
    for note in notes:
        note_time = note["time"]
        note_midi = note["midi"]
        expected_freq = librosa.midi_to_hz(note_midi)
        
        # Tìm pitch contour gần nhất với note time
        time_diff = np.abs(pitch_contour_times - note_time)
        if len(time_diff) > 0:
            idx = np.argmin(time_diff)
            if time_diff[idx] < 0.1:  # Trong vòng 100ms
                actual_freq = pitch_contour_freqs[idx]
                if actual_freq > 0:
                    actual_midi = librosa.hz_to_midi(actual_freq)
                    error = abs(actual_midi - note_midi)
                    errors.append(error)
    
    if len(errors) == 0:
        return 1.2, 0.8
    
    mean_error = float(np.clip(np.mean(errors), 0.05, 3.0))
    std_error = float(np.clip(np.std(errors), 0.02, 2.5))
    return mean_error, std_error


def compute_timing_metrics_from_notes(notes: List[Dict], tempo: float) -> Dict[str, float]:
    """Tính timing metrics từ notes."""
    if len(notes) < 2:
        return {
            "mean_timing_offset_ms": 10.0,
            "std_timing_offset_ms": 5.0,
            "onset_density": 0.5,
            "tempo_variation_pct": 5.0,
        }
    
    # Tính inter-onset intervals
    note_times = [note["time"] for note in notes]
    ioi = np.diff(note_times)
    
    # Expected IOI dựa trên tempo
    expected_ioi = 60.0 / tempo  # seconds per beat
    # Giả sử mỗi note là một beat (có thể điều chỉnh)
    
    offsets = []
    for i in range(len(ioi)):
        # So sánh với expected IOI
        offset = ioi[i] - expected_ioi
        offsets.append(offset)
    
    if len(offsets) == 0:
        return {
            "mean_timing_offset_ms": 10.0,
            "std_timing_offset_ms": 5.0,
            "onset_density": 0.5,
            "tempo_variation_pct": 5.0,
        }
    
    mean_offset_ms = float(np.mean(offsets) * 1000)
    std_offset_ms = float(np.std(offsets) * 1000)
    
    # Onset density
    total_duration = note_times[-1] - note_times[0] if len(note_times) > 1 else 1.0
    onset_density = float(len(notes) / max(total_duration, 1e-3))
    
    # Tempo variation
    tempo_variation = float(np.clip((np.std(ioi) / max(np.mean(ioi), 1e-3)) * 100, 0.1, 25.0))
    
    return {
        "mean_timing_offset_ms": mean_offset_ms,
        "std_timing_offset_ms": std_offset_ms,
        "onset_density": onset_density,
        "tempo_variation_pct": tempo_variation,
    }


def compute_metrics_from_annotations(jams_data: Dict) -> Dict[str, float]:
    """Tính toán các metrics từ annotations."""
    # Parse các annotations
    notes_ann = get_annotation_by_namespace(jams_data, "note_midi")
    pitch_ann = get_annotation_by_namespace(jams_data, "pitch_contour")
    tempo_ann = get_annotation_by_namespace(jams_data, "tempo")
    
    notes = parse_notes(notes_ann) if notes_ann else []
    tempo = parse_tempo(tempo_ann) if tempo_ann else 120.0
    
    # Parse pitch contour
    pitch_times, pitch_freqs = np.array([]), np.array([])
    if pitch_ann:
        pitch_times, pitch_freqs = parse_pitch_contour(pitch_ann)
    
    # Tính pitch metrics
    mean_pitch_error, std_pitch_error = compute_pitch_metrics_from_annotations(
        notes, pitch_times, pitch_freqs
    )
    
    # Tính timing metrics
    timing_metrics = compute_timing_metrics_from_notes(notes, tempo)
    
    return {
        "mean_pitch_error_semitones": mean_pitch_error,
        "std_pitch_error_semitones": std_pitch_error,
        "mean_timing_offset_ms": timing_metrics["mean_timing_offset_ms"],
        "std_timing_offset_ms": timing_metrics["std_timing_offset_ms"],
        "onset_density": timing_metrics["onset_density"],
        "tempo_variation_pct": timing_metrics["tempo_variation_pct"],
        "tempo_bpm": tempo,
    }


def compute_target_scores(features: Dict, tempo_bpm: float) -> Dict[str, float]:
    """Tính toán các target scores từ features."""
    # Pitch accuracy (0-100)
    pitch_accuracy = np.clip(
        100 - (features["mean_pitch_error_semitones"] * 18 + features["std_pitch_error_semitones"] * 8),
        0, 100
    )
    
    # Timing accuracy (0-100)
    timing_accuracy = np.clip(
        100 - (np.abs(features["mean_timing_offset_ms"]) * 0.9 + features["std_timing_offset_ms"] * 0.3),
        0, 100
    )
    
    # Timing stability (0-100)
    timing_stability = np.clip(
        100 - (features["std_timing_offset_ms"] * 0.8),
        0, 100
    )
    
    # Tempo deviation percent
    tempo_deviation_percent = features.get("tempo_variation_pct", 5.0)
    
    # Chord cleanliness (dựa trên buzz ratio và noise)
    chord_cleanliness = np.clip(
        100 - (
            features.get("buzz_ratio", 0.1) * 40 +
            features.get("missing_strings_ratio", 0.1) * 30 +
            features.get("extra_noise_level", 0.1) * 30
        ),
        0, 100
    )
    
    # Overall score
    overall_score = (
        pitch_accuracy * 0.35 +
        timing_accuracy * 0.3 +
        chord_cleanliness * 0.2 +
        (100 - tempo_deviation_percent * 3) * 0.15
    )
    overall_score = np.clip(overall_score / (0.35 + 0.3 + 0.2 + 0.15), 0, 100)
    
    # Level class (0: cần luyện thêm, 1: đạt cơ bản, 2: tốt)
    level_class = int(np.digitize(overall_score, bins=[60, 80]))
    
    return {
        "pitch_accuracy": float(pitch_accuracy),
        "timing_accuracy": float(timing_accuracy),
        "timing_stability": float(timing_stability),
        "tempo_deviation_percent": float(tempo_deviation_percent),
        "chord_cleanliness_score": float(chord_cleanliness),
        "overall_score": float(overall_score),
        "level_class": level_class,
    }


def process_single_file(
    jams_path: Path, audio_path: Path, sr: int = 22050
) -> Optional[Dict]:
    """Xử lý một cặp file annotation và audio."""
    try:
        # Đọc JAMS annotation
        jams_data = load_jams_file(jams_path)
        
        # Tính metrics từ annotations
        annotation_metrics = compute_metrics_from_annotations(jams_data)
        
        # Trích xuất features từ audio
        if not audio_path.exists():
            print(f"[WARN] Audio file không tồn tại: {audio_path}")
            return None
        
        audio_features = extract_features(audio_path, sr)
        
        # Kết hợp features từ annotations và audio
        # Ưu tiên metrics từ annotations nếu có
        combined_features = {
            **audio_features,
            "mean_pitch_error_semitones": annotation_metrics.get(
                "mean_pitch_error_semitones", audio_features.get("mean_pitch_error_semitones", 1.2)
            ),
            "std_pitch_error_semitones": annotation_metrics.get(
                "std_pitch_error_semitones", audio_features.get("std_pitch_error_semitones", 0.8)
            ),
            "mean_timing_offset_ms": annotation_metrics.get(
                "mean_timing_offset_ms", audio_features.get("mean_timing_offset_ms", 10.0)
            ),
            "std_timing_offset_ms": annotation_metrics.get(
                "std_timing_offset_ms", audio_features.get("std_timing_offset_ms", 5.0)
            ),
            "onset_density": annotation_metrics.get(
                "onset_density", audio_features.get("onset_density", 0.5)
            ),
            "tempo_variation_pct": annotation_metrics.get(
                "tempo_variation_pct", audio_features.get("tempo_variation_pct", 5.0)
            ),
        }
        
        # Tính target scores
        targets = compute_target_scores(combined_features, annotation_metrics.get("tempo_bpm", 120.0))
        
        # Kết hợp tất cả
        result = {
            **combined_features,
            **targets,
        }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi xử lý {jams_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_guitarset_dataset(
    annotation_dir: Path, audio_dir: Path, output_csv: Path, sr: int = 22050
) -> None:
    """Xử lý toàn bộ dataset Guitarset."""
    annotation_dir = Path(annotation_dir)
    audio_dir = Path(audio_dir)
    output_csv = Path(output_csv)
    
    # Tìm tất cả file .jams
    jams_files = sorted(annotation_dir.glob("*.jams"))
    print(f"[INFO] Tìm thấy {len(jams_files)} file annotation")
    
    results = []
    
    for jams_path in jams_files:
        # Tìm file audio tương ứng
        audio_filename = jams_path.stem + ".wav"
        audio_path = audio_dir / audio_filename
        
        if not audio_path.exists():
            print(f"[WARN] Không tìm thấy audio cho {jams_path.name}")
            continue
        
        print(f"[INFO] Đang xử lý {jams_path.name}...")
        result = process_single_file(jams_path, audio_path, sr)
        
        if result:
            result["file_id"] = jams_path.stem
            results.append(result)
    
    if len(results) == 0:
        print("[ERROR] Không có dữ liệu nào được xử lý thành công!")
        return
    
    # Tạo DataFrame
    df = pd.DataFrame(results)
    
    # Đảm bảo thứ tự cột đúng với config
    feature_columns = [
        "mean_pitch_error_semitones",
        "std_pitch_error_semitones",
        "mean_timing_offset_ms",
        "std_timing_offset_ms",
        "onset_density",
        "tempo_variation_pct",
        "buzz_ratio",
        "missing_strings_ratio",
        "extra_noise_level",
        "mean_snr_db",
        "attack_smoothness",
        "sustain_consistency",
    ]
    
    target_columns = [
        "pitch_accuracy",
        "timing_accuracy",
        "timing_stability",
        "tempo_deviation_percent",
        "chord_cleanliness_score",
        "overall_score",
        "level_class",
    ]
    
    # Sắp xếp cột
    all_columns = ["file_id"] + feature_columns + target_columns
    existing_columns = [col for col in all_columns if col in df.columns]
    df = df[existing_columns]
    
    # Lưu CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"[INFO] Đã tạo dataset với {len(df)} mẫu tại {output_csv}")
    print(f"[INFO] Thống kê:\n{df.describe()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parse Guitarset dataset và tạo CSV cho training."
    )
    parser.add_argument(
        "--annotation_dir",
        default="GuitarSet/annotation",
        help="Thư mục chứa file .jams",
    )
    parser.add_argument(
        "--audio_dir",
        default="GuitarSet/audio_mono-mic",
        help="Thư mục chứa file .wav",
    )
    parser.add_argument(
        "--output",
        default="data/guitarset_metrics.csv",
        help="Đường dẫn file CSV output",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sample rate để đọc audio",
    )
    
    args = parser.parse_args()
    
    process_guitarset_dataset(
        Path(args.annotation_dir),
        Path(args.audio_dir),
        Path(args.output),
        args.sr,
    )

