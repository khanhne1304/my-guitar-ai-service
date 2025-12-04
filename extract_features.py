import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from audioread import exceptions as audioread_exceptions
import librosa
import numpy as np
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")


def _as_file(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    candidate = Path(path_str).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        exe = candidate / "ffmpeg.exe"
        if exe.is_file():
            return exe
    return None


def find_ffmpeg_binary() -> Optional[Path]:
    direct = shutil.which("ffmpeg")
    if direct:
        return Path(direct)

    for env_key in ("FFMPEG_BIN", "FFMPEG_PATH"):
        env_val = _as_file(os.environ.get(env_key))
        if env_val:
            return env_val

    winget_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        for exe in winget_root.glob("**/ffmpeg.exe"):
            return exe

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract model-ready features from a guitar practice audio clip."
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Đường dẫn tới file âm thanh cần trích xuất đặc trưng.",
    )
    parser.add_argument(
        "--config",
        default="config/training_config.yaml",
        help="File YAML xác định danh sách feature_columns để giữ đúng thứ tự.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sample rate dùng để resample audio khi phân tích (mặc định 22050).",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_float(value: Any, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    try:
        num = float(value)
    except Exception:  # pylint: disable=broad-except
        return fallback
    if not np.isfinite(num):
        return fallback
    return num


def compute_pitch_stats(y: np.ndarray, sr: int) -> Tuple[float, float]:
    f0, _, _ = librosa.pyin(
        y,
        sr=sr,
        fmin=librosa.note_to_hz("E2"),
        fmax=librosa.note_to_hz("E6"),
        frame_length=2048,
        hop_length=256,
    )
    if f0 is None or np.all(np.isnan(f0)):
        return 1.2, 0.8
    midi = librosa.hz_to_midi(f0)
    midi = midi[np.isfinite(midi)]
    if midi.size == 0:
        return 1.2, 0.8
    median_pitch = np.median(midi)
    deviations = np.abs(midi - median_pitch)
    mean_err = np.clip(np.mean(deviations), 0.05, 3.0)
    std_err = np.clip(np.std(deviations), 0.02, 2.5)
    return float(mean_err), float(std_err)


def compute_timing_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)

    if onset_frames.size < 2:
        return {
            "mean_timing_offset_ms": 10.0,
            "std_timing_offset_ms": 5.0,
            "onset_density": 0.5,
            "tempo_variation_pct": 5.0,
        }

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    ioi = np.diff(onset_times)
    mean_ioi = np.mean(ioi)
    std_ioi = np.std(ioi)
    median_ioi = np.median(ioi)
    mean_offset_ms = float((mean_ioi - median_ioi) * 1000)
    std_offset_ms = float(std_ioi * 1000)

    duration = librosa.get_duration(y=y, sr=sr)
    onset_density = float(len(onset_times) / max(duration, 1e-3))
    tempo_variation = float(np.clip((std_ioi / max(mean_ioi, 1e-3)) * 100, 0.1, 25.0))

    return {
        "mean_timing_offset_ms": mean_offset_ms,
        "std_timing_offset_ms": std_offset_ms,
        "onset_density": onset_density,
        "tempo_variation_pct": tempo_variation,
    }


def spectral_energy_bands(y: np.ndarray, sr: int) -> Tuple[float, float]:
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    total_energy = np.sum(S)
    if total_energy <= 0:
        return 0.1, 0.1

    high_freq_mask = freqs >= 5000
    low_energy = np.sum(S[~high_freq_mask])
    high_energy = np.sum(S[high_freq_mask])
    high_ratio = float(np.clip(high_energy / total_energy, 0.0, 1.0))
    noise_ratio = float(np.clip((high_energy / max(low_energy, 1e-6)), 0.0, 2.0))

    return high_ratio, noise_ratio


def compute_dynamic_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_norm = rms / max(np.max(rms), 1e-6)
    missing_ratio = float(np.mean(rms_norm < 0.1))

    noise_floor = float(np.percentile(rms_norm, 15))
    signal_level = float(np.percentile(rms_norm, 85))
    snr_db = 20 * np.log10(max(signal_level, 1e-6) / max(noise_floor, 1e-6))
    snr_db = float(np.clip(snr_db, 5.0, 60.0))

    attack_gradients = np.diff(rms_norm)
    attack_smoothness = float(np.clip(1.0 - np.mean(np.abs(attack_gradients)), 0.05, 1.0))

    sustain_blocks = librosa.util.frame(rms_norm, frame_length=8, hop_length=4).mean(axis=0)
    sustain_consistency = float(np.clip(1.0 - np.std(sustain_blocks), 0.05, 1.0))

    return {
        "missing_strings_ratio": missing_ratio,
        "mean_snr_db": snr_db,
        "attack_smoothness": attack_smoothness,
        "sustain_consistency": sustain_consistency,
    }


def load_audio(audio_path: Path, sr: int) -> Tuple[np.ndarray, int]:
    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
        return librosa.util.normalize(y), sr_loaded
    except audioread_exceptions.NoBackendError as err:
        ffmpeg_bin = find_ffmpeg_binary()
        if ffmpeg_bin is None:
            raise RuntimeError(
                "Không thể đọc file audio (thiếu backend). "
                "Cài đặt FFmpeg và đảm bảo ffmpeg có trong PATH hệ thống."
            ) from err

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = Path(tmp_wav.name)

        try:
            cmd = [
                str(ffmpeg_bin),
                "-y",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                str(sr),
                str(tmp_path),
            ]
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            y, sr_loaded = librosa.load(tmp_path, sr=sr, mono=True)
            return librosa.util.normalize(y), sr_loaded
        except subprocess.CalledProcessError as ffmpeg_err:
            raise RuntimeError(
                "FFmpeg không thể chuyển đổi file audio. "
                "Hãy đảm bảo file không bị hỏng và FFmpeg hoạt động chính xác."
            ) from ffmpeg_err
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


def extract_features(audio_path: Path, sr: int) -> Dict[str, float]:
    y, sr = load_audio(audio_path, sr)
    duration = librosa.get_duration(y=y, sr=sr)
    if duration <= 0:
        raise ValueError("File âm thanh rỗng hoặc không thể xác định thời lượng.")

    mean_pitch_error, std_pitch_error = compute_pitch_stats(y, sr)
    timing = compute_timing_features(y, sr)
    buzz_ratio, extra_noise_level = spectral_energy_bands(y, sr)
    dynamics = compute_dynamic_features(y, sr)

    features = {
        "mean_pitch_error_semitones": mean_pitch_error,
        "std_pitch_error_semitones": std_pitch_error,
        "mean_timing_offset_ms": timing["mean_timing_offset_ms"],
        "std_timing_offset_ms": timing["std_timing_offset_ms"],
        "onset_density": timing["onset_density"],
        "tempo_variation_pct": timing["tempo_variation_pct"],
        "buzz_ratio": buzz_ratio,
        "missing_strings_ratio": dynamics["missing_strings_ratio"],
        "extra_noise_level": extra_noise_level,
        "mean_snr_db": dynamics["mean_snr_db"],
        "attack_smoothness": dynamics["attack_smoothness"],
        "sustain_consistency": dynamics["sustain_consistency"],
    }

    return features


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    try:
        cfg = load_config(config_path)
        feature_columns = cfg.get("feature_columns") or []
        if not audio_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file audio: {audio_path}")

        raw_features = extract_features(audio_path, args.sr)
        ordered = {}
        for key in feature_columns:
            ordered[key] = safe_float(raw_features.get(key), 0.0)

        output = {
            "success": True,
            "features": ordered,
            "metadata": {
                "audio_path": str(audio_path),
                "sample_rate": args.sr,
            },
        }
        print(json.dumps(output, ensure_ascii=False))
    except Exception as exc:  # pylint: disable=broad-except
        error_payload = {
            "success": False,
            "error": str(exc),
        }
        print(json.dumps(error_payload, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


