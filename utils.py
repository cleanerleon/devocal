import numpy as np
import librosa

def compute_and_split_mel(audio, config):
    # 1. 计算完整梅尔频谱
    # segment_length = cfg.duration
    # overlap = cfg.overlap
    # sr = cfg.sr
    # n_fft = cfg.n_fft
    # hop_length = cfg.hop_length
    # n_mels = cfg.n_mels
    last_strat = 0

    mono_audio = librosa.to_mono(audio.T)
    # audio, sr = librosa.load(audio_path, sr=sr)
    # duration = librosa.get_duration(y=mono_audio, sr=sr)
    mel_spec = librosa.feature.melspectrogram(
        # y=mono_audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        y=mono_audio, **config.melspec_param
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)  # (n_mels, total_frames)

    # 2. 按时间轴切分（单位：帧）
    frames_per_segment = config.fps
    frames_overlap = config.fo
    segments = []

    frames = mel_db.shape[1]
    end = frames

    for start in range(0, frames - frames_per_segment + 1, frames_overlap):
        end = start + frames_per_segment
        segment = mel_db[:, start:end]
        segments.append(segment)

    if end < frames:
        end = frames
        start = end - frames_per_segment
        segment = mel_db[:, start:end]
        segments.append(segment)
        last_start = start

    res = np.stack(segments)  # (num_segments, n_mels, frames_per_segment)
    return res, frames, last_start
