from dataclasses import dataclass
from multiprocessing import Pool
import numpy as np
import librosa
import musdb
import myenv
import utils


@dataclass
class Config:
    sr: int = 44100
    duration: int = 10  # 60秒
    overlap: int = 5
    n_fft: int = 2048
    # hop_length = 512
    # hop_length = 882
    hop_length: int = 860
    n_mels: int = 128


# def method_b(audio, segment_sec=1.0, sr=16000):
#     mel_full = librosa.feature.melspectrogram(y=audio, sr=sr)  # (freq_bins, total_frames)
#     frames_per_segment = int(segment_sec * (sr / 512))  # 假设默认hop_length=512
#     segments = librosa.util.frame(mel_full, frame_length=frames_per_segment, hop_length=frames_per_segment)
#     return segments.transpose(2, 0, 1)  # (num_segments, freq_bins, time_frames)

# def compute_and_split_spectrum(audio_path, segment_sec=60, sr=44100):
#     # 1. 加载完整音频并计算频谱
#     audio, sr = librosa.load(audio_path, sr=sr)
#     stft = librosa.stft(audio, n_fft=2048, hop_length=512)  # (freq_bins, time_frames)

#     # 2. 按时间轴切分频谱（1分钟≈517帧，假设sr=44100, hop_length=512）
#     frames_per_segment = int(segment_sec * sr / 512)
#     num_segments = stft.shape[1] // frames_per_segment
#     segments = np.array_split(stft, num_segments, axis=1)  # List of (freq_bins, frames_per_segment)

#     return segments

# # 示例：输入1分钟切分的复数频谱段
# spectrum_segments = compute_and_split_spectrum("song.wav")


# 示例
# mel_segments = compute_and_split_mel("song.wav", segment_length=60, overlap=30


def save_file(fname, data):
    np.savez_compressed(fname, mel=data)


def read_file(fname):
    loaded = np.load(fname)
    return loaded["mel"]


def process_musdb_track(audio, rate):
    cfg = Config(sr=rate)
    data, _ = utils.compute_and_split_mel(audio, cfg)
    return data


def process_track(track):
    print(f"process {track.name}")
    mixture = track.audio  # shape: (n_samples, 2) [立体声]
    # vocals = track.targets["vocals"].audio

    # 计算伴奏（混合音频 - 人声）
    # instrumental = mixture - vocals
    instrumental = track.targets["accompaniment"].audio
    name = track.name.replace(" - ", "-").replace(" ", "_")

    # 保存伴奏
    # sf.write(f"music/{name}_inst.mp3", instrumental, track.rate)
    # sf.write(f"music/{name}_mix.mp3", mixture, track.rate)

    # print("inst")
    inst_data = process_musdb_track(instrumental, track.rate)
    # inst_name = f"{myenv.OUTDIR}/temp/{name}_inst.npz"
    # print(f"{track.name} inst done")
    # save_file(inst_name, inst_data)

    # print("mix")
    mix_data = process_musdb_track(mixture, track.rate)
    # mix_name = f"{myenv.OUTDIR}/temp/{name}_mix.npz"
    # print(f"{track.name} mix done")
    # save_file(mix_name, mix_data)
    return name, mix_data, inst_data


def run():
    data_id = 0
    with Pool() as pool:
        # pool.map(process_track, (track for track in mus))
        for name, inst_data, mix_data in pool.imap_unordered(
            process_track, (track for track in mus)
        ):
            print(f"process {name}")
            for data in zip(mix_data, inst_data):
                fname = f"{myenv.OUTDIR}/slides/s{data_id}.npz"
                save_file(fname, data)
                print(f"save {fname}")
                data_id += 1


if __name__ == "__main__":
    mus = musdb.DB(root=myenv.INDIR, is_wav=myenv.ISWAV)
    print("mus: ", len(mus))
    run()
