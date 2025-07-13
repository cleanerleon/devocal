from dataclasses import dataclass
import torch

INDIR = "music"
OUTDIR = "music"
ISWAV = False
SLICE_NUM = 7007
DEV_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEV_TYPE)

@dataclass
class Config:
    sr: int = 44100
    duration: int = 10  # 60秒
    overlap: int = 5
    n_fft: int = 2048
    # hop_length = 512
    hop_length: int = 860
    n_mels: int = 128

    @property
    def fps(self): # frames_per_segment
        return int(self.duration * self.sr / self.hop_length)

    @property
    def fo(self): # frames of overlop
        return int(self.overlap * self.sr / self.hop_length)

    @property
    def melspec_param(self):
        return dict(sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
            