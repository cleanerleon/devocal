import torch
import os
import numpy as np
import librosa
import torch
import torch.nn as nn

# from transformers import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.amp import autocast, GradScaler
import myenv
from model import VocalRemoverTransformer, config

# def local_position_embedding(segment_length, d_model):
#     position = torch.arange(segment_length).unsqueeze(1)
#     div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#     pe = torch.zeros(segment_length, d_model)
#     pe[:, 0::2] = torch.sin(position * div_term)
#     pe[:, 1::2] = torch.cos(position * div_term)
#     return pe  # shape: (segment_length, d_model)

# segment_frames = 517  # 1分钟音频的帧数（假设sr=44.1k, hop=512）
# local_pe = local_position_embedding(segment_frames, d_model=128)

# freq_pe = local_position_embedding(n_mels, d_model)  # (n_mels, d_model)
# x = x + freq_pe.unsqueeze(1)  # 广播到所有时间帧


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for mixed_mel, accomp_mel in train_loader:
        mixed_mel = mixed_mel.to(device)
        accomp_mel = accomp_mel.to(device)

        optimizer.zero_grad()
        with autocast(myenv.DEV_TYPE):
            outputs = model(mixed_mel)
            loss = criterion(outputs, accomp_mel)
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算
        for mixed_mel, target_mel in val_loader:
            mixed_mel = mixed_mel.to(device)
            target_mel = target_mel.to(device)

            # 前向传播
            with autocast(myenv.DEV_TYPE):
                outputs = model(mixed_mel)
                loss = criterion(outputs, target_mel)

            # 累计损失
            total_loss += loss.item() * mixed_mel.size(0)
            total_samples += mixed_mel.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss


class MUSDBDataset(Dataset):
    def __init__(self, ids):
        self.cnt = len(ids)
        self.ids = ids
        self.data = dict()

    def __len__(self):
        return self.cnt

    # inst_data, mix_data
    def __getitem__(self, idx):
        assert idx < self.cnt
        sid = self.ids[idx]
        key = f's{sid}'
        data = self.data.get(key)
        if data is None:
            fname = f"{myenv.OUTDIR}/slices/{key}.npz"
            loaded = np.load(fname)
            data = loaded["mel"]
            data = torch.FloatTensor(data[0]), torch.FloatTensor(data[1])
            self.data[key] = data
        return data


def create_musdb_loaders(batch_size=16, val_size=0.2):
    # mus = musdb.DB(root=MUSDB_ROOT)
    # all_tracks = (track.name.replace(' - ', '-').replace(' ', '_') for track in mus)
    # 加载MUSDB数据集
    # mus_train = musdb.DB(root=root_dir, subsets="train", split='train')
    # mus_test = musdb.DB(root=root_dir, subsets="train", split='valid')

    # 合并所有track
    # all_tracks = mus_train.tracks + mus_test.tracks

    # 划分训练集和验证集
    train_ids, val_ids = train_test_split(
        range(myenv.SLICE_NUM), test_size=val_size, random_state=42
    )

    # 创建数据集
    train_dataset = MUSDBDataset(train_ids)
    val_dataset = MUSDBDataset(val_ids)

    # 创建DataLoader
    num_workers = os.cpu_count() * 3 // 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,  # 每个worker预加载的batch数
        persistent_workers=True,  # 避免频繁创建/销毁进程
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


# 初始化
device = torch.device(myenv.DEV_TYPE)
model = VocalRemoverTransformer(
    input_dim=128,
    model_dim=config["model_dim"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    dropout=config["dropout"],
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
scaler = GradScaler(myenv.DEV_TYPE)
train_loader, val_loader = create_musdb_loaders(batch_size=config["batch_size"])

# 训练循环
for epoch in range(config["epochs"]):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # 保存模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

torch.save(model.state_dict(), "model.pth")
