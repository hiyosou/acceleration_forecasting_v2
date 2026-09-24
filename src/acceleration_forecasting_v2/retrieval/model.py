"""波形埋め込み用のAutoencoder。`acceleration_retrieval/model.py`を無変更で移植。"""

from __future__ import annotations

from torch import nn

from .constants import EMBEDDING_DIM


class WaveformAutoencoder(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.encoder_convs = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
        )
        self.to_embedding = nn.Linear(64 * 63, self.embedding_dim)
        self.from_embedding = nn.Linear(self.embedding_dim, 64 * 63)
        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=2, padding=7, output_padding=1),
        )

    def encode(self, waveform):
        features = self.encoder_convs(waveform)
        return self.to_embedding(features.flatten(start_dim=1))

    def decode(self, embedding):
        features = self.from_embedding(embedding).reshape(-1, 64, 63)
        return self.decoder_convs(features)

    def forward(self, waveform):
        embedding = self.encode(waveform)
        return self.decode(embedding), embedding


def l2_normalize(embedding, eps=1e-12):
    return embedding / embedding.norm(dim=1, keepdim=True).clamp_min(eps)
