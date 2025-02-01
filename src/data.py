import os
from random import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class GameHistoryDataset(Dataset):
    def __init__(self, data_dir: str, p_flip: float = 0.5):
        self.p_flip = p_flip

        histories = [
            torch.load(os.path.join(data_dir, file), weights_only=True) for file in os.listdir(data_dir)
        ]

        self.boards = torch.concatenate([
            history["boards"] for history in histories
        ]).to(torch.long)
        self.players = torch.concatenate([
            history["players"] for history in histories
        ]).to(torch.long)
        self.qualities = torch.concatenate([
            history["qualities"] for history in histories
        ]).to(torch.float32)
        self.legals = torch.concatenate([
            history["legals"] for history in histories
        ]).to(torch.bool)

    def __len__(self):
        return self.boards.shape[0]

    def __getitem__(self, idx):
        board = F.one_hot(self.boards[idx], num_classes=3).to(torch.float32)
        player = self.players[idx]
        quality = self.qualities[idx]
        legal = self.legals[idx]

        if player != 1:
            board = board[:, :, [0, 2, 1]]

        if random() < self.p_flip:
            board = board.flip(0)
            quality = quality.flip(0)
            legal = legal.flip(0)

        return board, quality, legal
