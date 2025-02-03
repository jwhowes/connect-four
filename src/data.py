import os
from random import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class GameHistoryDataset(Dataset):
    def __init__(self, data_dir: str):
        histories = [
            torch.load(os.path.join(data_dir, file), weights_only=True) for file in os.listdir(data_dir)
        ]

        self.boards = torch.concatenate([
            history["boards"] for history in histories
        ]).to(torch.long)
        self.players = torch.concatenate([
            history["players"] for history in histories
        ]).to(torch.long)
        self.priors = torch.concatenate([
            history["priors"] for history in histories
        ]).to(torch.float32)
        self.winners = torch.concatenate([
            torch.tensor([history["winner"] for _ in range(history["boards"].shape[0])], dtype=torch.long)
            for history in histories
        ])

    def __len__(self):
        return self.boards.shape[0]

    def __getitem__(self, idx):
        board = F.one_hot(self.boards[idx], num_classes=3).to(torch.float32)
        player = self.players[idx]
        winner = self.winners[idx]
        prior = self.priors[idx]

        if player != 1:
            board = board[:, :, [0, 2, 1]]
            if winner != 0:
                winner = 3 - winner

        if random() < 0.5:
            board = board.flip(0)
            prior = prior.flip(0)

        return board, winner, prior
