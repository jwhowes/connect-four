import os
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
        self.mcts_probs = torch.concatenate([
            history["mcts_probs"] for history in histories
        ]).to(torch.float32)
        self.winners = torch.concatenate([
            torch.tensor([history["winner"] for _ in range(history["boards"].shape[0])])
            for history in histories
        ]).to(torch.long)

    def __len__(self):
        return self.boards.shape[0]

    def __getitem__(self, idx):
        board = F.one_hot(self.boards[idx], num_classes=3).permute(2, 0, 1).to(torch.float32)
        player = self.players[idx]
        winner = self.winners[idx]

        if player != 1:
            board = board[[0, 2, 1]]

            if winner != 0:
                winner = 3 - winner

        return board, self.mcts_probs[idx], winner
