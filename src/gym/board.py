# Note:
#   A board is 7 columns x 6 rows. DO NOT CHANGE!

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass


@dataclass
class Board:
    player: 1 | 2
    board_p1: Tensor
    board_p2: Tensor
    top: Tensor
    winner: Optional[0 | 1 | 2]

    @staticmethod
    def initial() -> Board:
        return Board(
            player=1,
            board_p1=torch.zeros(size=(), dtype=torch.long),
            board_p2=torch.zeros(size=(), dtype=torch.long),
            top=torch.zeros(7, dtype=torch.long),
            winner=None
        )

    @staticmethod
    def horizontal_winner(board, pos) -> bool:
        shift = pos - 3

        if shift < 0:
            masked = (board << -shift) & 0x7f
        else:
            masked = (board >> shift) & 0x7f

        return (
                masked & 0x78 == 0x78 or
                masked & 0x3c == 0x3c or
                masked & 0x1e == 0x1e or
                masked & 0xf == 0xf
        )

    @staticmethod
    def vertical_winner(board, pos) -> bool:
        shift = pos - 21

        if shift < 0:
            masked = (board << -shift) & 0x40810204081
        else:
            masked = (board >> shift) & 0x40810204081

        return (
            masked & 0x40810200000 == 0x40810200000 or
            masked & 0x810204000 == 0x810204000 or
            masked & 0x10204080 == 0x10204080 or
            masked & 0x204081 == 0x204081
        )

    @staticmethod
    def up_right_winner(board, pos) -> bool:
        shift = pos - 24
        if shift < 0:
            masked = (board << -shift) & 0x41041041040
        else:
            masked = (board >> shift) & 0x41041041040

        return (
            masked & 0x41041000000 == 0x41041000000 or
            masked & 0x820820000 == 0x820820000 or
            masked & 0x10410400 == 0x10410400 or
            masked & 0x208208 == 0x208208
        )

    @staticmethod
    def up_left_winner(board, pos) -> bool:
        shift = pos - 24
        if shift < 0:
            masked = (board << -shift) & 0x1010101010101
        else:
            masked = (board >> shift) & 0x1010101010101

        return (
            masked & 0x1010101000000 == 0x1010101000000 or
            masked & 0x20202020000 == 0x20202020000 or
            masked & 0x404040400 == 0x404040400 or
            masked & 0x8080808 == 0x8080808
        )

    @staticmethod
    def check_winner(board, pos) -> bool:
        return (
                Board.horizontal_winner(board, pos) or
                Board.vertical_winner(board, pos) or
                Board.up_right_winner(board, pos) or
                Board.up_left_winner(board, pos)
        )

    def step(self, action: int) -> Board:
        clone_p1 = self.board_p1.clone()
        clone_p2 = self.board_p2.clone()
        clone_top = self.top.clone()

        pos = action + clone_top[action] * 7

        if self.player == 1:
            clone_p1 |= 1 << pos

            winner = self.check_winner(clone_p1, pos)
        else:
            clone_p2 |= 1 << pos
            winner = self.check_winner(clone_p2, pos)

        clone_top[action] += 1

        return Board(
            player=-self.player + 3 if not winner else self.player,
            board_p1=clone_p1,
            board_p2=clone_p2,
            top=clone_top,
            winner=self.player if winner else (0 if (clone_top >= 6).all() else None)
        )

    @property
    def legal_moves(self) -> Tensor:
        return self.top < 6

    def tensor(self) -> Tensor:
        board = torch.zeros(6, 7, dtype=torch.long)

        for col in range(7):
            for row in range(6):
                idx = row * 7 + col

                if self.board_p1 & (1 << idx) > 0:
                    board[5 - row, col] = 1
                elif self.board_p2 & (1 << idx) > 0:
                    board[5 - row, col] = 2

        return board
