from abc import ABC, abstractmethod
from typing import List, Tuple

from tqdm import tqdm

from ..gym import Board


class AbstractMCTS(ABC):
    @abstractmethod
    def __init__(self):
        ...

    @property
    @abstractmethod
    def board(self) -> Board:
        ...

    @property

    @abstractmethod
    def run_sim(self):
        ...

    @abstractmethod
    def step(self, action: int):
        ...

    @abstractmethod
    def get_move(self) -> int:
        ...

    @classmethod
    def self_play(cls, num_simulations: int = 1000) -> List[Tuple[Board, int]]:
        mcts = cls()

        history: List[Tuple[Board, int]] = []
        pbar = tqdm()
        while mcts.board.winner is None:
            for _ in range(num_simulations):
                mcts.run_sim()

            action = mcts.get_move()
            history.append((mcts.board, action))
            mcts.step(action)

            pbar.update()

        return history
