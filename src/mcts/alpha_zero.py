from __future__ import annotations

import asyncio
import multiprocessing as mp
from typing import List, Optional
from math import sqrt

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm

from ..model import ModelConfig
from ..gym import Board

EXPLORATION_COEFF: float = sqrt(2.0)
VIRTUAL_LOSS: int = 3


# TODO
#   - Create a self_play function which generates a game history
#   - Make the model process remain between self play iterations (so we don't have to rebuild the model every time)


class AlphaZeroNode:
    def __init__(self, player: 1 | 2, parent: Optional[AlphaZeroNode] = None, action: Optional[int] = None):
        self.player = player
        self.parent = parent
        self.action = action

        self.total_value = torch.zeros(7, dtype=torch.float32)
        self.num_visits = torch.zeros(7, dtype=torch.long)

        self.children: List[Optional[AlphaZeroNode]] = [
            None for _ in range(7)
        ]
        self.expanded = torch.zeros(7, dtype=torch.bool)


class ModelProcess:
    def __init__(
            self, config: ModelConfig, input_queue: mp.Queue, output_queue: mp.Queue,
            batch_size: int = 4, device: torch.device = torch.device("cpu")
    ):
        self.model = config.build()

        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.device = device
        self.batch_size = batch_size

    @staticmethod
    def start(
            config: ModelConfig, input_queue: mp.Queue, output_queue: mp.Queue,
            batch_size: int = 4, device: torch.device = torch.device("cpu")
    ):
        process = ModelProcess(config, input_queue, output_queue, batch_size, device)

        asyncio.run(process.run())

    async def run(self):
        loop = asyncio.get_event_loop()

        while True:
            batch: List[Tensor] = []
            req_ids: List[int] = []

            for _ in range(self.batch_size):
                try:
                    req_id, board_p1, board_p2 = await loop.run_in_executor(None, self.input_queue.get_nowait)

                    if req_id == -1:
                        self.output_queue.put((-1, None))
                        return

                    req_ids.append(req_id)
                    batch.append(Board.to_tensor(board_p1, board_p2))
                except Exception:
                    break

            if len(batch) > 0:
                pred = self.model(torch.stack(batch).to(self.device))

                for req_id, p in zip(req_ids, pred):
                    self.output_queue.put((req_id, p.item()))
            else:
                await asyncio.sleep(0.01)


class SimulationProcess:
    def __init__(
            self, input_queue: mp.Queue, output_queue: mp.Queue, max_concurrent: int = 16, pbar: Optional[tqdm] = None
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.node_visits: int = 1
        self.board: Board = Board.initial()
        self.root = AlphaZeroNode(player=1)

        self.pending_futures = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)

        self.pbar = pbar

    @staticmethod
    def start(input_queue: mp.Queue, output_queue: mp.Queue, max_concurrent: int = 16, num_simulations: int = 50000):
        process = SimulationProcess(input_queue, output_queue, max_concurrent, pbar=tqdm(total=num_simulations))

        asyncio.run(process.run(num_simulations))

    async def run(self, num_simulations: int = 1000):
        manager = asyncio.create_task(self.manage_queue())

        await asyncio.gather(*[
            self.run_sim(req_id) for req_id in range(num_simulations)
        ])

        self.input_queue.put((-1, None, None))

        await manager

    async def evaluate(self, req_id: int, board: Board):
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending_futures[req_id] = future

        self.input_queue.put((req_id, board.board_p1, board.board_p2))

        return await future

    async def run_sim(self, req_id: int):
        async with self.semaphore:
            board = self.board.clone()

            node_visits = self.node_visits
            node = self.root

            while node.expanded.all():
                value = (
                    (node.total_value / node.num_visits) +
                    EXPLORATION_COEFF * torch.sqrt(np.log(node_visits) / node.num_visits)
                )
                value[~board.legal_moves] = float('-inf')

                child_idx = value.argmax().item()

                node_visits = node.num_visits[child_idx]

                node.num_visits[child_idx] += VIRTUAL_LOSS
                node.total_value[child_idx] -= VIRTUAL_LOSS

                node = node.children[child_idx]

            child_idx = np.random.choice(torch.where(~node.expanded)[0])
            board = board.step(child_idx)

            node.expanded[child_idx] = True
            node.children[child_idx] = AlphaZeroNode(player=board.player, parent=node, action=child_idx)

            node.num_visits[child_idx] += VIRTUAL_LOSS
            node.total_value[child_idx] -= VIRTUAL_LOSS

            pred_value = await self.evaluate(req_id, board)

            node.num_visits[child_idx] -= (VIRTUAL_LOSS - 1)
            node.total_value[child_idx] += VIRTUAL_LOSS + pred_value

            child_idx = node.action
            node = node.parent
            while node is not None:
                node.num_visits[child_idx] -= (VIRTUAL_LOSS - 1)
                node.total_value[child_idx] += VIRTUAL_LOSS + pred_value

                child_idx = node.action
                node = node.parent

            if self.pbar is not None:
                self.pbar.update()

    async def manage_queue(self):
        loop = asyncio.get_event_loop()

        while True:
            req_id, pred_value = await loop.run_in_executor(None, self.output_queue.get)

            if req_id == -1:
                return

            future = self.pending_futures.pop(req_id, None)

            if future is not None:
                future.set_result(pred_value)


def main():
    mp.set_start_method("spawn")

    input_queue = mp.Queue()
    output_queue = mp.Queue()

    config = ModelConfig.from_yaml("vit.yaml")

    model_process = mp.Process(target=ModelProcess.start, args=(config, input_queue, output_queue))
    sim_process = mp.Process(target=SimulationProcess.start, args=(input_queue, output_queue))

    model_process.start()
    sim_process.start()

    model_process.join()
    sim_process.join()


if __name__ == "__main__":
    main()
