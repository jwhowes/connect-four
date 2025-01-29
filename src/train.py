from __future__ import annotations

import time
import torch
import torch.nn.functional as F
import os

from multiprocessing import Value, Lock, Process
from torch import FloatTensor
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader

from .mcts import MCTS
from .model import ConvModel, ConvModelConfig, BaseModel
from .util import Config, timestamp
from .data import GameHistoryDataset


@dataclass
class TrainConfig(Config):
    queue_size: int = 16

    sims_per_move: int = 250
    temperature: float = 1.0

    batch_size: int = 16
    lr: float = 5e-5


class Trainer:
    def __init__(
            self, queue_size: int, batch_size: int, lr: float, sims_per_move: int, temperature: float,
            data_dir: str, model_dir: str, model_config: ConvModelConfig, resume: bool = False, data_workers: int = 4
    ):
        self.queue_size = queue_size

        self.sims_per_move = sims_per_move
        self.temperature = temperature

        self.batch_size = batch_size
        self.lr = lr

        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_config = model_config

        self.data_workers = data_workers

        self.model_lock = Lock()
        self.data_lock = Lock()

        self.shared_memory = ValueError

        self.timestamp = None
        if resume:
            for file in os.listdir(self.model_dir):
                if os.path.splitext(file)[1] == ".pt":
                    self.timestamp = Value('i', int(file))
                    break

            assert self.timestamp is not None, "No model found."
        else:
            model = ConvModel.from_config(self.model_config)

            to_remove = []
            for file in os.listdir(self.model_dir):
                if os.path.splitext(file)[1] == ".pt":
                    to_remove.append(os.path.join(self.model_dir, file))

            for file in to_remove:
                os.remove(file)

            for file in os.listdir(self.data_dir):
                os.remove(os.path.join(self.data_dir, file))

            self.timestamp = Value('i', timestamp())
            torch.save(
                model.state_dict(),
                os.path.join(self.model_dir, f"{self.timestamp.value}.pt")
            )

    def load_model(self, model: BaseModel):
        model.load_state_dict(torch.load(os.path.join(self.model_dir, f"{self.timestamp.value}.pt"), weights_only=True))

    def save_model(self, model: BaseModel):
        os.remove(os.path.join(self.model_dir, f"{self.timestamp.value}.pt"))

        self.timestamp.value = timestamp()
        torch.save(model.state_dict(), os.path.join(self.model_dir, f"{self.timestamp.value}.pt"))

    def train(self):
        train_worker = Process(target=self.train_worker)
        train_worker.start()

        data_workers = [
            Process(target=self.data_worker) for _ in range(self.data_workers)
        ]

        for worker in data_workers:
            worker.start()

        train_worker.join()

    def data_worker(self):
        model = ConvModel.from_config(self.model_config)
        model.eval()

        queue = [
            os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir)
        ]

        data_timestamp = 0
        while True:
            with self.model_lock:
                if data_timestamp != self.timestamp.value:
                    self.load_model(model)
                    data_timestamp = self.timestamp.value

            history = MCTS.self_play(self.sims_per_move, model, self.temperature)

            with self.data_lock:
                if len(queue) >= self.queue_size:
                    os.remove(queue.pop())

                queue = [os.path.join(self.data_dir, f"{timestamp()}.hist")] + queue
                history.save(queue[0])

    @staticmethod
    def loss(pred_winner, prior, winner, mcts_prob) -> FloatTensor:
        return (
            F.cross_entropy(pred_winner, winner) +
            -(mcts_prob * F.log_softmax(prior, dim=-1)).sum(-1).mean()
        )

    def train_worker(self):
        model = ConvModel.from_config(self.model_config)
        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        with self.model_lock:
            self.load_model(model)

        while True:
            if len(os.listdir(self.data_dir)) == 0:
                time.sleep(1)
                continue

            with self.data_lock:
                dataset = GameHistoryDataset(self.data_dir)

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True
            )

            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            total_loss = 0
            for i, (board, mcts_prob, winner) in pbar:
                opt.zero_grad()

                pred_winner, prior = model(board)
                loss = self.loss(pred_winner, prior, winner, mcts_prob)

                total_loss += loss.item()

                loss.backward()
                opt.step()

                pbar.set_description(f"Average Loss: {total_loss / (i + 1):.4f}")

            with self.model_lock:
                self.save_model(model)

            time.sleep(1)
