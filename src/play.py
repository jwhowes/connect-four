import time
from multiprocessing import Process, Event, Value
from typing import Optional

import torch

from .gym import State
from .search import Search
from .model import BaseModelConfig, BaseModel


class Player:
    def __init__(
            self, model_config: BaseModelConfig, model_path: str, thinking_time: float, temperature: Optional[float],
            computer_first: bool = False
    ):
        self.user_input = Event()

        self.computer_output_request = Event()
        self.computer_output_sent = Event()

        self.action = Value('i', 0)

        self.model_config = model_config
        self.model_path = model_path

        self.thinking_time = thinking_time
        self.temperature = temperature
        self.computer_first = computer_first

    def search_worker(self):
        search = Search()

        model: BaseModel = self.model_config.build_model()
        model.load_state_dict(torch.load(self.model_path, weights_only=True))
        model.eval()
        model.requires_grad_(False)

        player = not self.computer_first
        while search.root.winner is None:
            action = None

            if player and self.user_input.is_set():
                player = False
                action = self.action.value
                self.user_input.clear()

            if not player and self.computer_output_request.is_set():
                player = True
                self.computer_output_request.clear()

                policy = search.policy(self.temperature if self.temperature is not None else 0.1)
                if self.temperature is None:
                    action = policy.argmax()
                else:
                    action = torch.multinomial(policy, 1)[0]

                self.action.value = int(action)
                self.computer_output_sent.set()

            if action is not None:
                search.step(action)

                if search.root.winner is not None:
                    return

            search.search(model)

    def play(self):
        search_worker = Process(target=self.search_worker)
        search_worker.start()

        state = State.initial()

        player = not self.computer_first
        while state.winner() is None:
            state.display()
            if player:
                player = False
                action = int(input("Enter your move: "))
                self.action.value = action
                self.user_input.set()
            else:
                player = True

                time.sleep(self.thinking_time)
                self.computer_output_request.set()

                self.computer_output_sent.wait()
                self.computer_output_sent.clear()
                action = self.action.value

                print(f"Computer move: {int(action)}")

            state = state.step(action)

        state.display()
        winner = state.winner()

        if winner == 0:
            print("It's a tie!")
        elif winner == 1:
            print("Computer wins!" if self.computer_first else "Player wins!")
        else:
            print("Player wins!" if self.computer_first else "Computer wins!")
