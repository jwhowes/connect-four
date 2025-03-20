import asyncio
import time
import multiprocessing as mp
from typing import List
from random import random


def model(batch: List[int]) -> List[int]:
    time.sleep(random() * 2.0)
    return [2 * x for x in batch]


class ModelProcess:
    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, batch_size: int = 4):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.batch_size = batch_size

    @staticmethod
    def start(input_queue: mp.Queue, output_queue: mp.Queue, batch_size: int = 4):
        process = ModelProcess(input_queue, output_queue, batch_size)

        asyncio.run(process.run())

    async def run(self):
        loop = asyncio.get_event_loop()

        while True:
            batch = []
            req_ids = []

            for _ in range(self.batch_size):
                try:
                    request_id, data = await loop.run_in_executor(None, self.input_queue.get_nowait)

                    if request_id == -1:  # Terminated by simulation process
                        self.output_queue.put((-1, None))
                        return

                    req_ids.append(request_id)
                    batch.append(data)
                except Exception:
                    break

            if len(batch) > 0:
                pred = model(batch)

                for req_id, p in zip(req_ids, pred):
                    self.output_queue.put((req_id, p))
            else:
                await asyncio.sleep(0.01)


class SimulationProcess:
    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.pending_futures = {}

    @staticmethod
    def start(input_queue: mp.Queue, output_queue: mp.Queue):
        process = SimulationProcess(input_queue, output_queue)

        asyncio.run(process.run())

    async def run(self):
        sims = [
            self.run_sim(i, i) for i in range(10)
        ]

        manager = asyncio.create_task(self.manage_queue())
        await asyncio.gather(*sims)

        self.input_queue.put((-1, None))

        await manager

    async def run_sim(self, request_id: int, value: int):
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending_futures[request_id] = future

        await asyncio.sleep(random() * 2.0)
        self.input_queue.put((request_id, value))
        result = await future

        print(f"Simulation {request_id} received {result}")

    async def manage_queue(self):
        loop = asyncio.get_event_loop()
        while True:
            request_id, output = await loop.run_in_executor(None, self.output_queue.get)
            if request_id == -1:
                return

            future = self.pending_futures.pop(request_id, None)

            if future is not None:
                future.set_result(output)


def main():
    mp.set_start_method("spawn")

    input_queue = mp.Queue()
    output_queue = mp.Queue()

    model_process = mp.Process(target=ModelProcess.start, args=(input_queue, output_queue))
    sim_process = mp.Process(target=SimulationProcess.start, args=(input_queue, output_queue))

    model_process.start()
    sim_process.start()

    model_process.join()
    sim_process.join()


if __name__ == "__main__":
    main()
