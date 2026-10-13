"""Replay the saved responses through an LRU expert cache and count the copies.

Each layer caches ``slots`` of its experts on the GPUs; no expert is resident.
A routed expert that is not cached is a copy from host memory, and it evicts
the least recently used one.
The requests are served in order, one at a time, and the cache carries over
from one request to the next.

Run with ``python measure_policy.py --slots X`` from this directory.
"""

import base64
import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from random import shuffle

import numpy as np
import tyro
from tqdm import tqdm

random.seed(42)

RESPONSES = Path(__file__).resolve().parent / "responses"
NUM_EXPERTS = 256
NUM_LAYERS = 42
TOP_K = 6


def decode(routed_experts: str) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(routed_experts), dtype=np.int32)
    return raw.reshape(-1, NUM_LAYERS, TOP_K)


class LRU:
    """We're not doing one LRU per layer because we'll do a global version eventually"""

    def __init__(self, cache_size: int, resident_experts: list[set[int]] | None = None):
        self.cache_size = cache_size

        self.resident_experts: list[set[int]] = resident_experts or []
        self.cached: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
        """{layer: OrderedDict of cached experts}"""

        self.copy_count = {
            i: {e: 0 for e in range(NUM_EXPERTS) if e not in self.resident_experts[i]}
            for i in range(NUM_LAYERS)
        }
        """{layer: {expert: count}}"""

    def forward(self, layer: int, experts: list[int]):
        """One forward's distinct routed experts; returns how many were copied."""

        lru: OrderedDict[int, None] = self.cached[layer]
        copy_count_dict = self.copy_count[layer]
        missed = []
        for e in experts:
            if e in self.resident_experts[layer]:
                continue

            if e in lru:
                lru.move_to_end(e)
            else:
                missed.append(e)

        for m in missed:
            copy_count_dict[m] += 1
            lru[m] = None

        # Yes, technically lru may exceed max cache size until we pop, but in reality, we'll never evict an expert from LRU that we're just about to use i.e. if we iterate sequentially over experts [3, 5, 8], we won't ever evict 8 to make room for 5. Plus, we can always assume our cache is bigger than the number of routed experts.

        while len(lru) > self.cache_size:
            lru.popitem(last=False)  # FIFO


def main(cache_size: int, num_promote: int = 64, responses: Path = RESPONSES) -> None:
    """
    Args:
        slots: Cached experts per layer.
        responses: Directory of saved responses.
    """
    files = {}
    for file in responses.glob("*.json"):
        response = json.loads(file.read_text())["response"]
        experts = decode(response["sglext"]["routed_experts"])
        files[file] = experts

    resident_experts = [OrderedDict() for _ in range(NUM_LAYERS)]

    for iter in range(NUM_EXPERTS // num_promote):
        lru = LRU(cache_size, resident_experts)

        keys = list(files.keys())
        shuffle(keys)

        for file in tqdm(keys):
            experts = files[file]
            # experts is (seq_len, num_layers, top_k)
            experts = experts[:, 2:, :]  # layers 0 & 1 are dense layers
            seq_len, num_layers, _ = experts.shape

            for seq in range(seq_len):
                for layer in range(num_layers):
                    lru.forward(layer + 2, experts[seq, layer, :].tolist())

        for layer in range(2, num_layers + 2):
            pairs = lru.copy_count[layer].items()
            sorted_pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
            for p in range(num_promote):
                most_copied_expert, count = sorted_pairs[p]
                resident_experts[layer][most_copied_expert] = None

    with open("placement.json", "w") as f:
        rows = [
            f'    "{i}": {json.dumps(list(resident_experts[i]))}'
            for i in range(NUM_LAYERS)
        ]
        f.write("{\n" + ",\n".join(rows) + "\n}\n")


if __name__ == "__main__":
    tyro.cli(main)
