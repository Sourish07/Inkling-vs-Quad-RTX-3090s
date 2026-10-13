# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy", "tqdm"]
# ///
"""Plot how often each expert is routed to, one histogram per layer.

``responses/<n>.json`` holds the routed experts as base64 encoded int32 ids,
one row of ``[layers, top_k]`` per forwarded token. The counts are summed over
every token of every response and each layer is saved as
``expert_usage/layer_<n>.png``.

Run with ``uv run plot_expert_usage.py`` from this directory.
"""

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

NUM_LAYERS = 42
NUM_EXPERTS = 256
TOP_K = 6
RESPONSES = Path(__file__).resolve().parent / "responses"
OUT = Path(__file__).resolve().parent / "expert_usage"


def decode(routed_experts: str) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(routed_experts), dtype=np.int32)
    if raw.size % (NUM_LAYERS * TOP_K):
        raise ValueError(
            f"routed_experts has {raw.size} ids, not a multiple of {NUM_LAYERS}x{TOP_K}"
        )
    return raw.reshape(-1, NUM_LAYERS, TOP_K)


def count_experts(files: list[Path]) -> tuple[np.ndarray, int]:
    """Returns ``counts [layers, experts]`` and the number of tokens counted."""
    counts = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)
    tokens = 0
    for file in tqdm(files, desc="counting"):
        saved = json.loads(file.read_text())
        experts = decode(saved["response"]["sglext"]["routed_experts"])
        tokens += len(experts)
        for layer in range(NUM_LAYERS):
            counts[layer] += np.bincount(
                experts[:, layer].ravel(), minlength=NUM_EXPERTS
            )
    return counts, tokens


def plot_layer(layer: int, counts: np.ndarray, tokens: int, num_files: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(np.arange(NUM_EXPERTS), counts, width=1.0)
    # What every expert would get if the router spread the tokens evenly.
    uniform = tokens * TOP_K / NUM_EXPERTS
    ax.axhline(uniform, color="tab:red", linestyle="--", linewidth=1, label="uniform")
    ax.set_xlim(-0.5, NUM_EXPERTS - 0.5)
    ax.set_xlabel("expert id")
    ax.set_ylabel("times routed")
    ax.set_title(
        f"Layer {layer}: {np.count_nonzero(counts)}/{NUM_EXPERTS} experts used "
        f"over {tokens:,} tokens ({num_files} responses)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"layer_{layer:02d}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    files = sorted(RESPONSES.glob("*.json"), key=lambda file: int(file.stem))
    counts, tokens = count_experts(files)

    plotted = 0
    for layer in tqdm(range(NUM_LAYERS), desc="plotting"):
        # Dense layers have no router and report expert 0 in every slot.
        if not counts[layer, 1:].any():
            print(f"layer {layer} is dense, skipping")
            continue
        plot_layer(layer, counts[layer], tokens, len(files))
        plotted += 1
    print(f"plotted {plotted} layers into {OUT}")


if __name__ == "__main__":
    main()
