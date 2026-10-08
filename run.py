import time

import torch
import tyro
from huggingface_hub import snapshot_download
from loguru import logger
from torch.distributed import get_world_size
from torch.profiler import record_function
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    TextStreamer,
)

from my_inkling import MyInkling, MyInklingCache, apply_ep_plan, apply_tp_plan
from my_inkling.decode_graph import DecodeGraph
from utils import (
    Profiler,
    get_device_mesh,
    load_config,
    load_expert_state_dict,
    load_non_expert_state_dict,
    setup_ddp_local,
    setup_rank_aware_logger,
)

hf_repo = "thinkingmachines/Inkling-Small-NVFP4"


def main(
    profile: bool = False,
    log2_max_new_tokens: int = 7,
    gpu_experts_per_rank: int = 24,
    cuda_graph: bool = True,
    batch_size: int = 1,
) -> None:
    """Generate from the sharded Inkling model and report decode throughput.

    Args:
        profile: Record a torch profiler trace of generation on rank 0.
        log2_max_new_tokens: Generate at most 2**log2_max_new_tokens tokens.
        gpu_experts_per_rank: First N experts in each rank's shard stay on GPU; the
            rest use pinned CPU RAM.
        cuda_graph: Replay decode steps from a captured CUDA graph; disable to run
            them eagerly.
    """
    device, local_rank = setup_ddp_local()
    world_size = get_world_size()
    setup_rank_aware_logger()
    device_mesh = get_device_mesh(num_dimensions=1)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        hf_repo, trust_remote_code=True
    )
    local_hf_path = snapshot_download(repo_id=hf_repo, local_files_only=True)
    config = load_config(local_hf_path)

    with torch.device("meta"):
        model = MyInkling(config).bfloat16()
        model.restore_fp32()
        if world_size > 1:
            apply_tp_plan(model, config, device_mesh)

    logger.info("Loading state dict into sharded model")
    non_expert_state_dict = load_non_expert_state_dict(
        model, local_hf_path, device_mesh
    )
    missing, unexpected = model.load_state_dict(
        non_expert_state_dict, strict=False, assign=True
    )
    assert not unexpected and all(".mlp.experts." in name for name in missing)
    del non_expert_state_dict
    model.fuse_attention_projections()

    expert_state_dict = load_expert_state_dict(
        model, local_hf_path, device_mesh, gpu_experts_per_rank
    )
    apply_ep_plan(model, device_mesh, expert_state_dict, num_slots=10)

    torch.distributed.barrier()
    logger.info(f"Inkling model loaded on device {device}")

    with open("prompts.txt", "r") as f:
        prompt_text = f.readlines()[:batch_size]
    prompt = [[{"role": "user", "content": txt.strip()}] for txt in prompt_text]

    logger.info("Running tokenizer...")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(config.eos_token_id)

    inputs: BatchEncoding = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="none",
        padding=True,
    )
    seq_lens = inputs["attention_mask"].sum(dim=1)
    inputs = inputs.to(device)
    num_tokens = inputs["input_ids"].shape[1]
    logger.info(f"running generation with {num_tokens} input tokens")

    model.eval()
    max_new_tokens = 2**log2_max_new_tokens
    streamer = (
        TextStreamer(tokenizer, skip_special_tokens=True) if local_rank == 0 else None
    )
    next_input = inputs["input_ids"]

    # Step 0 is prefill (+ first token); the decode clock starts after it.
    decode_start = 0.0
    num_decode_tokens = 0
    cache = MyInklingCache(config.text_config, batch_size=batch_size)
    decode_graph = None

    profiler = Profiler(enable=profile)
    profiler.start()

    with torch.inference_mode():
        for step in range(max_new_tokens):
            # Names each forward pass in the profiler trace.
            label = "prefill" if step == 0 else f"decode_{step}"
            with record_function(label):
                if step == 0:
                    logits = model(next_input, cache=cache)
                    next_input = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    torch.distributed.broadcast(next_input, src=0)
                else:
                    assert decode_graph is not None
                    next_input = decode_graph.replay()

            host_token = next_input.cpu()
            if streamer is not None:
                streamer.put(host_token)
            is_eos = all(host_token.item() == tokenizer.eos_token_id for _ in range(host_token.shape[0]))

            if step == 0:
                if not is_eos and max_new_tokens > 1:
                    logger.info(
                        "Warming up and capturing complete decode CUDA graph"
                        if cuda_graph
                        else "Warming up eager decode (CUDA graph disabled)"
                    )
                    decode_graph = DecodeGraph(
                        model,
                        cache,
                        next_input,
                        capacity=num_tokens + max_new_tokens - 1,
                        capture=cuda_graph,
                    )
                    decode_start = time.perf_counter()
            else:
                num_decode_tokens += 1

            if is_eos:
                break

    if streamer is not None:
        streamer.end()
    decode_time = time.perf_counter() - decode_start
    profiler.stop()
    profiler.export_chrome_trace("trace.json.gz")
    if local_rank == 0 and num_decode_tokens > 0:
        logger.info(
            f"Decode: {num_decode_tokens} tokens in {decode_time:.2f}s "
            f"= {num_decode_tokens / decode_time:.2f} tok/s (excl. prefill)"
        )

    # NCCL communicator shutdown waits for every captured graph to be released.
    if decode_graph is not None and decode_graph.graph is not None:
        torch.cuda.synchronize()
        decode_graph.graph.reset()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    tyro.cli(main)
