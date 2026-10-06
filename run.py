import time
from concurrent.futures import ThreadPoolExecutor

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
    gpu_experts_per_rank: int | None = None,
    bs: int = 1,
    prompts: list[str] | None = None,
) -> None:
    """Generate from the sharded Inkling model and report decode throughput.

    Args:
        profile: Record a torch profiler trace of generation on rank 0.
        log2_max_new_tokens: Generate at most 2**log2_max_new_tokens tokens.
        gpu_experts_per_rank: First N experts in each rank's shard stay on GPU;
            the rest use pinned CPU RAM. Auto: 24 at bs=1, decreasing to 20 at bs=16
            to leave room for batched KV buffers and CUDA graph state.
        bs: Number of requests decoded together, from 1 through 16.
        prompts: One user prompt per request. Defaults to repeating the demo prompt.
    """
    if not 1 <= bs <= 16:
        raise ValueError("bs must be between 1 and 16")
    if prompts is not None and len(prompts) != bs:
        raise ValueError("provide exactly bs prompts")
    if log2_max_new_tokens < 0:
        raise ValueError("log2_max_new_tokens must be nonnegative")
    if gpu_experts_per_rank is None:
        gpu_experts_per_rank = 24 - (bs - 1 + 3) // 4
    device, local_rank = setup_ddp_local()
    world_size = get_world_size()
    setup_rank_aware_logger()
    logger.info(
        f"bs={bs}, resident GPU experts per rank={gpu_experts_per_rank}, cache slots=10"
    )
    device_mesh = get_device_mesh(num_dimensions=1)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        hf_repo, trust_remote_code=True
    )
    local_hf_path = snapshot_download(repo_id=hf_repo, local_files_only=True)
    config = load_config(local_hf_path)
    # Small's stop ID lives on the multimodal config, while the text config and
    # tokenizer may retain defaults or omit their special-token metadata.
    eos_ids = config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if eos_ids is None:
        eos_ids = config.text_config.eos_token_id
    eos_ids = (
        [] if eos_ids is None else ([eos_ids] if isinstance(eos_ids, int) else eos_ids)
    )

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

    if prompts is None:
        prompts = [
            "What are the computational benefits of Mixture-of-Experts models?"
        ] * bs
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        pad_id = config.text_config.pad_token_id
        if pad_id is None:
            if not eos_ids:
                raise ValueError("batched tokenization requires a pad or EOS token ID")
            pad_id = eos_ids[0]
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_id)

    logger.info("Running tokenizer...")
    inputs: BatchEncoding = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="none",
        padding=True,
    )
    inputs = inputs.to(device)
    num_tokens = inputs["input_ids"].shape[1]
    logger.info(f"running bs={bs} generation with padded prompt length {num_tokens}")

    model.eval()
    max_new_tokens = 2**log2_max_new_tokens
    streamer = (
        TextStreamer(tokenizer, skip_special_tokens=True)
        if local_rank == 0 and bs == 1
        else None
    )
    output_pool = ThreadPoolExecutor(max_workers=1) if streamer is not None else None
    output_future = None
    next_input = inputs["input_ids"]

    # Step 0 is prefill (+ first token); the decode clock starts after it.
    decode_start = 0.0
    num_decode_tokens = 0
    cache = MyInklingCache(
        config.text_config,
        left_padding=num_tokens - inputs["attention_mask"].sum(dim=1),
    )
    decode_graph = None
    finished = [False] * bs
    generated = [[] for _ in range(bs)]

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
            active_count = sum(not done for done in finished)
            for request, token in enumerate(host_token[:, 0].tolist()):
                if not finished[request]:
                    generated[request].append(token)
                    finished[request] = token in eos_ids
            if streamer is not None:
                assert output_pool is not None
                if output_future is not None:
                    output_future.result()
                output_future = output_pool.submit(streamer.put, host_token)
            all_finished = all(finished)
            if step == 0:
                if not all_finished and max_new_tokens > 1:
                    logger.info("Warming up and capturing complete decode CUDA graph")
                    decode_graph = DecodeGraph(
                        model,
                        cache,
                        next_input,
                        capacity=num_tokens + max_new_tokens - 1,
                        eos_token_ids=eos_ids,
                    )
                    decode_start = time.perf_counter()
            else:
                num_decode_tokens += active_count
            if all_finished:
                break

    if streamer is not None:
        assert output_pool is not None and output_future is not None
        output_future.result()
        output_pool.shutdown()
        streamer.end()
    decode_time = time.perf_counter() - decode_start
    profiler.stop()
    profiler.export_chrome_trace("trace.json.gz")
    if local_rank == 0 and bs > 1:
        for request, tokens in enumerate(generated):
            print(
                f"\nRequest {request + 1}: {prompts[request]}\n"
                f"{tokenizer.decode(tokens, skip_special_tokens=True)}"
            )
    if local_rank == 0 and num_decode_tokens > 0:
        logger.info(
            f"Decode (bs={bs}): {num_decode_tokens} tokens in {decode_time:.2f}s "
            f"= {num_decode_tokens / decode_time:.2f} tok/s (excl. prefill)"
        )

    # NCCL communicator shutdown waits for every captured graph to be released.
    if decode_graph is not None:
        torch.cuda.synchronize()
        decode_graph.graph.reset()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    tyro.cli(main)
