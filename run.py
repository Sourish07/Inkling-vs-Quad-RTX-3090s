import time

import torch
from huggingface_hub import snapshot_download
from loguru import logger
from torch.distributed import get_world_size
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    TextStreamer,
)

from my_inkling import MyInkling
from my_inkling.cache import MyInklingCache
from my_inkling.ep_plan import apply_ep_plan
from my_inkling.tp_plan import apply_tp_plan
from utils.checkpointing import (
    load_config,
    load_expert_state_dict,
    load_non_expert_state_dict,
)
from utils.dist import get_device_mesh, setup_ddp_local, setup_rank_aware_logger

hf_repo = "thinkingmachines/Inkling-Small-NVFP4"
# First N experts in each rank's shard stay on GPU; the rest use pinned CPU RAM.
gpu_experts_per_rank = 24

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
non_expert_state_dict = load_non_expert_state_dict(model, local_hf_path, device_mesh)
missing, unexpected = model.load_state_dict(
    non_expert_state_dict, strict=False, assign=True
)
assert not unexpected and all(".mlp.experts." in name for name in missing)

expert_state_dict = load_expert_state_dict(
    model, local_hf_path, device_mesh, gpu_experts_per_rank
)
apply_ep_plan(model, device_mesh, expert_state_dict)

torch.distributed.barrier()
logger.info(f"Inkling model loaded on device {device}")

prompt_text = "What are the computational benefits of Mixture-of-Experts models?"
prompt = [{"role": "user", "content": prompt_text}]

logger.info("Running tokenizer...")
inputs: BatchEncoding = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="none",
)
inputs = inputs.to(device)
num_tokens = inputs["input_ids"].shape[1]
logger.info(f"running generation with {num_tokens} input tokens")

model.eval()
max_new_tokens = 2**5
streamer = (
    TextStreamer(tokenizer, skip_special_tokens=True) if local_rank == 0 else None
)
next_input = inputs["input_ids"]

# Step 0 is prefill (+ first token); the decode clock starts after it.
decode_start = 0.0
num_decode_tokens = 0
cache = MyInklingCache(config.text_config, device=device)

# DTensor shared-expert views require no_grad with the installed PyTorch version.
with torch.no_grad():
    for step in range(max_new_tokens):
        logits = model(next_input, cache=cache)
        next_input = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.distributed.broadcast(next_input, src=0)
        if streamer is not None:
            streamer.put(next_input.cpu())
        # .item() syncs the device, so the timestamps are accurate.
        is_eos = next_input.item() == tokenizer.eos_token_id
        if step == 0:
            decode_start = time.perf_counter()
        else:
            num_decode_tokens += 1
        if is_eos:
            break

decode_time = time.perf_counter() - decode_start
if streamer is not None:
    streamer.end()
if local_rank == 0 and num_decode_tokens > 0:
    logger.info(
        f"Decode: {num_decode_tokens} tokens in {decode_time:.2f}s "
        f"= {num_decode_tokens / decode_time:.2f} tok/s (excl. prefill)"
    )

torch.distributed.destroy_process_group()
