"""One full-parameter update for complete AGL trajectories on the Hygon runtime.

VERL supplies the causal LM forward; project v1 supplies the explicitly selected
OpenRLHF-style objective. One optimizer step follows all trajectories. Padding
is removed per microbatch without removing any attended environment tokens.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.workers.actor.dp_actor import DataParallelPPOActor

from .loss import group_advantages, grpo_loss
from .trajectory import Trajectory, training_row
from .batch import as_dataproto


class UnpaddedMicroActor(DataParallelPPOActor):
    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        # Microbatch=1. Trim only padding; feedback stays in input_ids/attention.
        batch = dict(micro_batch)
        response_length = batch["responses"].shape[-1]
        prompt_length = batch["input_ids"].shape[-1] - response_length
        visible = batch["attention_mask"][0].nonzero().flatten()
        left, end = int(visible[0]), int(visible[-1]) + 1
        actual_response_length = end - prompt_length
        for key in ["input_ids", "attention_mask", "position_ids"]:
            batch[key] = batch[key][:, left:end]
        batch["responses"] = batch["responses"][:, :actual_response_length]
        entropy, log_probs = super()._forward_micro_batch(batch, temperature, calculate_entropy)
        if not torch.is_grad_enabled() and hasattr(self, "forward_progress"):
            self.forward_progress["completed"] += 1
            count = self.forward_progress["completed"]
            if count % 64 == 0 or count == self.forward_progress["total"]:
                print(json.dumps(self.forward_progress), flush=True)
        return entropy, torch.nn.functional.pad(log_probs, (0, response_length-actual_response_length))


def run(model_path, reference_path, trajectories_path, output, checkpoint, *, resume=None):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.manual_seed(42)
    fd, rendezvous = tempfile.mkstemp(prefix="reex-research-", dir=output)
    os.close(fd)
    torch.distributed.init_process_group("gloo", init_method=Path(rendezvous).resolve().as_uri(), rank=0, world_size=1)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        trajectories = [Trajectory.from_dict(x) for x in json.loads(trajectories_path.read_text())]
        rows = [training_row(t) for t in trajectories]
        batch = as_dataproto(rows, max_prompt_length=max(len(r.prompt_ids) for r in rows),
                            max_response_length=max(len(r.response_ids) for r in rows),
                            pad_token_id=tokenizer.pad_token_id, device="cpu", score_dtype=torch.float32)
        batch.meta_info.update(temperature=1.0, micro_batch_size=1, use_dynamic_bsz=False)
        config = OmegaConf.create({"use_remove_padding":False, "use_fused_kernels":False,
            "ulysses_sequence_parallel_size":1, "entropy_from_logits_with_chunking":False,
            "use_torch_compile":False, "fsdp_config":{"dtype":"bfloat16"}, "grad_clip":1.0})

        def load_model(path):
            return AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                torch_dtype=torch.float32, attn_implementation="eager").to("cuda")

        print(json.dumps({"phase":"reference_log_probs", "rows":len(rows)}), flush=True)
        reference = load_model(reference_path)
        reference_actor = UnpaddedMicroActor(config, reference, None)
        reference_actor.forward_progress = {"phase":"reference_log_probs", "completed":0, "total":len(rows)}
        ref_log_probs, _ = reference_actor.compute_log_prob(batch)
        ref_log_probs = ref_log_probs.cpu()
        del reference_actor, reference
        gc.collect(); torch.cuda.empty_cache()

        model = load_model(model_path)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, betas=(.9,.95), eps=1e-8, weight_decay=0)
        previous_step = 0
        if resume:
            state = torch.load(resume/"training_state.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            previous_step = state["step"]
            torch.set_rng_state(state["torch_rng_state"])
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
            del state
        actor = UnpaddedMicroActor(config, model, optimizer)
        actor.forward_progress = {"phase":"old_policy_log_probs", "completed":0, "total":len(rows)}
        print(json.dumps({"phase":"old_policy_log_probs"}), flush=True)
        old_log_probs, _ = actor.compute_log_prob(batch)
        old_log_probs = old_log_probs.cpu()
        # Preserve evidence for the observed intermittent KL spikes. Detached
        # CPU copies only; these values never feed back into the training loss.
        diagnostic_log_probs = torch.empty_like(old_log_probs, dtype=torch.float32, device="cpu")
        trajectory_metrics = []
        advantages = group_advantages(torch.tensor([r.reward for r in rows]), [r.group_id for r in rows])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {"policy_loss":0., "kl":0., "clip_fraction":0.}
        masked_grad_max = 0.
        policy_grad_max = 0.

        def audit(gradient, mask):
            nonlocal masked_grad_max, policy_grad_max
            if (~mask).any():
                masked_grad_max = max(masked_grad_max, float(gradient[~mask].abs().max()))
            policy_grad_max = max(policy_grad_max, float(gradient[mask].abs().max()))

        for index, micro_batch in enumerate(batch.split(1)):
            micro_batch = micro_batch.to("cuda")
            inputs = dict(micro_batch.batch)
            mask = inputs["response_mask"]
            _, logp = actor._forward_micro_batch(inputs, temperature=1.0)
            logp.register_hook(lambda g, mask=mask.bool(): audit(g, mask))
            loss, metrics = grpo_loss(logp, old_log_probs[index:index+1].to("cuda"),
                ref_log_probs[index:index+1].to("cuda"), advantages[index].to("cuda"), mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at trajectory {rows[index].rollout_id}")
            (loss / len(rows)).backward()
            for key in sums:
                sums[key] += float(metrics[key]) / len(rows)
            diagnostic_log_probs[index:index+1].copy_(logp.detach().float().cpu())
            trajectory_metrics.append({"rollout_id":rows[index].rollout_id,
                "group_id":rows[index].group_id, **{key:float(value) for key,value in metrics.items()}})
            if (index+1) % 8 == 0:
                print(json.dumps({"phase":"backward", "completed":index+1, "total":len(rows)}), flush=True)
        torch.save({"step":previous_step+1, "input_model":str(model_path),
            "reference_model":str(reference_path), "trajectory_metrics":trajectory_metrics,
            "old_log_probs":old_log_probs, "ref_log_probs":ref_log_probs,
            "current_log_probs":diagnostic_log_probs, "policy_mask":batch.batch["response_mask"].cpu()},
            output/"logprob-diagnostics.pt")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        checkpoint.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(checkpoint)
        torch.save({"step":previous_step+1, "optimizer":optimizer.state_dict(),
            "torch_rng_state":torch.get_rng_state(), "cuda_rng_state_all":torch.cuda.get_rng_state_all()},
            checkpoint/"training_state.pt")
        model_files = {}
        for path in checkpoint.glob("*.safetensors"):
            with path.open("rb") as stream:
                model_files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        report = {"step":previous_step+1, "rows":len(rows), "groups":len(set(r.group_id for r in rows)),
            "input_model":str(model_path), "fixed_reference_model":str(reference_path),
            "objective":"project-v1-openrlhf-v0.6.3-details", "metrics":sums,
            "grad_norm":float(grad_norm), "feedback_and_padding_logprob_gradient_max":masked_grad_max,
            "policy_logprob_gradient_max":policy_grad_max,
            "optimizer_steps":sorted({int(s["step"]) for s in optimizer.state.values()}),
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "nonzero_advantage_trajectories":int(advantages.count_nonzero()),
            "elapsed_seconds":time.monotonic()-started, "model_files_sha256":model_files,
            "max_memory_allocated_bytes":torch.cuda.max_memory_allocated(),
            "exploration_reward":False}
        (checkpoint/"checkpoint.json").write_text(json.dumps(report, indent=2))
        (output/"update-report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["model-path", "reference-path", "trajectories", "output", "checkpoint"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(args.model_path, args.reference_path, args.trajectories, args.output, args.checkpoint, resume=args.resume)
