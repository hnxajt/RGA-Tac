"""Single-GPU TVL-LLaMA fine-tuning."""

from functools import partial
import random
from typing import Any

import numpy as np

from rga_tac.integrations.tvl import add_tvl_generation_paths


def run_finetune(args) -> None:
    add_tvl_generation_paths(args.tvl_root)
    import torch
    from data.dataset import FinetuneDataset
    from llama.llama_adapter import LLaMA_adapter
    import util.lr_sched as lr_sched
    import util.misc as misc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    tokenizer = args.llama_path / "tokenizer.model"
    model = LLaMA_adapter(str(args.llama_path / args.llama_type), str(tokenizer), args=args, load=True)
    model.to(device)
    _enable_activation_checkpointing(model)
    checkpoint = torch.load(args.pretrained_path, map_location="cpu", weights_only=False)
    model.load_state_dict({key.replace("llma", "llama"): value for key, value in checkpoint["model"].items()}, strict=False)
    del checkpoint
    args.lr = args.blr * args.batch_size * args.accum_iter / 256
    optimizer = torch.optim.AdamW(misc.add_weight_decay(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95))
    scaler = misc.NativeScalerWithGradNormCount()
    dataset_train = FinetuneDataset(str(args.data_config), max_words=args.max_words, tokenizer_path=str(tokenizer), crop_tacvis=True)
    dataset_dev = FinetuneDataset(str(args.val_data_config), max_words=args.max_words, tokenizer_path=str(tokenizer), crop_tacvis=True)
    generator = torch.Generator().manual_seed(args.seed)
    worker_init = partial(_seed_worker, seed=args.seed)
    sampler = torch.utils.data.DistributedSampler(dataset_train, num_replicas=1, rank=0, shuffle=True, seed=args.seed)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init, generator=generator,
    )
    # The upstream model's inference KV cache has max_batch_size=1.
    dev_loader = torch.utils.data.DataLoader(
        dataset_dev, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=worker_init, generator=generator,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        logger = misc.MetricLogger(delimiter="  ")
        optimizer.zero_grad()
        for step, batch in enumerate(logger.log_every(train_loader, 10, f"Epoch: [{epoch}]")):
            if step % args.accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, step / len(train_loader) + epoch, args)
            c_loss, _ = _model_losses(model, batch, device)
            update = (step + 1) % args.accum_iter == 0 or step + 1 == len(train_loader)
            scaler(c_loss / args.accum_iter, optimizer, parameters=model.parameters(), update_grad=update)
            if update:
                optimizer.zero_grad()
            logger.update(closs=c_loss.item(), lr=optimizer.param_groups[0]["lr"])
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                c_loss, _ = _model_losses(model, batch, device)
                dev_loss += c_loss.item()
        dev_loss /= len(dev_loader)
        print(f"Epoch {epoch + 1}: train_loss={logger.meters['closs'].global_avg:.6f}, dev_loss={dev_loss:.6f}")
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.output_dir / "checkpoint-best.pth")


def _model_losses(model, batch: dict, device):
    import torch

    examples = batch["input2"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    active = ((examples != 0).any(dim=0) | (labels != 0).any(dim=0)).nonzero()
    length = int(active[-1]) + 1 if len(active) else 1
    observations = {name: batch[field].to(device, non_blocking=True) for name, field in (("vision", "image"), ("tactile", "tactile"))}
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        return model(examples[:, :length], labels[:, :length], observations)


def _seed_worker(worker_id: int, *, seed: int) -> None:
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _enable_activation_checkpointing(model) -> None:
    import torch
    from torch.utils.checkpoint import checkpoint

    for layer in model.llama.layers:
        forward = layer.forward

        def checkpointed_forward(*args: Any, _forward=forward, _layer=layer, **kwargs: Any):
            if _layer.training and torch.is_grad_enabled():
                return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
            return _forward(*args, **kwargs)

        layer.forward = checkpointed_forward
