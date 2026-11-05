"""Speaker adaptation from a short calibration session.

The user mouths a handful of known phrases. We fine-tune a small part of the
lip-reading model on them (hybrid CTC/attention loss, the model's own training
objective) and keep the result only if it also helps on held-out phrases.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from vsr_engine import CROP, MEAN, STD, VSREngine

CTC_WEIGHT = 0.1  # mtlalpha the model was trained with


def trainable_params(model, mode: str) -> list[torch.nn.Parameter]:
    if mode == "frontend":
        return list(model.encoder.frontend.parameters())
    if mode == "bitfit":  # biases + normalization affine params everywhere in the encoder
        return [p for n, p in model.encoder.named_parameters() if n.endswith("bias") or "norm" in n or ".bn" in n]
    if mode == "frontend+bitfit":
        ids = set()
        out = []
        for p in trainable_params(model, "frontend") + trainable_params(model, "bitfit"):
            if id(p) not in ids:
                ids.add(id(p))
                out.append(p)
        return out
    if mode == "encoder":
        return list(model.encoder.parameters())
    raise ValueError(mode)


def _tensor(frames: np.ndarray, device, augment: bool) -> torch.Tensor:
    t, h, w = frames.shape
    if augment:
        y0, x0 = random.randint(0, h - CROP), random.randint(0, w - CROP)
    else:
        y0, x0 = (h - CROP) // 2, (w - CROP) // 2
    x = torch.from_numpy(frames[:, y0 : y0 + CROP, x0 : x0 + CROP].copy()).float().div_(255.0)
    x = (x - MEAN) / STD
    if augment and t > 20:  # time masking, like Auto-AVSR training
        for _ in range(max(1, t // 50)):
            L = random.randint(0, min(10, t // 5))
            s = random.randint(0, t - L)
            x[s : s + L] = 0.0
    return x.unsqueeze(0).unsqueeze(0).to(device)  # (B=1, C=1, T, H, W)


def loss_for(engine: VSREngine, frames: np.ndarray, text: str, augment: bool) -> torch.Tensor:
    model = engine.model
    dev = engine.device
    ids = engine.text_to_ids(text)
    eos = model.odim - 1
    x = _tensor(frames, dev, augment)
    enc, _ = model.encoder(x, None)  # (1, T, D)

    # CTC (computed on CPU: not every PyTorch build has an MPS ctc_loss kernel)
    lp = model.ctc.ctc_lo(enc).log_softmax(-1).transpose(0, 1).float().cpu()  # (T, 1, V)
    tgt = torch.tensor([ids], dtype=torch.long)
    l_ctc = F.ctc_loss(lp, tgt, torch.tensor([lp.size(0)]), torch.tensor([len(ids)]),
                       blank=0, reduction="sum", zero_infinity=True) / max(len(ids), 1)

    # Attention decoder, teacher forced
    ys_in = torch.tensor([[eos] + ids], device=dev)
    ys_out = torch.tensor([ids + [eos]], device=dev)
    L = ys_in.size(1)
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev)).unsqueeze(0)
    logits, _ = model.decoder(ys_in, mask, enc, None)
    l_att = F.cross_entropy(logits.view(-1, logits.size(-1)), ys_out.view(-1), label_smoothing=0.1)
    return CTC_WEIGHT * l_ctc.to(dev) + (1 - CTC_WEIGHT) * l_att


@torch.no_grad()
def mean_loss(engine: VSREngine, data) -> float:
    engine.model.eval()
    return float(np.mean([loss_for(engine, f, t, augment=False).item() for f, t in data]))


def _set_train(model):
    model.train()
    for m in model.modules():  # few clips: never update BatchNorm running stats
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()


DEFAULT_MODE = "frontend"  # best on the dev video: visual frontend only (11M of 250M params)


def adapt(engine: VSREngine, train, val, mode: str = DEFAULT_MODE, steps: int = 60,
          lr: float = 1e-4, on_step=None) -> dict:
    """Fine-tune in place. train/val: lists of (uint8 frames (T,96,96), text).

    Returns a report; if held-out loss got worse the original weights are restored.
    """
    model = engine.model
    params = trainable_params(model, mode)
    backup = {id(p): p.detach().clone() for p in params}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)

    t0 = time.perf_counter()
    before = mean_loss(engine, val) if val else None
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    _set_train(model)
    order = []
    for step in range(steps):
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)
        f, t = train[order.pop()]
        loss = loss_for(engine, f, t, augment=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if on_step:
            on_step(step + 1, steps, loss.item())
    model.eval()
    for p in params:
        p.requires_grad_(False)

    after = mean_loss(engine, val) if val else None
    kept = before is None or after < before
    if not kept:
        with torch.no_grad():
            for p in params:
                p.copy_(backup[id(p)])
    return {"mode": mode, "steps": steps, "val_loss_before": before, "val_loss_after": after,
            "kept": kept, "seconds": round(time.perf_counter() - t0, 1)}


def adapted_state(engine: VSREngine, mode: str) -> dict:
    names = {id(p): n for n, p in engine.model.named_parameters()}
    return {names[id(p)]: p.detach().cpu().clone() for p in trainable_params(engine.model, mode)}


def load_adapted_state(engine: VSREngine, state: dict) -> None:
    params = dict(engine.model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v.to(params[n].device))
