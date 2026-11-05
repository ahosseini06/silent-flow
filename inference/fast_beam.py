"""Lean joint CTC/attention + LM beam search for a single utterance.

Same scoring as ESPnet's BatchBeamSearch (decoder, CTC prefix score, LM,
pre-beam on decoder scores), but built for low latency on Apple Silicon:

* decoder and LM run batched over the beam with true per-layer KV caches,
  and encoder cross-attention K/V computed once per utterance (fast_step.py),
* the CTC prefix scorer runs in NumPy on the CPU, which beats ~150 tiny GPU
  kernel launches per step,
* exactly one device->host sync per output token,
* exact early stop: every score term is a log-probability (<= 0), so once a
  finished hypothesis beats every running one, nothing can overtake it.
"""

from __future__ import annotations

import numpy as np
import torch

from fast_step import FastDecoder, FastLM, select

NEG = -1e10


class CTCPrefixScorer:
    """Vectorized CTC prefix scoring (Watanabe et al., 2017) over candidates."""

    def __init__(self, logp: np.ndarray, blank: int, eos: int):
        self.x = logp  # (T, O) float32 log-probs
        self.T = logp.shape[0]
        self.blank = blank
        self.eos = eos

    def initial_state(self):
        r = np.full((self.T, 2, 1), NEG, dtype=np.float32)
        r[:, 1, 0] = np.cumsum(self.x[:, self.blank])
        return r, np.zeros(1, dtype=np.float32)

    def score(self, r_prev, psi_prev, last, out_len, hyp_idx, cand):
        """Score candidate extensions.

        r_prev: (T, 2, W) forward probs of the running prefixes
        psi_prev: (W,) prefix scores
        last: (W,) last token of each prefix
        out_len: current prefix length (excluding sos)
        hyp_idx, cand: (N,) which hypothesis each candidate extends, and the token
        Returns ctc score deltas (N,) and new forward probs (T, 2, N).
        """
        T, x = self.T, self.x
        n = cand.shape[0]
        xc = x[:, cand]  # (T, N)
        xb = x[:, self.blank][:, None]  # (T, 1)

        r_sum = np.logaddexp(r_prev[:, 0], r_prev[:, 1])  # (T, W)
        log_phi = r_sum[:, hyp_idx]  # (T, N)
        same = cand == last[hyp_idx]
        if same.any():
            log_phi[:, same] = r_prev[:, 1, hyp_idx[same]]

        r = np.full((T, 2, n), NEG, dtype=np.float32)
        if out_len == 0:
            r[0, 0] = xc[0]
        start = max(out_len, 1)
        rn, rb = r[start - 1, 0].copy(), r[start - 1, 1].copy()
        for t in range(start, T):
            rn_new = np.logaddexp(rn, log_phi[t - 1]) + xc[t]
            rb = np.logaddexp(rn, rb) + xb[t]
            rn = rn_new
            r[t, 0], r[t, 1] = rn, rb

        if start < T:
            phi_x = log_phi[start - 1 : T - 1] + xc[start:T]
            log_psi = np.logaddexp(np.logaddexp.reduce(phi_x, axis=0), r[start - 1, 0])
        else:
            log_psi = r[start - 1, 0].copy()

        is_eos = cand == self.eos
        if is_eos.any():
            log_psi[is_eos] = r_sum[T - 1, hyp_idx[is_eos]]
        log_psi[cand == self.blank] = NEG
        return log_psi - psi_prev[hyp_idx], r, log_psi


def _repeat_bans(seqs: list[list[int]], n: int) -> tuple[list[int], list[int]]:
    rows, cols = [], []
    for w, seq in enumerate(seqs):
        toks = seq[1:]  # drop sos
        if len(toks) < n - 1:
            continue
        tail = toks[len(toks) - n + 1:]
        for i in range(len(toks) - n + 1):
            if toks[i : i + n - 1] == tail:
                rows.append(w)
                cols.append(toks[i + n - 1])
    return rows, cols


@torch.inference_mode()
def beam_search(model, lm, enc: torch.Tensor, beam: int = 8, ctc_weight: float = 0.1,
                lm_weight: float = 0.3, max_len: int | None = None, nbest: int = 1,
                no_repeat: int = 4):
    """Decode one utterance. enc: (T, D) encoder output.

    Returns up to `nbest` (score, token ids without sos/eos), best first. The
    search only stops once the n-th best finished hypothesis beats every running
    one, so the returned list is the exact top-n under the beam."""
    dev = enc.device
    eos = sos = model.odim - 1
    blank = 0
    T = enc.size(0)
    max_len = max_len or T
    pre_beam = min(int(1.5 * beam), model.odim)
    use_lm = lm is not None and lm_weight > 0

    ctc_logp = model.ctc.log_softmax(enc.unsqueeze(0))[0].float().cpu().numpy()
    ctc = CTCPrefixScorer(ctc_logp, blank, eos)
    r_prev, psi_prev = ctc.initial_state()

    dec = FastDecoder(model.decoder)
    dec.start(enc)
    flm = FastLM(lm) if use_lm else None
    last_tok = torch.full((1,), sos, dtype=torch.long, device=dev)
    ys_np = [[sos]]
    scores = np.zeros(1, dtype=np.float32)
    dec_cache = None
    lm_cache = None
    finished: list[tuple[float, list[int]]] = []

    for step in range(max_len):
        W = len(ys_np)
        L = step + 1  # prefix length including sos
        dec_logp, dec_cache = dec.step(last_tok, step, dec_cache)
        weighted = (1.0 - ctc_weight) * dec_logp
        if use_lm:
            lm_logp, lm_cache = flm.step(last_tok, lm_cache)
            weighted = weighted + lm_weight * lm_logp
        dec_logp[:, blank] = NEG
        weighted[:, blank] = NEG
        if no_repeat:  # the attention decoder can loop ("I'VE GOT IT I'VE GOT IT ..."): ban repeated n-grams
            rows, cols = _repeat_bans(ys_np, no_repeat)
            if rows:
                r = torch.tensor(rows, device=dev)
                c = torch.tensor(cols, device=dev)
                dec_logp[r, c] = NEG
                weighted[r, c] = NEG
        if step == max_len - 1:  # force termination
            dec_logp[:, :eos] = NEG
            weighted[:, :eos] = NEG
        _, cand_t = dec_logp.topk(pre_beam, dim=-1)  # pre-beam on decoder score
        part_t = weighted.gather(1, cand_t)
        packed = torch.cat([cand_t.float(), part_t], dim=1).cpu().numpy()  # single sync
        cand = packed[:, :pre_beam].astype(np.int64).reshape(-1)
        part = packed[:, pre_beam:].reshape(-1)
        hyp_idx = np.repeat(np.arange(W), pre_beam)

        last = np.array([s[-1] for s in ys_np], dtype=np.int64)
        ctc_delta, r_new, psi_new = ctc.score(r_prev, psi_prev, last, L - 1, hyp_idx, cand)
        total = scores[hyp_idx] + part + ctc_weight * ctc_delta

        order = np.argsort(-total)[:beam]
        keep = []
        for n in order:
            seq = ys_np[hyp_idx[n]] + [int(cand[n])]
            if cand[n] == eos:
                finished.append((float(total[n]), seq[1:-1]))
            else:
                keep.append(n)

        best_running = total[keep[0]] if keep else NEG
        if len(finished) >= nbest:
            nth = sorted((f[0] for f in finished), reverse=True)[nbest - 1]
            if nth >= best_running:
                break
        if not keep:
            break
        keep = np.array(keep)
        src = torch.from_numpy(hyp_idx[keep]).to(dev)
        last_tok = torch.from_numpy(cand[keep]).to(dev)
        ys_np = [ys_np[hyp_idx[n]] + [int(cand[n])] for n in keep]
        scores = total[keep]
        dec_cache = select(dec_cache, src)
        lm_cache = select(lm_cache, src)
        r_prev = r_new[:, :, keep]
        psi_prev = psi_new[keep]

    finished.sort(key=lambda f: -f[0])
    out, seen = [], set()
    for score, ids in finished:
        if tuple(ids) not in seen:
            seen.add(tuple(ids))
            out.append((score, ids))
    return out[:nbest]
