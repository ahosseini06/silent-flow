"""Context-aware correction of lip-reading output with a small local LLM (MLX).

Lip reading can't see sound, so it confuses words that look alike on the lips
("thread"/"threat", "leg"/"land", p/b/m, f/v, t/d/n). The LLM sees the n-best
candidates for the latest sentence plus the previous subtitles and writes the
most plausible sentence. A guard rejects rewrites that stray too far from the
top candidate, so it can fix look-alike words but not invent new sentences.

Measured on the dev video: n-best *rescoring* barely helped (the right word is
rarely in the n-best list), generative correction with Qwen3.5-4B took WER from
14.0% to 13.3%; the 9B model did worse (15.1%) and is slower.
"""

from __future__ import annotations

import time

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from textutil import edits, normalize

DEFAULT_REPO = "mlx-community/Qwen3.5-4B-MLX-4bit"

SYSTEM = (
    "You correct the output of a lip-reading model. Lip reading cannot see sound, so it confuses words that "
    "look alike on the lips: p/b/m, f/v, th, t/d/n/s/z/l, k/g/h, sh/ch/j, w/r, and similar vowels. Given the "
    "earlier part of the conversation and the model's candidate readings of the latest sentence (most likely "
    "first), write what the person most likely actually said. Keep the words and length of the top candidate "
    "unless a look-alike substitution makes the sentence clearly more sensible. Reply with the corrected "
    "sentence only."
)


class LLMCorrector:
    def __init__(self, repo: str = DEFAULT_REPO, max_change: float = 0.4):
        self.model, self.tok = load(repo)
        self.max_change = max_change
        self.sampler = make_sampler(temp=0.0)
        self.correct(["hello there", "hello bear"], [])  # warm-up (first call compiles kernels)

    def correct(self, candidates: list[str], context: list[str]) -> tuple[str | None, dict]:
        """candidates: lowercase n-best, best first. Returns corrected text or None if unchanged/rejected."""
        t = time.perf_counter()
        user = ("Conversation so far: " + (" ".join(context[-3:]) or "(none)") + "\n\nCandidates:\n"
                + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates)))
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                              enable_thinking=False)
        out = generate(self.model, self.tok, prompt, max_tokens=80, sampler=self.sampler)
        out = out.strip().split("\n")[0].strip().strip('"').strip()
        timing = {"llm_ms": round((time.perf_counter() - t) * 1e3, 1)}

        top, new = normalize(candidates[0]), normalize(out)
        if not new or new == top:
            return None, timing
        if edits(top, new) > max(2, self.max_change * len(top)):
            timing["rejected"] = out
            return None, timing
        return " ".join(new), timing
