"""Text helpers shared by the server and the eval tools."""
import re

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    return str(n)


def normalize(text: str) -> list[str]:
    """Lowercase words without punctuation, digits spelled out (for WER / comparisons)."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"\d+", lambda m: _num(int(m.group())), text)
    text = re.sub(r"[^a-z' ]", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


def edits(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]


def tidy(text: str) -> str:
    """Model emits UPPERCASE words; make it read like a subtitle."""
    text = " ".join(text.lower().split())
    if not text:
        return text
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"\bi'(m|ll|ve|d)\b", lambda m: "I'" + m.group(1), text)
    return text[0].upper() + text[1:]
