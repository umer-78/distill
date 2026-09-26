"""The teacher's recorded work: GPT-3.5 Turbo (0613) labelling movie reviews as positive or
negative, from HELM Classic's IMDB run (v0.3.0), with the gold label for each.

1,000 real reviews, plus the variants HELM made of them to test robustness (typos and
casing, dialect, swapped gender terms and names), each labelled by the teacher in its own
right. Near-identical texts are dropped, and so are teacher answers outside the label set
(it sometimes says "Neutral" or "Mixed"). Downloaded on first use into DISTILL_DATA
(default ~/.cache/distill), with the student's sentence embedder.
"""
import hashlib
import json
import os
import re
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://storage.googleapis.com/crfm-helm-public/classic/benchmark_output/runs/v0.3.0"
TEACHER = "openai_gpt-3.5-turbo-0613"
RUN = f"imdb:model={TEACHER},data_augmentation=canonical"
LABELS = ("Negative", "Positive")
EMBEDDER = ("https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz",
            "3858004b3822f64f940280874b8f2d2dc25b34a4f3eb3cdf617bdceeb21ed9ed", "fast-bge-small-en-v1.5")


def cache_dir():
    path = Path(os.environ.get("DISTILL_DATA", Path.home() / ".cache" / "distill"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch(url, path, tries=4):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=300) as r:
                    body = r.read()
                break
            except OSError:
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)
        path.write_bytes(body)
    return path


def normalise(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def examples():
    """([{"id", "variant", "text", "gold", "teacher", "prompt_tokens", "output_tokens"}], teacher answers dropped).
    `id` is the review; a variant shares its review's id, so splits can keep them together."""
    run = urllib.parse.quote(RUN, safe="")
    preds = json.loads(fetch(f"{BUCKET}/{run}/display_predictions.json", cache_dir() / run / "p.json").read_text())
    insts = json.loads(fetch(f"{BUCKET}/{run}/instances.json", cache_dir() / run / "i.json").read_text())
    variant = lambda x: (x.get("perturbation") or {}).get("name", "original")
    by_key = {(i["id"], variant(i)): i for i in insts}
    out, seen, invalid = [], set(), 0
    for p in sorted(preds, key=lambda p: variant(p) != "original"):       # originals first, so a duplicate variant is what goes
        inst = by_key.get((p["instance_id"], variant(p)))
        if inst is None or p.get("train_trial_index", 0) != 0:
            continue
        key = normalise(inst["input"]["text"])
        if key in seen:
            continue
        seen.add(key)
        teacher = p["predicted_text"].strip()
        if teacher not in LABELS:          # schema check: drop, don't teach, a malformed answer
            invalid += 1
            continue
        gold = next(r["output"]["text"] for r in inst["references"] if "correct" in r["tags"])
        out.append({"id": inst["id"], "variant": variant(inst), "text": inst["input"]["text"], "gold": LABELS.index(gold),
                    "teacher": LABELS.index(teacher), "prompt_tokens": p["stats"]["num_prompt_tokens"],
                    "output_tokens": p["stats"]["num_output_tokens"]})
    return out, invalid


def embedder_dir():
    url, sha, folder = EMBEDDER
    target = cache_dir() / folder
    if not (target / "tokenizer.json").exists():
        archive = fetch(url, cache_dir() / f"{folder}.tar.gz")
        if hashlib.sha256(archive.read_bytes()).hexdigest() != sha:
            archive.unlink()
            raise RuntimeError("the embedder download does not match its pinned sha256")
        with tarfile.open(archive) as tar:
            tar.extractall(cache_dir(), members=[m for m in tar.getmembers() if not Path(m.name).name.startswith("._")],
                           filter="data")
    return target
