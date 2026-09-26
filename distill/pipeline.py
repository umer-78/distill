"""The distillation run.

1. Split by review, before anything is tuned: 60% train, 20% validation, 20% test. A review's
   variants stay with it, so no near-copy of a test review is ever trained on.
2. Train two students on the teacher's labels: TF-IDF with logistic regression, and a
   logistic-regression head on bge-small sentence embeddings. Regularisation is picked on
   validation agreement with the teacher, because in production there are no gold labels.
3. On the untouched test reviews: accuracy against gold, agreement with the teacher,
   robustness on the variants, latency and throughput on this CPU, and cost. For reference,
   the same students trained on gold labels show what distillation gives up.
"""
import hashlib
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import data
from .embed import Embedder

PRICES = Path(__file__).with_name("prices.yaml")
CS = (0.1, 1.0, 10.0)


def split_of(review):
    h = int(hashlib.sha256(f"distill:{review}".encode()).hexdigest(), 16) % 10
    return "train" if h < 6 else "val" if h < 8 else "test"


class Tfidf:
    name = "TF-IDF + logistic regression"

    def fit(self, texts, y, c):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        self.clf = LogisticRegression(C=c, max_iter=2000).fit(self.vec.fit_transform(texts), y)
        return self

    def predict(self, texts):
        return self.clf.predict(self.vec.transform(texts))


class Bge:
    name = "bge-small + logistic regression"

    def __init__(self, embedder, vectors):
        self.embedder, self.vectors = embedder, vectors       # vectors: {text: embedding} precomputed for the bench

    def fit(self, texts, y, c):
        self.clf = LogisticRegression(C=c, max_iter=2000).fit(np.vstack([self.vectors[t] for t in texts]), y)
        return self

    def predict(self, texts, live=False):
        x = self.embedder.encode(texts) if live else np.vstack([self.vectors[t] for t in texts])
        return self.clf.predict(x)


def tuned(make, train, val, labels="teacher"):
    best = max(CS, key=lambda c: np.mean(make().fit([e["text"] for e in train], [e[labels] for e in train], c)
                                        .predict([e["text"] for e in val]) == [e["teacher"] for e in val]))
    return make().fit([e["text"] for e in train], [e[labels] for e in train], best), best


def speed(predict, texts, single=50):
    t = []
    for text in texts[:single]:
        start = time.perf_counter()
        predict([text])
        t.append(time.perf_counter() - start)
    start = time.perf_counter()
    predict(texts)
    return {"p50_ms": 1000 * float(np.median(t)), "p95_ms": 1000 * float(np.percentile(t, 95)),
            "per_second": len(texts) / (time.perf_counter() - start)}


def economics(per_second, teacher_cost, labelled, prices):
    """Monthly cost of the teacher and the student at each volume, and the break-even volume."""
    host = prices["student_host"]
    one_off = labelled * teacher_cost / prices["amortise_months"]

    def student(volume):
        peak = volume / (30 * 86400) * prices["peak_to_average"]
        machines = max(1, math.ceil(peak / per_second))
        return machines * host["per_hour"] * 730 + one_off
    volumes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    grid = np.unique(np.logspace(3, 9, 6001).astype(int))
    even = next((int(v) for v in grid if student(v) <= v * teacher_cost), None)
    return {"teacher_per_request": teacher_cost, "labelling_cost": labelled * teacher_cost, "break_even_per_month": even,
            "monthly": [{"requests": v, "teacher": v * teacher_cost, "student": student(v)} for v in volumes]}


def run(seed=0):
    prices = yaml.safe_load(PRICES.read_text())
    rows, dropped = data.examples()
    for e in rows:
        e["split"] = split_of(e["id"])
    part = lambda s, variants=False: [e for e in rows if e["split"] == s and (variants or e["variant"] == "original")]
    train, val, test = part("train", True), part("val"), part("test")
    test_variants = [e for e in rows if e["split"] == "test" and e["variant"] != "original"]
    embedder = Embedder(threads=2)       # what latency and throughput are measured on: a 2-vCPU machine
    bulk = Embedder(threads=os.cpu_count() or 2)
    vectors = dict(zip([e["text"] for e in rows], bulk.cached([e["text"] for e in rows])))
    gold = lambda es: np.array([e["gold"] for e in es])
    teacher = lambda es: np.array([e["teacher"] for e in es])
    out = {"examples": len(rows), "dropped_off_schema": dropped, "train": len(train), "val": len(val), "test": len(test),
           "test_variants": len(test_variants), "cpu": platform.processor() or platform.machine(), "threads": 2,
           "teacher": {"accuracy": float(np.mean(gold(test) == teacher(test))),
                       "accuracy_variants": float(np.mean(gold(test_variants) == teacher(test_variants)))},
           "students": {}}
    tokens_in = np.mean([e["prompt_tokens"] for e in rows])
    tokens_out = np.mean([e["output_tokens"] for e in rows])
    teacher_cost = (tokens_in * prices["teacher"]["input_per_million"] + tokens_out * prices["teacher"]["output_per_million"]) / 1e6
    out["teacher"].update(prompt_tokens=float(tokens_in), cost_per_request=teacher_cost)
    for key, make in (("tfidf", Tfidf), ("bge", lambda: Bge(embedder, vectors))):
        model, c = tuned(make, train, val)
        on_gold, _ = tuned(make, train, val, labels="gold")
        texts = [e["text"] for e in test]
        live = (lambda t: model.predict(t, live=True)) if key == "bge" else model.predict
        s = speed(live, texts)
        p, pv = model.predict(texts), model.predict([e["text"] for e in test_variants])
        out["students"][key] = {"name": model.name, "C": c, "accuracy": float(np.mean(p == gold(test))),
                                "agreement": float(np.mean(p == teacher(test))),
                                "accuracy_variants": float(np.mean(pv == gold(test_variants))),
                                "accuracy_trained_on_gold": float(np.mean(on_gold.predict(texts) == gold(test))),
                                "speed": s, "economics": economics(s["per_second"], teacher_cost, len(train), prices)}
    reviews = sorted({e["id"] for e in train})
    rng = np.random.default_rng(seed)
    out["learning_curve"] = []
    for n in (25, 50, 100, 200, 400, len(reviews)):
        keep = set(rng.choice(reviews, size=min(n, len(reviews)), replace=False))
        subset = [e for e in train if e["id"] in keep]
        model, _ = tuned(lambda: Bge(embedder, vectors), subset, val)
        p = model.predict([e["text"] for e in test])
        out["learning_curve"].append({"reviews": len(keep), "examples": len(subset), "accuracy": float(np.mean(p == gold(test))),
                                      "agreement": float(np.mean(p == teacher(test)))})
    return out
