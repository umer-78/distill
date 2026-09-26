"""python -m distill bench     the distillation run, written to results/"""
import json
import sys
from pathlib import Path

from .pipeline import run

RESULTS = Path(__file__).resolve().parent.parent / "results"


def report(r):
    pct = lambda x: f"{100 * x:.1f}%"
    t, s = r["teacher"], r["students"]
    names = [("tfidf", s["tfidf"]), ("bge", s["bge"])]
    lines = [f"{r['examples']:,} teacher-labelled texts ({r['dropped_off_schema']} off-schema answers dropped); "
             f"{r['train']:,} train, {r['val']} validation and {r['test']} test reviews, plus {r['test_variants']} test variants.", "",
             "| | Teacher (GPT-3.5 Turbo) | " + " | ".join(m["name"] for _, m in names) + " |", "|---|---|---|---|",
             f"| Accuracy on test reviews | {pct(t['accuracy'])} | " + " | ".join(pct(m["accuracy"]) for _, m in names) + " |",
             "| Agreement with the teacher | — | " + " | ".join(pct(m["agreement"]) for _, m in names) + " |",
             f"| Accuracy on test variants (typos, dialect, names) | {pct(t['accuracy_variants'])} | "
             + " | ".join(pct(m["accuracy_variants"]) for _, m in names) + " |",
             "| Same student trained on gold labels | — | " + " | ".join(pct(m["accuracy_trained_on_gold"]) for _, m in names) + " |",
             f"| Latency, one request (p50 / p95) | API round trip | "
             + " | ".join(f"{m['speed']['p50_ms']:.1f} / {m['speed']['p95_ms']:.1f} ms" for _, m in names) + " |",
             f"| Throughput on {r['threads']} CPU threads | — | " + " | ".join(f"{m['speed']['per_second']:,.0f} per second" for _, m in names) + " |",
             f"| Cost per 1,000 requests | ${1000 * t['cost_per_request']:.2f} | see below | see below |", "",
             f"The teacher's prompt averaged {t['prompt_tokens']:,.0f} tokens (HELM's five worked examples plus the review).", "",
             "## Break-even (monthly cost, teacher against the bge student on one or more 2-vCPU machines)", "",
             "| Requests per month | Teacher | Student |", "|---|---|---|"]
    econ = s["bge"]["economics"]
    lines += [f"| {m['requests']:,} | ${m['teacher']:,.0f} | ${m['student']:,.0f} |" for m in econ["monthly"]]
    lines += ["", f"Owning the student is cheaper above **{econ['break_even_per_month']:,} requests a month**, including "
              f"${econ['labelling_cost']:,.2f} of teacher labelling for the training set, spread over a year.", "",
              "## Learning curve (bge student, trained on the teacher's labels)", "",
              "| Training reviews | Examples with variants | Accuracy | Agreement with teacher |", "|---|---|---|---|"]
    lines += [f"| {c['reviews']} | {c['examples']} | {pct(c['accuracy'])} | {pct(c['agreement'])} |" for c in r["learning_curve"]]
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1:] != ["bench"]:
        sys.exit(__doc__)
    result = run()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(result, indent=1))
    text = report(result)
    (RESULTS / "bench.md").write_text(text + "\n")
    print(text)
