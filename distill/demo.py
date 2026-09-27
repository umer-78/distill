"""python -m distill.demo   write the live demo's data (docs/data.json): results/summary.json and the prices
the break-even uses (distill/prices.yaml), so the page can redo the economics with other numbers."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def build(out=ROOT / "docs"):
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    prices = yaml.safe_load((ROOT / "distill" / "prices.yaml").read_text())
    out.mkdir(exist_ok=True)
    (out / "data.json").write_text(json.dumps({"summary": summary, "prices": prices}, indent=1))
    print(f"wrote {out / 'data.json'}")


if __name__ == "__main__":
    build()
