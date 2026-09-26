import numpy as np

from distill.data import normalise
from distill.pipeline import Tfidf, economics, split_of, tuned

PRICES = {"student_host": {"per_hour": 0.1, "vcpus": 2}, "peak_to_average": 3, "amortise_months": 12}


def test_splits_keep_a_review_and_its_variants_together():
    assert split_of("id1") == split_of("id1")
    share = np.mean([split_of(f"id{i}") == "train" for i in range(5000)])
    assert 0.57 < share < 0.63


def test_normalise_catches_near_copies():
    assert normalise("Great  film!!") == normalise("great film")


def test_student_is_tuned_on_teacher_labels():
    rows = [{"text": f"{w} movie {i}", "teacher": int(w == "good"), "gold": int(w == "good")}
            for i in range(60) for w in ("good", "bad")]
    model, c = tuned(Tfidf, rows[:80], rows[80:])
    assert c in (0.1, 1.0, 10.0)
    assert np.mean(model.predict([r["text"] for r in rows[80:]]) == [r["teacher"] for r in rows[80:]]) == 1.0


def test_break_even_is_where_the_fixed_machine_pays_for_itself():
    e = economics(per_second=100, teacher_cost=0.001, labelled=1000, prices=PRICES)
    fixed = 0.1 * 730 + 1000 * 0.001 / 12
    assert abs(e["break_even_per_month"] - fixed / 0.001) / (fixed / 0.001) < 0.01
    big = e["monthly"][-1]
    assert big["student"] > fixed and big["teacher"] == big["requests"] * 0.001
