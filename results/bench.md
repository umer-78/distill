3,627 teacher-labelled texts (70 off-schema answers dropped); 2,081 train, 202 validation and 215 test reviews, plus 580 test variants.

| | Teacher (GPT-3.5 Turbo) | TF-IDF + logistic regression | bge-small + logistic regression |
|---|---|---|---|
| Accuracy on test reviews | 95.3% | 85.1% | 94.9% |
| Agreement with the teacher | — | 87.0% | 93.0% |
| Accuracy on test variants (typos, dialect, names) | 95.2% | 84.3% | 92.6% |
| Same student trained on gold labels | — | 84.7% | 94.4% |
| Latency, one request (p50 / p95) | API round trip | 1.2 / 5.7 ms | 71.0 / 91.7 ms |
| Throughput on 2 CPU threads | — | 3,678 per second | 12 per second |
| Cost per 1,000 requests | $3.91 | see below | see below |

The teacher's prompt averaged 2,606 tokens (HELM's five worked examples plus the review).

## Break-even (monthly cost, teacher against the bge student on one or more 2-vCPU machines)

| Requests per month | Teacher | Student |
|---|---|---|
| 10,000 | $39 | $66 |
| 100,000 | $391 | $66 |
| 1,000,000 | $3,911 | $66 |
| 10,000,000 | $39,108 | $66 |
| 100,000,000 | $391,077 | $652 |

Owning the student is cheaper above **16,865 requests a month**, including $8.14 of teacher labelling for the training set, spread over a year.

## Learning curve (bge student, trained on the teacher's labels)

| Training reviews | Examples with variants | Accuracy | Agreement with teacher |
|---|---|---|---|
| 25 | 91 | 88.8% | 87.9% |
| 50 | 185 | 92.6% | 91.6% |
| 100 | 355 | 93.0% | 91.2% |
| 200 | 729 | 94.0% | 93.0% |
| 400 | 1457 | 94.4% | 92.6% |
| 571 | 2081 | 94.9% | 93.0% |
