report-lab3.md

corpus: 30 docs -> 91 chunks (mean 707 chars)
config                           hit_rate@1     hit_rate@5       recall@5            mrr        ndcg@10 latency_p95_ms
----------------------------------------------------------------------------------------------------------------------
baseline sliding-800 dense           0.7857         0.9286         0.8452         0.8451         0.8053       949.3100

kind             hit_rate@5     n
---------------------------------
aggregation          1.0000     4
multi_hop            1.0000    10
paraphrase           1.0000     5
single_hop           0.8889    18
trap_archived        1.0000     3
unanswerable         0.5000     2

Write these numbers down before you change anything.