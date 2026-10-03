# CSE-307 Term Paper — Track 1: Learned Page Replacement

## Objective

This project compares classical page-replacement policies (FIFO, LRU, and Optimal/Belady) with a lightweight learned policy based on a Decision Tree. The workload deliberately changes halfway through: Phase 1 is locality-heavy/sequential, while Phase 2 is random/bursty.

## Experimental setup

- Number of frames: 6
- Page universe: 50 pages
- Trace length: 4,000 accesses
- Phase 1: first 2,000 accesses, locality-heavy/sequential pattern
- Phase 2: next 2,000 accesses, random/bursty pattern
- Learned model: DecisionTreeClassifier (max depth 5)
- Learned features: candidate-page recency, recent-window frequency, and observed global frequency

The learned policy is trained on separate Phase-1-style traces and then evaluated on the complete shifted test trace. Future accesses are used only to create training labels for the Optimal policy; they are not used as learned-policy input features.

## Run

```bash
python -m pip install -r requirements.txt
python src/main.py
```

The script creates:

- `results/metrics.csv` — policy and phase metrics
- `results/trace.csv` — generated test trace

## AI disclosure

ChatGPT was used for implementation assistance and project-structure guidance. The workload design, experimental execution, measured results, and final analysis are reviewed and validated by me(Fojla Rabbi) before submission.

