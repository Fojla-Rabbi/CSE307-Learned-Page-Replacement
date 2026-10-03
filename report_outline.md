# CSE-307 Term Paper
## Learning-Augmented OS Heuristics: Classical Algorithms Meet Adaptive Prediction
### Track 1 — Learned Page Replacement

## 1. Problem Framing
Need to explain why page replacement matters and introduce FIFO, LRU, and Optimal. State the research question: how do classical policies and a lightweight learned policy respond when page-access behavior changes halfway through a workload?

## 2. Implementation
Need to describe FIFO, LRU, Optimal (Belady), and the Decision Tree. State the learned features: recency, recent-window frequency, and observed global frequency. Explain that Optimal labels are used during model training but future accesses are not supplied as learned-policy features.

## 3. Experimental Setup
Need to state the number of frames, page universe, trace length, Phase 1 pattern, Phase 2 pattern, model configuration, and software environment. Explain that the same test trace is used for every policy.

## 4. Results
Need to insert the measured `results/metrics.csv` values and at least one chart. Report total faults and hit ratio, plus Phase 1 versus Phase 2 values.

## 5. Analysis
Need to discuss what changed after the workload shift and why. Compare the learned policy with the classical policies without claiming a result before observing the actual measurements.

## 6. References
Need to use proper in-text citations and list the sources here.
