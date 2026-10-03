import csv
import os
import random
from collections import deque, Counter

from sklearn.tree import DecisionTreeClassifier


PAGE_COUNT = 50
FRAMES = 6
PHASE_LENGTH = 2000
RECENT_WINDOW = 20
SEED = 307


def fifo(trace, capacity):
    frames, order, faults = set(), deque(), 0
    hits_by_phase = [0, 0]
    faults_by_phase = [0, 0]
    phase_start = PHASE_LENGTH
    for i, page in enumerate(trace):
        phase = 0 if i < phase_start else 1
        if page in frames:
            hits_by_phase[phase] += 1
        else:
            faults += 1
            faults_by_phase[phase] += 1
            if len(frames) == capacity:
                victim = order.popleft()
                frames.remove(victim)
            frames.add(page)
            order.append(page)
    return metrics(trace, faults, hits_by_phase, faults_by_phase)


def lru(trace, capacity):
    frames, last_used, faults = set(), {}, 0
    hits_by_phase = [0, 0]
    faults_by_phase = [0, 0]
    for i, page in enumerate(trace):
        phase = 0 if i < PHASE_LENGTH else 1
        if page in frames:
            hits_by_phase[phase] += 1
        else:
            faults += 1
            faults_by_phase[phase] += 1
            if len(frames) == capacity:
                victim = min(frames, key=lambda p: last_used.get(p, -1))
                frames.remove(victim)
            frames.add(page)
        last_used[page] = i
    return metrics(trace, faults, hits_by_phase, faults_by_phase)


def optimal(trace, capacity):
    frames, faults = set(), 0
    hits_by_phase = [0, 0]
    faults_by_phase = [0, 0]
    for i, page in enumerate(trace):
        phase = 0 if i < PHASE_LENGTH else 1
        if page in frames:
            hits_by_phase[phase] += 1
            continue
        faults += 1
        faults_by_phase[phase] += 1
        if len(frames) == capacity:
            future = trace[i + 1:]
            victim = max(frames, key=lambda p: future.index(p) if p in future else float('inf'))
            frames.remove(victim)
        frames.add(page)
    return metrics(trace, faults, hits_by_phase, faults_by_phase)


def generate_trace(seed=SEED, length=PHASE_LENGTH * 2):
    rng = random.Random(seed)
    trace = []

    # Phase 1: locality-heavy sequential pattern. Most requests follow a
    # small hot working set, with occasional cold-page accesses.
    hot = list(range(6))
    for i in range(PHASE_LENGTH):
        if rng.random() < 0.88:
            trace.append(hot[i % len(hot)])
        else:
            trace.append(rng.randrange(6, 16))

    # Phase 2: random/bursty pattern over a much larger page universe.
    for i in range(PHASE_LENGTH, length):
        if i % 50 < 15:
            burst_base = rng.randrange(20, PAGE_COUNT - 5)
            trace.append(rng.randrange(burst_base, burst_base + 5))
        else:
            trace.append(rng.randrange(PAGE_COUNT))
    return trace[:length]


def metrics(trace, faults, hits_by_phase, faults_by_phase):
    total = len(trace)
    return {
        'total_accesses': total,
        'faults': faults,
        'hits': total - faults,
        'hit_ratio': (total - faults) / total,
        'phase1_faults': faults_by_phase[0],
        'phase1_hit_ratio': hits_by_phase[0] / PHASE_LENGTH,
        'phase2_faults': faults_by_phase[1],
        'phase2_hit_ratio': hits_by_phase[1] / PHASE_LENGTH,
    }


def candidate_features(trace, index, frames, page, last_used, recent_counts):
    recency = index - last_used.get(page, -1)
    frequency = recent_counts.get(page, 0)
    global_frequency = sum(1 for x in trace[:index] if x == page)
    return [recency, frequency, global_frequency]


def build_training_data(traces, capacity):
    X, y = [], []
    for trace in traces:
        frames = set()
        last_used = {}
        recent = deque()
        recent_counts = Counter()
        for i, page in enumerate(trace):
            if page in frames:
                last_used[page] = i
                recent.append(page); recent_counts[page] += 1
                if len(recent) > RECENT_WINDOW:
                    old = recent.popleft(); recent_counts[old] -= 1
                continue

            if len(frames) == capacity:
                future = trace[i + 1:]
                victim = max(frames, key=lambda p: future.index(p) if p in future else float('inf'))
                for candidate in frames:
                    X.append(candidate_features(trace, i, frames, candidate, last_used, recent_counts))
                    y.append(1 if candidate == victim else 0)
                frames.remove(victim)

            frames.add(page)
            last_used[page] = i
            recent.append(page); recent_counts[page] += 1
            if len(recent) > RECENT_WINDOW:
                old = recent.popleft(); recent_counts[old] -= 1
    return X, y


def learned_policy(trace, capacity, model):
    frames, faults = set(), 0
    last_used = {}
    recent = deque()
    recent_counts = Counter()
    hits_by_phase = [0, 0]
    faults_by_phase = [0, 0]

    for i, page in enumerate(trace):
        phase = 0 if i < PHASE_LENGTH else 1
        if page in frames:
            hits_by_phase[phase] += 1
        else:
            faults += 1
            faults_by_phase[phase] += 1
            if len(frames) == capacity:
                candidates = list(frames)
                features = [candidate_features(trace, i, frames, c, last_used, recent_counts) for c in candidates]
                probabilities = model.predict_proba(features)[:, 1]
                victim = candidates[max(range(len(candidates)), key=lambda j: probabilities[j])]
                frames.remove(victim)
            frames.add(page)
        last_used[page] = i
        recent.append(page); recent_counts[page] += 1
        if len(recent) > RECENT_WINDOW:
            old = recent.popleft(); recent_counts[old] -= 1

    return metrics(trace, faults, hits_by_phase, faults_by_phase)


def main():
    os.makedirs('results', exist_ok=True)
    train_traces = [generate_trace(seed=100 + i)[:PHASE_LENGTH] for i in range(8)]
    X, y = build_training_data(train_traces, FRAMES)
    model = DecisionTreeClassifier(max_depth=5, random_state=SEED)
    model.fit(X, y)

    trace = generate_trace()
    results = {
        'FIFO': fifo(trace, FRAMES),
        'LRU': lru(trace, FRAMES),
        'Optimal': optimal(trace, FRAMES),
        'Learned-DecisionTree': learned_policy(trace, FRAMES, model),
    }

    with open('results/metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['policy', 'faults', 'hit_ratio', 'phase1_faults', 'phase1_hit_ratio', 'phase2_faults', 'phase2_hit_ratio'])
        for name, r in results.items():
            writer.writerow([name, r['faults'], r['hit_ratio'], r['phase1_faults'], r['phase1_hit_ratio'], r['phase2_faults'], r['phase2_hit_ratio']])

    with open('results/trace.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'page', 'phase'])
        for i, page in enumerate(trace):
            writer.writerow([i, page, 1 if i < PHASE_LENGTH else 2])

    print('\nTrack 1 — Learned Page Replacement')
    print(f'Frames: {FRAMES}, pages: {PAGE_COUNT}, accesses: {len(trace)}')
    print(f'Phase 1: locality/sequential ({PHASE_LENGTH} accesses)')
    print(f'Phase 2: random/bursty ({PHASE_LENGTH} accesses)')
    print('\nPolicy                  Faults   Hit ratio   P1 faults   P1 hit   P2 faults   P2 hit')
    for name, r in results.items():
        print(f'{name:22} {r["faults"]:6d}   {r["hit_ratio"]:8.3f}   {r["phase1_faults"]:9d}   {r["phase1_hit_ratio"]:6.3f}   {r["phase2_faults"]:9d}   {r["phase2_hit_ratio"]:6.3f}')


if __name__ == '__main__':
    main()
