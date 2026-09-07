import json, hashlib, hmac, math, os, platform, time, statistics
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -----------------------
# Reproducible experiment
# -----------------------
TRIAL_SEEDS = list(range(2101, 2121))  # 20 independent trials
N_EVENTS = 20_000
BLOCK_SIZE = 100
N_VALIDATORS = 4
QUORUM = 3
TRAIN_FRAC = 0.70

STAGES = np.array(["farm", "transport", "storage", "processing", "retail"])
STAGE_P = np.array([0.28, 0.20, 0.18, 0.17, 0.17])

# Benchmark policy thresholds; these are scenario rules, not universal legal organic standards.
RULES = {
    "farm": {"temp": (10.0, 38.0), "humidity": (30.0, 90.0), "soil": (25.0, 75.0)},
    "transport": {"temp": (2.0, 12.0), "humidity": (30.0, 90.0)},
    "storage": {"temp": (2.0, 10.0), "humidity": (35.0, 88.0)},
    "processing": {"temp": (5.0, 25.0), "humidity": (25.0, 85.0)},
    "retail": {"temp": (4.0, 18.0), "humidity": (25.0, 85.0)},
}

VALIDATOR_KEYS = [hashlib.sha256(f"validator-{i}-benchmark-key".encode()).digest() for i in range(N_VALIDATORS)]


def canonical_json(obj: Dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def merkle_root(hashes: List[bytes]) -> bytes:
    if not hashes:
        return b"\x00" * 32
    level = hashes[:]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [sha256_bytes(level[i] + level[i+1]) for i in range(0, len(level), 2)]
    return level[0]


def endorse(block_hash: bytes) -> List[bytes]:
    return [hmac.new(k, block_hash, hashlib.sha256).digest() for k in VALIDATOR_KEYS[:QUORUM]]


def verify_endorsements(block_hash: bytes, sigs: List[bytes]) -> bool:
    valid = 0
    for key, sig in zip(VALIDATOR_KEYS, sigs):
        if hmac.compare_digest(hmac.new(key, block_hash, hashlib.sha256).digest(), sig):
            valid += 1
    return valid >= QUORUM


def excess(x, lo, hi):
    if x < lo:
        return (lo - x) / max(abs(lo), 1.0)
    if x > hi:
        return (x - hi) / max(abs(hi), 1.0)
    return 0.0


def compliance_score(e: Dict) -> float:
    r = RULES[e["stage"]]
    score = 0.0
    score += 1.0 * excess(e["temperature_c"], *r["temp"])
    score += 0.7 * excess(e["humidity_pct"], *r["humidity"])
    if "soil" in r and e["soil_moisture_pct"] is not None:
        score += 0.8 * excess(e["soil_moisture_pct"], *r["soil"])
    score += 1.25 * (1 - e["certificate_valid"])
    score += 1.10 * e["custody_gap"]
    score += 0.65 * e["seal_breach"]
    return float(score)


def generate_events(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stage = rng.choice(STAGES, size=n, p=STAGE_P)
    latent_violation = rng.binomial(1, 0.16, size=n)
    latent_unobserved = (latent_violation == 1) & (rng.random(n) < 0.12)

    temperature = np.empty(n)
    humidity = np.empty(n)
    soil = np.full(n, np.nan)
    cert_valid = np.ones(n, dtype=int)
    custody_gap = np.zeros(n, dtype=int)
    seal_breach = np.zeros(n, dtype=int)

    # Compliant-state distributions centered comfortably inside benchmark policy ranges.
    centers = {
        "farm": (24.0, 60.0),
        "transport": (7.0, 65.0),
        "storage": (6.0, 65.0),
        "processing": (16.0, 55.0),
        "retail": (10.0, 55.0),
    }
    temp_sd = {"farm": 4.0, "transport": 2.2, "storage": 1.8, "processing": 4.0, "retail": 3.0}
    hum_sd = {s: 9.0 for s in STAGES}

    for s in STAGES:
        idx = np.where(stage == s)[0]
        temperature[idx] = rng.normal(centers[s][0], temp_sd[s], len(idx))
        humidity[idx] = rng.normal(centers[s][1], hum_sd[s], len(idx))
        if s == "farm":
            soil[idx] = rng.normal(50.0, 11.0, len(idx))

    # Inject observable violations for most non-compliant cases.
    for i in np.where((latent_violation == 1) & (~latent_unobserved))[0]:
        kind = rng.choice(["temp", "humidity", "cert", "custody", "seal", "soil"],
                          p=[0.28, 0.14, 0.20, 0.17, 0.13, 0.08])
        s = stage[i]
        r = RULES[s]
        if kind == "temp":
            lo, hi = r["temp"]
            temperature[i] = hi + rng.uniform(2.0, 10.0) if rng.random() < 0.75 else lo - rng.uniform(2.0, 8.0)
        elif kind == "humidity":
            lo, hi = r["humidity"]
            humidity[i] = hi + rng.uniform(4.0, 18.0) if rng.random() < 0.5 else lo - rng.uniform(4.0, 15.0)
        elif kind == "cert":
            cert_valid[i] = 0
        elif kind == "custody":
            custody_gap[i] = 1
        elif kind == "seal":
            seal_breach[i] = 1
        elif kind == "soil" and s == "farm":
            soil[i] = rng.choice([rng.uniform(5.0, 20.0), rng.uniform(80.0, 95.0)])
        else:
            cert_valid[i] = 0

    # Sensor/record noise can create a small number of false positives among compliant cases.
    noisy = (latent_violation == 0) & (rng.random(n) < 0.025)
    for i in np.where(noisy)[0]:
        if rng.random() < 0.55:
            lo, hi = RULES[stage[i]]["temp"]
            temperature[i] = hi + rng.uniform(0.5, 3.0)
        else:
            custody_gap[i] = 1

    # Hidden violations represent pesticide/input/process issues not directly observed by these IoT features.
    # They remain non-compliant ground truth but may be missed by the rule score.
    df = pd.DataFrame({
        "event_id": np.arange(n, dtype=int),
        "stage": stage,
        "temperature_c": np.round(temperature, 3),
        "humidity_pct": np.round(humidity, 3),
        "soil_moisture_pct": np.where(np.isnan(soil), np.nan, np.round(soil, 3)),
        "certificate_valid": cert_valid,
        "custody_gap": custody_gap,
        "seal_breach": seal_breach,
        "noncompliant": latent_violation.astype(int),
    })
    return df


def row_to_event(row) -> Dict:
    sm = None if pd.isna(row.soil_moisture_pct) else float(row.soil_moisture_pct)
    return {
        "event_id": int(row.event_id),
        "stage": str(row.stage),
        "temperature_c": float(row.temperature_c),
        "humidity_pct": float(row.humidity_pct),
        "soil_moisture_pct": sm,
        "certificate_valid": int(row.certificate_valid),
        "custody_gap": int(row.custody_gap),
        "seal_breach": int(row.seal_breach),
    }


def choose_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    # Deterministic training-only threshold grid; maximize F1, then prefer higher threshold on ties.
    candidates = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 181)))
    best = (-1.0, 0.0)
    for thr in candidates:
        pred = (scores >= thr).astype(int)
        val = f1_score(y, pred, zero_division=0)
        if val > best[0] + 1e-12 or (abs(val - best[0]) <= 1e-12 and thr > best[1]):
            best = (val, float(thr))
    return best[1]


def classification_trial(df: pd.DataFrame) -> Dict:
    scores = np.array([compliance_score(row_to_event(r)) for r in df.itertuples(index=False)])
    y = df["noncompliant"].to_numpy()
    split = int(len(df) * TRAIN_FRAC)
    thr = choose_threshold(scores[:split], y[:split])
    pred = (scores[split:] >= thr).astype(int)
    yt = y[split:]
    return {
        "threshold": thr,
        "accuracy": accuracy_score(yt, pred),
        "precision": precision_score(yt, pred, zero_division=0),
        "recall": recall_score(yt, pred, zero_division=0),
        "f1": f1_score(yt, pred, zero_division=0),
        "auc": roc_auc_score(yt, scores[split:]),
    }


@dataclass
class Block:
    index: int
    prev_hash: bytes
    merkle: bytes
    block_hash: bytes
    signatures: List[bytes]
    event_hashes: List[bytes]


def benchmark_method(events: List[Dict], method: str) -> Dict:
    lat_us = []
    serialized_bytes = 0
    event_hashes = []
    blocks: List[Block] = []
    prev_hash = b"\x00" * 32
    buffer_hashes: List[bytes] = []

    start_all = time.perf_counter_ns()
    for e in events:
        t0 = time.perf_counter_ns()
        payload = canonical_json(e)
        serialized_bytes += len(payload)
        _ = compliance_score(e)

        if method == "centralized":
            # raw append only
            pass
        elif method == "event_hash":
            h = sha256_bytes(payload)
            event_hashes.append(h)
            serialized_bytes += 32
        elif method == "quorum_chain":
            h = sha256_bytes(payload)
            event_hashes.append(h)
            buffer_hashes.append(h)
            serialized_bytes += 32
            if len(buffer_hashes) == BLOCK_SIZE:
                mr = merkle_root(buffer_hashes)
                idx = len(blocks)
                header = idx.to_bytes(8, "big") + prev_hash + mr
                bh = sha256_bytes(header)
                sigs = endorse(bh)
                # verify quorum before commit
                if not verify_endorsements(bh, sigs):
                    raise RuntimeError("endorsement verification failed")
                blocks.append(Block(idx, prev_hash, mr, bh, sigs, buffer_hashes[:]))
                prev_hash = bh
                # logical binary metadata: block index + prev + merkle + block hash + 3 HMAC signatures
                serialized_bytes += 8 + 32 + 32 + 32 + QUORUM * 32
                buffer_hashes = []
        else:
            raise ValueError(method)
        lat_us.append((time.perf_counter_ns() - t0) / 1000.0)

    if method == "quorum_chain" and buffer_hashes:
        mr = merkle_root(buffer_hashes)
        idx = len(blocks)
        header = idx.to_bytes(8, "big") + prev_hash + mr
        bh = sha256_bytes(header)
        sigs = endorse(bh)
        if not verify_endorsements(bh, sigs):
            raise RuntimeError("endorsement verification failed")
        blocks.append(Block(idx, prev_hash, mr, bh, sigs, buffer_hashes[:]))
        serialized_bytes += 8 + 32 + 32 + 32 + QUORUM * 32

    total_s = (time.perf_counter_ns() - start_all) / 1e9
    arr = np.asarray(lat_us)
    return {
        "method": method,
        "mean_us": float(arr.mean()),
        "p95_us": float(np.percentile(arr, 95)),
        "p99_us": float(np.percentile(arr, 99)),
        "throughput_eps": float(len(events) / total_s),
        "bytes_per_event": float(serialized_bytes / len(events)),
        "blocks": blocks,
        "event_hashes": event_hashes,
    }


def verify_chain(blocks: List[Block]) -> bool:
    prev = b"\x00" * 32
    for b in blocks:
        if b.prev_hash != prev:
            return False
        mr = merkle_root(b.event_hashes)
        if mr != b.merkle:
            return False
        header = b.index.to_bytes(8, "big") + b.prev_hash + b.merkle
        bh = sha256_bytes(header)
        if bh != b.block_hash:
            return False
        if not verify_endorsements(bh, b.signatures):
            return False
        prev = b.block_hash
    return True


def tamper_stress_test(events: List[Dict], chain_blocks: List[Block], event_hashes: List[bytes], seed: int) -> Dict:
    rng = np.random.default_rng(seed + 99999)
    # Select one attack of each class. Report whether the mechanism has evidence to detect it.
    idx = int(rng.integers(0, len(events)))
    modified = events[idx].copy()
    modified["temperature_c"] += 7.0
    payload_mod_detect_central = False
    payload_mod_detect_hash = sha256_bytes(canonical_json(modified)) != event_hashes[idx]

    # For the chain, mutate an event hash inside its block and verify failure.
    bidx = idx // BLOCK_SIZE
    pos = idx % BLOCK_SIZE
    mutated_blocks = [Block(b.index, b.prev_hash, b.merkle, b.block_hash, b.signatures[:], b.event_hashes[:]) for b in chain_blocks]
    if bidx < len(mutated_blocks) and pos < len(mutated_blocks[bidx].event_hashes):
        mutated_blocks[bidx].event_hashes[pos] = sha256_bytes(canonical_json(modified))
    payload_mod_detect_chain = not verify_chain(mutated_blocks)

    # Deletion and reordering are not detectable in a list of independent hashes without a trusted sequence manifest.
    deletion_detect_central = False
    deletion_detect_hash = False
    reorder_detect_central = False
    reorder_detect_hash = False

    # Chain deletion: remove one event hash from a block; Merkle root fails.
    del_blocks = [Block(b.index, b.prev_hash, b.merkle, b.block_hash, b.signatures[:], b.event_hashes[:]) for b in chain_blocks]
    if del_blocks and del_blocks[bidx].event_hashes:
        del del_blocks[bidx].event_hashes[min(pos, len(del_blocks[bidx].event_hashes)-1)]
    deletion_detect_chain = not verify_chain(del_blocks)

    # Chain reordering: swap two event hashes in the same block; ordered Merkle root fails.
    re_blocks = [Block(b.index, b.prev_hash, b.merkle, b.block_hash, b.signatures[:], b.event_hashes[:]) for b in chain_blocks]
    rb = re_blocks[bidx]
    if len(rb.event_hashes) >= 2:
        i1, i2 = 0, 1
        rb.event_hashes[i1], rb.event_hashes[i2] = rb.event_hashes[i2], rb.event_hashes[i1]
    reorder_detect_chain = not verify_chain(re_blocks)

    return {
        "payload_mod": [payload_mod_detect_central, payload_mod_detect_hash, payload_mod_detect_chain],
        "deletion": [deletion_detect_central, deletion_detect_hash, deletion_detect_chain],
        "reordering": [reorder_detect_central, reorder_detect_hash, reorder_detect_chain],
    }


def ci95(vals: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    h = float(t.ppf(0.975, len(arr)-1) * se)
    return mean, mean-h, mean+h


def main():
    cls_rows = []
    bench_rows = []
    tamper_counts = {a: np.zeros(3, dtype=int) for a in ["payload_mod", "deletion", "reordering"]}

    for seed in TRIAL_SEEDS:
        df = generate_events(seed, N_EVENTS)
        cls = classification_trial(df)
        cls["seed"] = seed
        cls_rows.append(cls)

        events = [row_to_event(r) for r in df.itertuples(index=False)]
        results = {}
        for method in ["centralized", "event_hash", "quorum_chain"]:
            br = benchmark_method(events, method)
            results[method] = br
            bench_rows.append({k: v for k, v in br.items() if k not in ["blocks", "event_hashes"]} | {"seed": seed})

        attacks = tamper_stress_test(events, results["quorum_chain"]["blocks"], results["event_hash"]["event_hashes"], seed)
        for a, vals in attacks.items():
            tamper_counts[a] += np.array(vals, dtype=int)

    cls_df = pd.DataFrame(cls_rows)
    bench_df = pd.DataFrame(bench_rows)

    print("SYSTEM")
    print({"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor()})
    print("\nCLASSIFICATION SUMMARY (mean [95% CI])")
    for m in ["accuracy", "precision", "recall", "f1", "auc", "threshold"]:
        mean, lo, hi = ci95(cls_df[m].tolist())
        print(f"{m:10s}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    print("\nAPPLICATION-LAYER MICROBENCHMARK (trial means, mean [95% CI])")
    for method in ["centralized", "event_hash", "quorum_chain"]:
        sub = bench_df[bench_df.method == method]
        print("\n", method)
        for m in ["mean_us", "p95_us", "p99_us", "throughput_eps", "bytes_per_event"]:
            mean, lo, hi = ci95(sub[m].tolist())
            print(f"{m:16s}: {mean:.3f} [{lo:.3f}, {hi:.3f}]")

    print("\nTAMPER STRESS TEST detection rate over trials")
    labels = ["centralized", "event_hash", "quorum_chain"]
    for a, c in tamper_counts.items():
        print(a, {lab: float(v/len(TRIAL_SEEDS)) for lab, v in zip(labels, c)})

    outdir = "/mnt/data/organic_traceability_results"
    os.makedirs(outdir, exist_ok=True)
    cls_df.to_csv(os.path.join(outdir, "classification_trials.csv"), index=False)
    bench_df.to_csv(os.path.join(outdir, "benchmark_trials.csv"), index=False)
    tamper_df = pd.DataFrame([
        {"attack": a, "method": lab, "detection_rate": float(c[i]/len(TRIAL_SEEDS))}
        for a, c in tamper_counts.items() for i, lab in enumerate(labels)
    ])
    tamper_df.to_csv(os.path.join(outdir, "tamper_detection.csv"), index=False)

    # Produce summary tables for direct manuscript use.
    cls_summary = []
    for m in ["accuracy", "precision", "recall", "f1", "auc"]:
        mean, lo, hi = ci95(cls_df[m].tolist())
        cls_summary.append({"metric": m, "mean": mean, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(cls_summary).to_csv(os.path.join(outdir, "classification_summary.csv"), index=False)

    bench_summary = []
    for method in ["centralized", "event_hash", "quorum_chain"]:
        sub = bench_df[bench_df.method == method]
        row = {"method": method}
        for m in ["mean_us", "p95_us", "p99_us", "throughput_eps", "bytes_per_event"]:
            mean, lo, hi = ci95(sub[m].tolist())
            row[m] = mean; row[m+"_lo"] = lo; row[m+"_hi"] = hi
        bench_summary.append(row)
    pd.DataFrame(bench_summary).to_csv(os.path.join(outdir, "benchmark_summary.csv"), index=False)


if __name__ == "__main__":
    main()
