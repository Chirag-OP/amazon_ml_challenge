"""
Benchmark LSH construction + query cost on a SMALL subset before running
on the full ~50 lakh records per dataset. Measures:

    - build time
    - peak memory (RSS) during build
    - queries/sec
    - avg candidates returned per query

Run this FIRST. If memory or time projections look bad at this scale,
it's much cheaper to find out now than after a multi-hour full build.
"""

import os
import time
import resource

import pandas as pd

from LSH_bucket import EntityLSHIndex


TRAIN_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
SOURCE1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
SOURCE2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")

# Subset sizes -- adjust as needed. Keep the ratio similar to your real
# data (source2/3 are ~50 lakh, source1 queries against them).
N_QUERY_ROWS = 5_000
N_REFERENCE_ROWS = 50_000


def get_peak_memory_mb():
    """Peak resident set size in MB. Linux: ru_maxrss is in KB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def prepare_dataframe(df):
    df = df.copy()
    df["entity_id"] = df["entity_id"].astype(str)
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["business_address"] = df["business_address"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str).str.strip()
    return df


def benchmark_build(df, label):
    mem_before = get_peak_memory_mb()
    start = time.time()

    index = EntityLSHIndex()
    index.build(df)

    elapsed = time.time() - start
    mem_after = get_peak_memory_mb()

    n_name_minhashes = len(index.minhashes["name"])
    n_addr_minhashes = len(index.minhashes["address"])

    print(f"\n[{label}] rows={len(df)}")
    print(f"  build time: {elapsed:.2f}s  ({len(df) / elapsed:.1f} rows/sec)")
    print(f"  peak RSS before: {mem_before:.1f} MB, after: {mem_after:.1f} MB "
          f"(delta: {mem_after - mem_before:.1f} MB)")
    print(f"  minhashes stored: name={n_name_minhashes}, address={n_addr_minhashes}")

    # Extrapolation to full scale -- rough, but gives an early warning.
    scale_factor = 50_00_000 / len(df)
    print(f"  --> projected at 50 lakh rows: "
          f"~{elapsed * scale_factor / 60:.1f} min build time, "
          f"~{(mem_after - mem_before) * scale_factor / 1024:.2f} GB memory delta")

    return index, elapsed, mem_after - mem_before


def benchmark_query(index, query_df, n_samples=500):
    sample = query_df.sample(min(n_samples, len(query_df)), random_state=42)

    start = time.time()
    total_candidates = 0

    for row in sample.itertuples(index=False):
        result = index.query(
            country=row.country,
            business_name=row.business_name,
            business_address=row.business_address,
        )
        total_candidates += len(result["union"])

    elapsed = time.time() - start
    qps = len(sample) / elapsed if elapsed > 0 else float("inf")
    avg_candidates = total_candidates / len(sample)

    print(f"\n  queries/sec: {qps:.1f}")
    print(f"  avg candidates/query: {avg_candidates:.1f}")
    print(f"  --> projected time for 50 lakh queries: {50_00_000 / qps / 3600:.2f} hours")

    return qps, avg_candidates


if __name__ == "__main__":

    print("Loading subsets...")
    source1 = prepare_dataframe(pd.read_csv(SOURCE1_PATH, sep="\t", nrows=N_QUERY_ROWS))
    source2 = prepare_dataframe(pd.read_csv(SOURCE2_PATH, sep="\t", nrows=N_REFERENCE_ROWS))

    # Country normalization should match pipeline.py's normalize_country()
    # for realistic partition sizes -- using raw values here is fine for
    # a rough benchmark, but note results may differ slightly once
    # normalization consolidates variant country spellings into the same
    # partition.

    index, build_time, build_mem = benchmark_build(source2, f"source2 ({N_REFERENCE_ROWS} rows)")
    benchmark_query(index, source1)

    print("\nDone. Compare the projected numbers above against your available "
          "RAM and time budget before running on the full ~50 lakh dataset. "
          "If the memory projection is too high, consider reducing num_perm "
          "in EntityLSHIndex, or recomputing candidate MinHashes on the fly "
          "at ranking time instead of pre-storing all of them.")