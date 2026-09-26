import os
import json
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from rapidfuzz import fuzz
from sklearn.model_selection import GroupKFold
from sklearn.metrics import fbeta_score

from LSH_bucket import EntityLSHIndex
from embedding_store import EmbeddingStore, dot_sim
from transformer import (
    CharTokenizer,
    FieldEmbeddingTransformer,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

SOURCE1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
SOURCE2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
SOURCE3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")

# Adjust this filename to match whatever your ground truth file is
# actually called.
GROUND_TRUTH_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

# Test-set source1 file to generate the final submission for.
TEST_SOURCE1_PATH = os.path.join(TEST_DIR, "test_source1.tsv")

SUBMISSION_PATH = "submission.tsv"

# Change these after training your Transformer models
NAME_MODEL_PATH = "models/name_model.pt"
ADDRESS_MODEL_PATH = "models/address_model.pt"

XGB_MODEL_PATH = "models/xgb_classifier.json"
XGB_META_PATH = "models/xgb_meta.json"

TOP_K = 50

# Batch size for transformer forward passes during embedding precompute.
# At 30-50 lakh rows, computing embeddings one row at a time would take
# far too long -- always batch.
EMBED_BATCH_SIZE = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)


# ============================================================
# 1. LOAD DATA (train split -- used to build labeled candidates
#    for training the final classifier)
# ============================================================

print("\nLoading datasets...")

source1 = pd.read_csv(SOURCE1_PATH, sep="\t")
source2 = pd.read_csv(SOURCE2_PATH, sep="\t")
source3 = pd.read_csv(SOURCE3_PATH, sep="\t")

print("Source 1:", source1.shape)
print("Source 2:", source2.shape)
print("Source 3:", source3.shape)


# ============================================================
# 2. BASIC CLEANING
# ============================================================

def prepare_dataframe(df):
    df = df.copy()

    df["entity_id"] = df["entity_id"].astype(str)

    df["business_name"] = (
        df["business_name"]
        .fillna("")
        .astype(str)
    )

    df["business_address"] = (
        df["business_address"]
        .fillna("")
        .astype(str)
    )

    df["country"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df


source1 = prepare_dataframe(source1)
source2 = prepare_dataframe(source2)
source3 = prepare_dataframe(source3)


# ============================================================
# 3. COUNTRY NORMALIZATION
# ============================================================
#
# IMPORTANT:
# We are NOT extracting states.
#
# Country is an existing column, so we can safely normalize
# known textual variations.
#
# Do not hardcode states/cities here.
# ============================================================

COUNTRY_MAP = {
    "us": "US",
    "u.s.": "US",
    "u.s": "US",
    "usa": "US",
    "united states": "US",
    "united states of america": "US",

    "india": "INDIA",

    "uk": "UK",
    "u.k.": "UK",
    "united kingdom": "UK",

    "canada": "CANADA",
    "australia": "AUSTRALIA",
}


def normalize_country(country):
    country = str(country).strip().lower()

    return COUNTRY_MAP.get(
        country,
        country.upper()
    )


source1["country"] = source1["country"].apply(normalize_country)
source2["country"] = source2["country"].apply(normalize_country)
source3["country"] = source3["country"].apply(normalize_country)


# ============================================================
# 4. TRANSFORMER MODELS + TOKENIZERS
# ============================================================

# Name and address have separate models.
#
# Name:
#     max_len = 64
#
# Address:
#     max_len = 256
#
# Your address dataset has maximum length 256, so we preserve
# the complete address.

name_tokenizer = CharTokenizer(max_len=64)

address_tokenizer = CharTokenizer(max_len=256)


def load_model(checkpoint_path, tokenizer):
    """
    Load a FieldEmbeddingTransformer checkpoint.

    IMPORTANT: d_model / nhead / num_layers / embed_dim here must match
    whatever was used when the checkpoint was TRAINED, or load_state_dict
    will fail with a shape mismatch. Only vocab_size and max_len are
    tokenizer-derived; the rest are architecture choices you made at
    training time -- keep them in sync.
    """

    model = FieldEmbeddingTransformer(
        vocab_size=tokenizer.vocab_size,
        max_len=tokenizer.max_len,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # Handles either:
    #   torch.save(model.state_dict(), ...)
    # or:
    #   torch.save({"model_state_dict": ...}, ...)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# 5. BATCHED EMBEDDING PRECOMPUTE
# ============================================================
#
# FIX: the original per-row loop called the transformer once per record.
# At 30-50 lakh rows x 2 fields x 3 datasets that's tens of millions of
# batch-size-1 forward passes -- won't finish in reasonable time. Always
# batch the encode + forward pass.
# ============================================================

def batch_get_embeddings(model, tokenizer, texts, batch_size=EMBED_BATCH_SIZE):
    """
    Compute embeddings for many texts via batched forward passes.
    Returns a plain NumPy array, not a list of tensors -- this is what
    lets precompute_embeddings build one contiguous EmbeddingStore
    instead of millions of individually-allocated tensor objects.
    """
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            ids = tokenizer.batch_encode(batch_texts).to(DEVICE)
            batch_emb = model(ids)
            all_embeddings.append(batch_emb.cpu().numpy())

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32)

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def precompute_embeddings(df, name_model, address_model, batch_size=EMBED_BATCH_SIZE):
    """
    Compute Transformer embeddings once for every entity in a dataframe.

    Returns:
        name_store: EmbeddingStore
        address_store: EmbeddingStore

    NOT a dict of {id: tensor} -- at 50 lakh rows that costs several
    times more memory than the raw floats. See embedding_store.py.
    """

    entity_ids = df["entity_id"].tolist()
    names = df["business_name"].tolist()
    addresses = df["business_address"].tolist()

    name_vecs = batch_get_embeddings(name_model, name_tokenizer, names, batch_size)
    address_vecs = batch_get_embeddings(address_model, address_tokenizer, addresses, batch_size)

    name_store = EmbeddingStore(entity_ids, name_vecs)
    address_store = EmbeddingStore(entity_ids, address_vecs)

    print(
        f"  name embeddings: {name_store.nbytes() / 1e6:.1f} MB, "
        f"address embeddings: {address_store.nbytes() / 1e6:.1f} MB "
        f"({len(entity_ids)} rows)"
    )

    return name_store, address_store


# ============================================================
# 6. RANK LSH CANDIDATES USING MINHASH
# ============================================================

def rank_lsh_candidates(
    index,
    query_name,
    query_address,
    candidates,
    top_k=50,
):
    """
    Candidates have already been retrieved by LSH.

    We now calculate actual MinHash/Jaccard similarity for those
    candidates and keep only the top K. No Transformer is used here.

    FIX: compute the query's own MinHash ONCE, outside the per-candidate
    loop, instead of recomputing it for every candidate.
    """

    query_name_minhash = index.get_minhash(query_name)
    query_address_minhash = index.get_minhash(query_address)

    ranked = []

    for candidate_id in candidates:

        name_sim = index.similarity_to_candidate(
            "name",
            query_name_minhash,
            candidate_id
        )

        address_sim = index.similarity_to_candidate(
            "address",
            query_address_minhash,
            candidate_id
        )

        # Retrieval score: use the stronger of the two signals so a
        # candidate with a very strong name OR address match isn't
        # discarded too early.
        retrieval_score = max(
            name_sim,
            address_sim
        )

        ranked.append(
            (
                retrieval_score,
                name_sim,
                address_sim,
                candidate_id,
            )
        )

    ranked.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return ranked[:top_k]


# ============================================================
# 7. TRANSFORMER RANKING
# ============================================================

def rank_with_transformer(
    query_name_vec,
    query_address_vec,
    candidates,
    name_store,
    address_store,
):
    """
    Calculate Transformer cosine similarities for the already-retrieved
    candidates. name_store/address_store are EmbeddingStore instances;
    dot_sim() is a plain NumPy dot product (valid since embeddings are
    L2-normalized), which is also faster than calling torch's
    cosine_similarity per pair -- no tensor wrapping overhead.
    """

    results = []

    for candidate_id in candidates:

        name_sim = dot_sim(query_name_vec, name_store.get(candidate_id))
        address_sim = dot_sim(query_address_vec, address_store.get(candidate_id))

        results.append({
            "entity_id": candidate_id,
            "name_similarity": name_sim,
            "address_similarity": address_sim,
        })

    return results


# ============================================================
# 8. PROCESS ONE QUERY
# ============================================================

def process_query(
    query_id,
    query_name,
    query_address,
    query_country,
    lsh_index,
    name_embeddings,
    address_embeddings,
    query_name_embedding,
    query_address_embedding,
    top_k=50,
):
    """
    Process one Dataset 1 entity against one reference dataset.

    FIX: lsh_index.query() takes (country, business_name, business_address)
    -- called here with keyword args so field order can never be silently
    swapped again. Query embeddings are passed in (precomputed upfront),
    not recomputed per row.
    """

    lsh_results = lsh_index.query(
        country=query_country,
        business_name=query_name,
        business_address=query_address,
    )

    candidates = lsh_results["union"]

    if not candidates:
        return []

    # --------------------------------------------------------
    # MinHash ranking
    # --------------------------------------------------------

    ranked_candidates = rank_lsh_candidates(
        index=lsh_index,
        query_name=query_name,
        query_address=query_address,
        candidates=candidates,
        top_k=top_k,
    )

    candidate_ids = [
        item[3]
        for item in ranked_candidates
    ]

    # --------------------------------------------------------
    # Transformer similarity
    # --------------------------------------------------------

    transformer_results = rank_with_transformer(
        query_name_embedding,
        query_address_embedding,
        candidate_ids,
        name_embeddings,
        address_embeddings,
    )

    minhash_lookup = {
        item[3]: {
            "minhash_name": item[1],
            "minhash_address": item[2],
        }
        for item in ranked_candidates
    }

    for result in transformer_results:

        result["query_id"] = query_id

        result.update(
            minhash_lookup[result["entity_id"]]
        )

    return transformer_results


# ============================================================
# 9. MAIN CANDIDATE-GENERATION PIPELINE
# ============================================================

def run_pipeline(
    source1_df,
    lsh_source2,
    lsh_source3,
    name_embeddings_d1,
    address_embeddings_d1,
    name_embeddings_d2,
    address_embeddings_d2,
    name_embeddings_d3,
    address_embeddings_d3,
    top_k=50,
):
    """
    Run entity resolution candidate generation for every record in
    source1_df. Dataset 2 and Dataset 3 are treated independently.

    Works for EITHER the train split or the test split -- pass in
    whichever source1_df + matching precomputed d1 embeddings.
    """

    all_results = []
    total = len(source1_df)

    for i, row in enumerate(source1_df.itertuples(index=False)):

        if i % 1000 == 0:
            print(f"Processing {i}/{total}")

        query_id = row.entity_id
        query_name = row.business_name
        query_address = row.business_address
        query_country = row.country

        query_name_embedding = name_embeddings_d1[query_id]
        query_address_embedding = address_embeddings_d1[query_id]

        # Dataset 2
        results_d2 = process_query(
            query_id=query_id,
            query_name=query_name,
            query_address=query_address,
            query_country=query_country,
            lsh_index=lsh_source2,
            name_embeddings=name_embeddings_d2,
            address_embeddings=address_embeddings_d2,
            query_name_embedding=query_name_embedding,
            query_address_embedding=query_address_embedding,
            top_k=top_k,
        )
        for result in results_d2:
            result["source"] = "source2"

        # Dataset 3
        results_d3 = process_query(
            query_id=query_id,
            query_name=query_name,
            query_address=query_address,
            query_country=query_country,
            lsh_index=lsh_source3,
            name_embeddings=name_embeddings_d3,
            address_embeddings=address_embeddings_d3,
            query_name_embedding=query_name_embedding,
            query_address_embedding=query_address_embedding,
            top_k=top_k,
        )
        for result in results_d3:
            result["source"] = "source3"

        all_results.extend(results_d2)
        all_results.extend(results_d3)

    return pd.DataFrame(all_results)


# ============================================================
# 10. GROUND TRUTH LOADING + LABELING
# ============================================================
#
# Ground truth format:
#   source1_entity_id \t matched_entity_ids
# matched_entity_ids is a comma-separated string of IDs from source2
# and/or source3 (already prefixed, e.g. "S2-...", "S3-..."). Blank
# means the source1 entity is a singleton (no true match).
# ============================================================

def load_ground_truth(path):
    """Returns dict: {source1_entity_id: set(matched_entity_ids)}."""

    gt_df = pd.read_csv(path, sep="\t", dtype=str)
    gt_df["source1_entity_id"] = gt_df["source1_entity_id"].astype(str)
    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")

    ground_truth = {}
    for row in gt_df.itertuples(index=False):
        ids_str = row.matched_entity_ids.strip()
        if ids_str:
            matched = set(x.strip() for x in ids_str.split(",") if x.strip())
        else:
            matched = set()
        ground_truth[row.source1_entity_id] = matched

    return ground_truth


def label_candidates(candidates_df, ground_truth):
    """
    Adds a 'label' column: 1 if (query_id, entity_id) is a true match
    per ground truth, else 0. Only keeps rows whose query_id is present
    in ground_truth (nothing to train on otherwise).

    NOTE: this only labels candidates that SURVIVED blocking. If a true
    match never made it into the candidate pool (a blocking recall miss),
    it simply won't appear here as a positive -- check your Pairs
    Completeness against ground_truth separately; no classifier can fix
    a match that was never retrieved.
    """

    candidates_df = candidates_df[
        candidates_df["query_id"].isin(ground_truth.keys())
    ].copy()

    labels = [
        int(eid in ground_truth.get(qid, set()))
        for qid, eid in zip(candidates_df["query_id"], candidates_df["entity_id"])
    ]
    candidates_df["label"] = labels

    return candidates_df


# ============================================================
# 11. FEATURE ENGINEERING
# ============================================================

def build_text_lookup(*dfs):
    """id -> (business_name, business_address), across any dataframes."""
    lookup = {}
    for df in dfs:
        for row in df.itertuples(index=False):
            lookup[row.entity_id] = (row.business_name, row.business_address)
    return lookup


def add_string_similarity_features(candidates_df, text_lookup):
    """
    Adds Levenshtein ratio (name, address) and name token Jaccard as
    extra features alongside the transformer/minhash scores already in
    candidates_df. Cheap, and catches cases the learned embeddings miss.
    """

    name_lev, addr_lev, name_jac = [], [], []

    for row in candidates_df.itertuples(index=False):
        q_name, q_addr = text_lookup.get(row.query_id, ("", ""))
        c_name, c_addr = text_lookup.get(row.entity_id, ("", ""))

        name_lev.append(fuzz.ratio(q_name, c_name) / 100)
        addr_lev.append(fuzz.ratio(q_addr, c_addr) / 100)

        q_tokens = set(q_name.lower().split())
        c_tokens = set(c_name.lower().split())
        union = q_tokens | c_tokens
        name_jac.append(len(q_tokens & c_tokens) / len(union) if union else 0.0)

    candidates_df = candidates_df.copy()
    candidates_df["name_levenshtein"] = name_lev
    candidates_df["address_levenshtein"] = addr_lev
    candidates_df["name_jaccard"] = name_jac

    return candidates_df


def add_margin_feature(candidates_df):
    """
    For each query, how much stronger is its best candidate than its
    second-best? A small margin signals ambiguity -- exactly the case
    where predicting "no match" protects F_0.5 precision.
    """

    candidates_df = candidates_df.copy()
    candidates_df["combo_score"] = (
        candidates_df["name_similarity"] + candidates_df["address_similarity"]
    ) / 2

    def _margin(s):
        if len(s) < 2:
            return 0.0
        top2 = s.nlargest(2).values
        return float(top2[0] - top2[1])

    candidates_df["score_margin"] = candidates_df.groupby("query_id")[
        "combo_score"
    ].transform(_margin)

    return candidates_df


def trim_candidates_per_query(candidates_df, max_per_query=20):
    """
    NOT called by default yet -- deferred until candidate counts from a
    real run are known (see benchmark_lsh.py and the TOP_K setting).

    At 50 lakh queries x up to 100 candidates each (TOP_K=50 from each
    of source2/source3), add_string_similarity_features() would run
    RapidFuzz on ~500 million pairs. If that proves too slow once you
    measure it, call this BEFORE add_string_similarity_features() to
    keep only the top N candidates per query by transformer similarity
    -- cuts pairs by up to 5x with minimal recall loss, since a true
    match is very unlikely to also have a weak transformer score.
    """
    candidates_df = candidates_df.copy()
    candidates_df["_combo"] = (
        candidates_df["name_similarity"] + candidates_df["address_similarity"]
    ) / 2
    trimmed = (
        candidates_df.sort_values("_combo", ascending=False)
        .groupby("query_id", group_keys=False)
        .head(max_per_query)
    )
    return trimmed.drop(columns=["_combo"])


FEATURE_COLUMNS = [
    "name_similarity", "address_similarity",
    "minhash_name", "minhash_address",
    "name_levenshtein", "address_levenshtein", "name_jaccard",
    "score_margin",
]


# ============================================================
# 12. XGBOOST CLASSIFIER: TRAIN + THRESHOLD TUNING
# ============================================================

def train_and_tune_classifier(train_df, feature_columns=FEATURE_COLUMNS, n_splits=5):
    """
    GroupKFold by query_id (so a query's candidates never leak across
    train/val), out-of-fold predictions used to sweep the decision
    threshold for F_0.5 (this competition rewards precision over
    recall -- do NOT tune for F1 or accuracy here). Final model is
    retrained on all labeled data at the chosen threshold.
    """

    X = train_df[feature_columns].values
    y = train_df["label"].values
    groups = train_df["query_id"].values

    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(train_df))

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )
        model.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    best_thresh, best_f05 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (oof_preds >= t).astype(int)
        f05 = fbeta_score(y, preds, beta=0.5, zero_division=0)
        if f05 > best_f05:
            best_f05, best_thresh = f05, t

    print(f"Best threshold={best_thresh:.2f}  OOF F0.5={best_f05:.4f}")

    final_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    final_model.fit(X, y)

    return final_model, best_thresh


def save_classifier(model, threshold, model_path=XGB_MODEL_PATH, meta_path=XGB_META_PATH):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    with open(meta_path, "w") as f:
        json.dump({"threshold": threshold, "feature_columns": FEATURE_COLUMNS}, f)


def load_classifier(model_path=XGB_MODEL_PATH, meta_path=XGB_META_PATH):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta["threshold"], meta["feature_columns"]


# ============================================================
# 13. WRITE SUBMISSION FILE
# ============================================================

def write_submission(
    all_source1_ids,
    candidates_df,
    classifier,
    threshold,
    feature_columns,
    output_path=SUBMISSION_PATH,
):
    """
    Every candidate above `threshold` is kept (multi-match, not just
    top-1 -- your ground truth format allows several true matches per
    source1 entity). Every source1 id is written, even ones with zero
    LSH candidates -- those get a blank match, which is worth full
    credit for correctly-predicted singletons.
    """

    if len(candidates_df) > 0:
        X = candidates_df[feature_columns].values
        candidates_df = candidates_df.copy()
        candidates_df["match_proba"] = classifier.predict_proba(X)[:, 1]

        matched = candidates_df[candidates_df["match_proba"] >= threshold]
        grouped = matched.groupby("query_id")["entity_id"].apply(
            lambda ids: ",".join(sorted(ids))
        ).to_dict()
    else:
        grouped = {}

    rows = []
    for source1_id in all_source1_ids:
        rows.append({
            "source1_entity_id": source1_id,
            "matched_entity_ids": grouped.get(source1_id, ""),
        })

    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(output_path, sep="\t", index=False)
    print(f"Submission written to {output_path}")


# ============================================================
# 14. MAIN
# ============================================================

if __name__ == "__main__":

    print("\nLoading trained Transformer models...")
    name_model = load_model(NAME_MODEL_PATH, name_tokenizer)
    address_model = load_model(ADDRESS_MODEL_PATH, address_tokenizer)

    # --------------------------------------------------------
    # FIX: EntityLSHIndex() takes no dataframe in its constructor --
    # you must call .build(df) separately, or self.indexes stays {}
    # and every query silently returns zero candidates forever.
    # --------------------------------------------------------

    print("\nBuilding Dataset 2 LSH index...")
    lsh_source2 = EntityLSHIndex()
    lsh_source2.build(source2)

    print("Building Dataset 3 LSH index...")
    lsh_source3 = EntityLSHIndex()
    lsh_source3.build(source3)

    print("\nPrecomputing Dataset 2 embeddings...")
    name_embeddings_d2, address_embeddings_d2 = precompute_embeddings(
        source2, name_model, address_model
    )

    print("Precomputing Dataset 3 embeddings...")
    name_embeddings_d3, address_embeddings_d3 = precompute_embeddings(
        source3, name_model, address_model
    )

    # ==========================================================
    # TRAIN: generate candidates for train source1, label with
    # ground truth, train + tune the XGBoost classifier
    # ==========================================================

    print("\nPrecomputing TRAIN source1 embeddings...")
    name_embeddings_train, address_embeddings_train = precompute_embeddings(
        source1, name_model, address_model
    )

    print("Generating TRAIN candidates...")
    train_candidates = run_pipeline(
        source1_df=source1,
        lsh_source2=lsh_source2,
        lsh_source3=lsh_source3,
        name_embeddings_d1=name_embeddings_train,
        address_embeddings_d1=address_embeddings_train,
        name_embeddings_d2=name_embeddings_d2,
        address_embeddings_d2=address_embeddings_d2,
        name_embeddings_d3=name_embeddings_d3,
        address_embeddings_d3=address_embeddings_d3,
        top_k=TOP_K,
    )

    print("Loading ground truth and labeling candidates...")
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    train_candidates = label_candidates(train_candidates, ground_truth)

    print(
        f"Labeled candidates: {len(train_candidates)} rows, "
        f"{train_candidates['label'].sum()} positive"
    )

    text_lookup = build_text_lookup(source1, source2, source3)
    train_candidates = add_string_similarity_features(train_candidates, text_lookup)
    train_candidates = add_margin_feature(train_candidates)

    print("Training XGBoost classifier...")
    classifier, threshold = train_and_tune_classifier(train_candidates)
    save_classifier(classifier, threshold)

    # ==========================================================
    # TEST: generate candidates for test source1, score with the
    # trained classifier, write the submission file
    # ==========================================================

    print("\nLoading TEST source1...")
    test_source1 = pd.read_csv(TEST_SOURCE1_PATH, sep="\t")
    test_source1 = prepare_dataframe(test_source1)
    test_source1["country"] = test_source1["country"].apply(normalize_country)

    print("Precomputing TEST source1 embeddings...")
    name_embeddings_test, address_embeddings_test = precompute_embeddings(
        test_source1, name_model, address_model
    )

    print("Generating TEST candidates...")
    test_candidates = run_pipeline(
        source1_df=test_source1,
        lsh_source2=lsh_source2,
        lsh_source3=lsh_source3,
        name_embeddings_d1=name_embeddings_test,
        address_embeddings_d1=address_embeddings_test,
        name_embeddings_d2=name_embeddings_d2,
        address_embeddings_d2=address_embeddings_d2,
        name_embeddings_d3=name_embeddings_d3,
        address_embeddings_d3=address_embeddings_d3,
        top_k=TOP_K,
    )

    test_text_lookup = build_text_lookup(test_source1, source2, source3)
    test_candidates = add_string_similarity_features(test_candidates, test_text_lookup)
    test_candidates = add_margin_feature(test_candidates)

    print("Writing submission...")
    write_submission(
        all_source1_ids=test_source1["entity_id"].tolist(),
        candidates_df=test_candidates,
        classifier=classifier,
        threshold=threshold,
        feature_columns=FEATURE_COLUMNS,
        output_path=SUBMISSION_PATH,
    )

    print("\nDone.")