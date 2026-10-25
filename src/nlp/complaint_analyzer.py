"""NLP Complaint Intelligence Module for Retention Copilot.

Responsibilities:
1. Generate dense semantic embeddings from pre-cutoff customer complaint texts using
   `sentence-transformers` (`all-MiniLM-L6-v2`) when installed, or a deterministic
   domain-aware TF-IDF + Truncated SVD semantic embedding engine when executing
   offline without HuggingFace weights.
2. Cluster customer complaint embeddings into interpretable business themes using
   Seeded / K-Means++ clustering with optimal bipartite Hungarian matching (`scipy.optimize.linear_sum_assignment`)
   to ensure 5 distinct, non-overlapping complaint themes.
3. Extract cluster keywords, prevalence, average severity, resolution time, and empirical
   churn rate per complaint theme.
4. Append leak-free NLP features (`complaint_cluster_id`, cluster similarity scores, and
   top semantic embedding components) to the customer feature table so churn models can be
   trained and compared **with and without** complaint-derived NLP features.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.utils.config_loader import get_logger, get_project_root, load_config

logger = get_logger("nlp.complaint_analyzer")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = False


STOPWORDS = {
    "the", "and", "for", "with", "our", "after", "without", "from", "that", "this",
    "was", "were", "are", "has", "have", "had", "not", "but", "too", "very", "into",
    "plan", "severity", "resolution", "wait", "basic", "standard", "enterprise",
}

THEME_BUSINESS_LABELS: Dict[str, str] = {
    "billing_and_pricing": "Billing Surprises, Overage Fees & Renewal Pricing",
    "technical_outage_and_bugs": "Platform Outages, API Latency & Sync Bugs",
    "support_responsiveness": "Slow Support SLA, Escalations & Unresolved Tickets",
    "missing_features_and_ux": "Missing Integrations, Reporting UX & Admin Controls",
    "account_and_contract_flexibility": "Rigid Contract Lock-In & Seat Reduction Policies",
}

THEME_ANCHOR_KEYWORDS: Dict[str, List[str]] = {
    "billing_and_pricing": ["price", "invoice", "charged", "billing", "overage", "refund", "fee", "discount"],
    "technical_outage_and_bugs": ["timeouts", "gateway", "sync", "authentication", "crashes", "latency", "server", "errors"],
    "support_responsiveness": ["waiting", "support", "agent", "transferred", "representatives", "onboarding", "escalation", "queue"],
    "missing_features_and_ux": ["permissions", "webhook", "interface", "confusing", "warehouse", "inflexible", "scheduled", "exports"],
    "account_and_contract_flexibility": ["lock", "seasonal", "pause", "cancellation", "notice", "minimum", "commitment", "procurement"],
}


@dataclass
class ComplaintAnalysisResult:
    """Outputs of the NLP complaint embedding and clustering pipeline."""

    enriched_features_df: pd.DataFrame
    cluster_summary_df: pd.DataFrame
    nlp_feature_columns: List[str]
    embedding_backend: str


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _compute_tfidf_svd_embeddings(
    texts: List[str],
    embedding_dim: int = 16,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """Compute L2-normalized TF-IDF + SVD semantic embeddings using NumPy/SciPy."""
    vocab_counts: Dict[str, int] = {}
    tokenized_docs: List[List[str]] = []
    for txt in texts:
        toks = _tokenize(txt)
        tokenized_docs.append(toks)
        for t in set(toks):
            vocab_counts[t] = vocab_counts.get(t, 0) + 1

    sorted_vocab = sorted(vocab_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:250]
    vocab = [w for w, _ in sorted_vocab]
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    n_docs = len(texts)
    n_vocab = max(len(vocab), 1)
    tf_matrix = np.zeros((n_docs, n_vocab), dtype=np.float32)
    doc_freq = np.zeros(n_vocab, dtype=np.float32)

    for d_i, toks in enumerate(tokenized_docs):
        if not toks:
            continue
        for t in toks:
            idx = word_to_idx.get(t)
            if idx is not None:
                tf_matrix[d_i, idx] += 1.0
        for t in set(toks):
            idx = word_to_idx.get(t)
            if idx is not None:
                doc_freq[idx] += 1.0

    row_sums = np.maximum(tf_matrix.sum(axis=1, keepdims=True), 1.0)
    tf_norm = tf_matrix / row_sums
    idf = np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0
    tfidf = tf_norm * idf

    non_empty_mask = tf_matrix.sum(axis=1) > 0
    k = min(embedding_dim, tfidf.shape[1], max(2, int(non_empty_mask.sum()) - 1))
    if non_empty_mask.sum() > k:
        sample_idx = np.where(non_empty_mask)[0]
        rng = np.random.default_rng(seed)
        fit_idx = rng.choice(sample_idx, size=min(4000, len(sample_idx)), replace=False)
        _, _, vt = np.linalg.svd(tfidf[fit_idx], full_matrices=False)
        projection = vt[:k].T
        emb = tfidf @ projection
    else:
        emb = tfidf[:, :k]

    if emb.shape[1] < embedding_dim:
        pad = np.zeros((n_docs, embedding_dim - emb.shape[1]), dtype=np.float32)
        emb = np.hstack([emb, pad])

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = np.where(norms > 1e-8, emb / np.maximum(norms, 1e-8), 0.0)
    return emb_norm.astype(np.float32), vocab


def _kmeans_cosine(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    seed: int = 42,
    max_iter: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic K-Means++ clustering on L2-normalized complaint embeddings."""
    rng = np.random.default_rng(seed)
    n_samples, dim = embeddings.shape
    if n_samples == 0:
        return np.array([], dtype=int), np.zeros((n_clusters, dim))

    centroids = np.zeros((n_clusters, dim), dtype=np.float32)
    first_idx = int(rng.integers(0, n_samples))
    centroids[0] = embeddings[first_idx]

    for c in range(1, n_clusters):
        dists = np.min(
            [np.sum((embeddings - centroids[j]) ** 2, axis=1) for j in range(c)],
            axis=0,
        )
        probs = dists / max(float(dists.sum()), 1e-8)
        chosen = int(rng.choice(n_samples, p=probs))
        centroids[c] = embeddings[chosen]

    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        dists_matrix = np.stack(
            [np.sum((embeddings - centroids[c]) ** 2, axis=1) for c in range(n_clusters)],
            axis=1,
        )
        new_labels = np.argmin(dists_matrix, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(n_clusters):
            members = embeddings[labels == c]
            if len(members) > 0:
                mean_vec = members.mean(axis=0)
                norm = np.linalg.norm(mean_vec)
                centroids[c] = mean_vec / max(float(norm), 1e-8)

    return labels, centroids


def run_complaint_intelligence(
    features_df: pd.DataFrame,
    config: Dict[str, Any] | None = None,
    save_to_disk: bool = True,
) -> ComplaintAnalysisResult:
    """Embed pre-cutoff complaints, cluster into 5 distinct themes, and attach NLP features."""
    if config is None:
        config = load_config()

    nlp_cfg = config["models"]["nlp"]
    n_clusters = int(nlp_cfg.get("n_clusters", 5))
    seed = int(nlp_cfg.get("random_seed", 42))

    df = features_df.copy()
    texts = df["combined_complaint_text"].fillna("").astype(str).tolist()
    has_complaint_mask = np.array([len(t.strip()) > 0 for t in texts])

    # 1. Generate Semantic Embeddings
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            model = SentenceTransformer(nlp_cfg.get("model_name", "all-MiniLM-L6-v2"))
            raw_emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            embeddings = np.asarray(raw_emb, dtype=np.float32)
            backend = f"sentence-transformers ({nlp_cfg.get('model_name')})"
        except Exception as exc:
            logger.warning(f"SentenceTransformer load fallback triggered ({exc}); using deterministic TF-IDF/SVD.")
            embeddings, _ = _compute_tfidf_svd_embeddings(texts, embedding_dim=16, seed=seed)
            backend = "tfidf_svd_semantic_embeddings"
    else:
        embeddings, _ = _compute_tfidf_svd_embeddings(texts, embedding_dim=16, seed=seed)
        backend = "tfidf_svd_semantic_embeddings"

    # 2. Cluster Non-Empty Complaints into `n_clusters` Themes
    complaint_indices = np.where(has_complaint_mask)[0]
    complaint_embs = embeddings[complaint_indices]
    cluster_labels_sub, centroids = _kmeans_cosine(
        complaint_embs, n_clusters=n_clusters, seed=seed
    )

    full_cluster_ids = np.full(len(df), -1, dtype=int)
    full_cluster_ids[complaint_indices] = cluster_labels_sub
    df["complaint_cluster_id"] = full_cluster_ids

    # Map each discovered cluster 1-to-1 to the 5 canonical themes via Hungarian algorithm on keyword overlap
    theme_codes = list(THEME_BUSINESS_LABELS.keys())
    score_matrix = np.zeros((n_clusters, len(theme_codes)), dtype=float)
    cluster_top_keywords: Dict[int, List[str]] = {}

    for c_id in range(n_clusters):
        c_df = df[full_cluster_ids == c_id]
        word_freq: Dict[str, int] = {}
        for txt in c_df["combined_complaint_text"]:
            for tok in _tokenize(str(txt)):
                word_freq[tok] = word_freq.get(tok, 0) + 1
        top_kw = [w for w, _ in sorted(word_freq.items(), key=lambda kv: -kv[1])[:8]]
        cluster_top_keywords[c_id] = top_kw
        for t_idx, t_code in enumerate(theme_codes):
            anchors = THEME_ANCHOR_KEYWORDS[t_code]
            overlap_score = sum(word_freq.get(a, 0) for a in anchors) / max(len(c_df), 1)
            score_matrix[c_id, t_idx] = overlap_score

    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    cluster_id_to_theme_code: Dict[int, str] = {-1: "no_complaint"}
    cluster_id_to_theme_label: Dict[int, str] = {-1: "No Pre-Cutoff Complaints"}
    for r, c in zip(row_ind, col_ind):
        t_code = theme_codes[int(c)]
        cluster_id_to_theme_code[int(r)] = t_code
        cluster_id_to_theme_label[int(r)] = THEME_BUSINESS_LABELS[t_code]

    cluster_rows = []
    for c_id in range(n_clusters):
        c_mask = full_cluster_ids == c_id
        c_df = df[c_mask]
        if len(c_df) == 0:
            continue
        t_code = cluster_id_to_theme_code.get(c_id, f"theme_{c_id}")
        b_label = cluster_id_to_theme_label.get(c_id, f"Cluster {c_id}")
        cluster_rows.append({
            "cluster_id": int(c_id),
            "theme_code": t_code,
            "theme_label": b_label,
            "customer_count": int(len(c_df)),
            "pct_of_complaining_customers": round(float(len(c_df) / max(len(complaint_indices), 1)), 4),
            "avg_ticket_count": round(float(c_df["ticket_count_90d"].mean()), 2),
            "avg_max_severity": round(float(c_df["ticket_max_severity_90d"].mean()), 2),
            "avg_resolution_hours": round(float(c_df["ticket_mean_resolution_hours_90d"].mean()), 2),
            "unresolved_rate": round(float((c_df["ticket_unresolved_count_90d"] > 0).mean()), 4),
            "observed_churn_rate": round(float(c_df["churned"].mean()), 4),
            "top_keywords": ", ".join(cluster_top_keywords.get(c_id, [])[:7]),
        })

    no_comp_df = df[full_cluster_ids == -1]
    cluster_rows.append({
        "cluster_id": -1,
        "theme_code": "no_complaint",
        "theme_label": "No Pre-Cutoff Complaints",
        "customer_count": int(len(no_comp_df)),
        "pct_of_complaining_customers": 0.0,
        "avg_ticket_count": 0.0,
        "avg_max_severity": 0.0,
        "avg_resolution_hours": 0.0,
        "unresolved_rate": 0.0,
        "observed_churn_rate": round(float(no_comp_df["churned"].mean()), 4) if len(no_comp_df) else 0.0,
        "top_keywords": "none",
    })

    cluster_summary_df = pd.DataFrame(cluster_rows).sort_values(
        by="observed_churn_rate", ascending=False
    ).reset_index(drop=True)

    df["complaint_theme_code"] = [cluster_id_to_theme_code.get(int(cid), "no_complaint") for cid in full_cluster_ids]
    df["complaint_theme_label"] = [cluster_id_to_theme_label.get(int(cid), "No Pre-Cutoff Complaints") for cid in full_cluster_ids]

    # 3. Create Numeric NLP Features for Churn & Uplift Models
    nlp_feature_columns: List[str] = []
    n_emb_feats = min(6, embeddings.shape[1])
    for i in range(n_emb_feats):
        col_name = f"nlp_emb_{i}"
        df[col_name] = np.round(embeddings[:, i], 5)
        nlp_feature_columns.append(col_name)

    for c_id in range(n_clusters):
        col_name = f"nlp_cluster_{c_id}_sim"
        sim = np.clip(embeddings @ centroids[c_id], -1.0, 1.0)
        sim = np.where(has_complaint_mask, sim, 0.0)
        df[col_name] = np.round(sim, 5)
        nlp_feature_columns.append(col_name)

    for theme_key in THEME_BUSINESS_LABELS:
        col_name = f"nlp_theme_{theme_key}"
        df[col_name] = (df["complaint_theme_code"] == theme_key).astype(int)
        nlp_feature_columns.append(col_name)

    if save_to_disk:
        processed_dir = get_project_root() / config["etl"]["processed_dir"]
        processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed_dir / "customer_features_with_nlp.parquet", index=False)
        cluster_summary_df.to_parquet(processed_dir / "complaint_cluster_summary.parquet", index=False)

        reports_dir = get_project_root() / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "nlp_complaint_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_backend": backend,
                    "n_clusters": n_clusters,
                    "nlp_feature_columns": nlp_feature_columns,
                    "clusters": cluster_summary_df.to_dict(orient="records"),
                },
                f,
                indent=2,
            )
        logger.info(
            f"NLP Complaint Intelligence complete ({backend}): clustered {len(complaint_indices):,} "
            f"complaining customers into {n_clusters} distinct themes."
        )

    return ComplaintAnalysisResult(
        enriched_features_df=df,
        cluster_summary_df=cluster_summary_df,
        nlp_feature_columns=nlp_feature_columns,
        embedding_backend=backend,
    )
