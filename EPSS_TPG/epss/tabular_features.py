"""
Tabular Feature Extractor — Encode structured NVD + EPSS + ExploitDB fields
=============================================================================
Extracts and encodes non-text features from CVE records:

    NVD features (53 dims):
        - CVSS v3 base score (continuous, normalized 0-1)
        - Has-CVSS indicator (binary)
        - CVSS v3 vector components (8 categorical → 22 one-hot)
        - Top-K CWE IDs (multi-hot + "other") = 26 dims
        - Number of CWE IDs (count, normalized)
        - Number of references (log-normalized)
        - Vulnerability age (log-normalized days since publication)

    EPSS features (2 dims):
        - EPSS score (probability 0-1, from bulk CSV)
        - EPSS percentile (rank 0-1, from bulk CSV)

    ExploitDB features (2 dims):
        - has_public_exploit (binary: is there a public PoC?)
        - num_exploits (log-normalized count of public exploits)

    TOTAL: 57 dimensions
"""

import logging
import math
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# CVSS v3.1 vector component definitions
CVSS_COMPONENTS = {
    "AV": ["N", "A", "L", "P"],      # Attack Vector
    "AC": ["L", "H"],                  # Attack Complexity
    "PR": ["N", "L", "H"],            # Privileges Required
    "UI": ["N", "R"],                  # User Interaction
    "S":  ["U", "C"],                  # Scope
    "C":  ["N", "L", "H"],            # Confidentiality Impact
    "I":  ["N", "L", "H"],            # Integrity Impact
    "A":  ["N", "L", "H"],            # Availability Impact
}

# Total one-hot dimension for CVSS vector: 4+2+3+2+2+3+3+3 = 22
CVSS_ONEHOT_DIM = sum(len(v) for v in CVSS_COMPONENTS.values())


class TabularFeatureExtractor:
    """Extracts and encodes tabular features from CVE records.

    Call `fit()` first to discover top-K CWEs from the dataset,
    then `encode()` for each CVE record.

    Args:
        top_k_cwes: Number of most frequent CWEs to one-hot encode.
        reference_date: Date for computing vulnerability age.
                       Defaults to 2025-01-01 (fixed for reproducibility).
    """

    def __init__(
        self,
        top_k_cwes: int = 25,
        reference_date: Optional[str] = None,
        include_epss_feature: bool = True,
    ):
        self.top_k_cwes = top_k_cwes
        self.ref_date = datetime.fromisoformat(
            reference_date or "2025-01-01T00:00:00"
        )
        self.include_epss_feature = include_epss_feature
        self.cwe_to_idx: Dict[str, int] = {}
        self._fitted = False

    def fit(self, labeled_cves: Dict[str, dict]) -> "TabularFeatureExtractor":
        """Discover top-K CWEs from the dataset.

        Args:
            labeled_cves: Full labeled CVE dataset (CVE-ID -> record).

        Returns:
            self (for chaining).
        """
        cwe_counter = Counter()
        for record in labeled_cves.values():
            for cwe in record.get("cwe_ids", []):
                cwe_counter[cwe] += 1

        top_cwes = [cwe for cwe, _ in cwe_counter.most_common(self.top_k_cwes)]
        self.cwe_to_idx = {cwe: i for i, cwe in enumerate(top_cwes)}

        logger.info(
            "Tabular features: top %d CWEs cover %.1f%% of all CWE assignments",
            len(top_cwes),
            100 * sum(cwe_counter[c] for c in top_cwes) / max(sum(cwe_counter.values()), 1),
        )
        logger.info("Total tabular feature dim: %d", self.feature_dim)

        self._fitted = True
        return self

    @property
    def feature_dim(self) -> int:
        """Total dimension of the encoded feature vector.

        With include_epss_feature=True  (default): 57 dims
        With include_epss_feature=False (no-leakage): 55 dims
        """
        base = (
            1                           # cvss3_score (normalized)
            + 1                         # has_cvss (binary)
            + CVSS_ONEHOT_DIM           # CVSS vector components (22)
            + (self.top_k_cwes + 1)     # CWE multi-hot + "other" (26)
            + 1                         # num_cwes (count)
            + 1                         # num_references (log-normalized)
            + 1                         # vulnerability_age (log-normalized days)
            + 1                         # has_public_exploit (binary)
            + 1                         # num_exploits (log-normalized)
        )
        if self.include_epss_feature:
            base += 2                   # epss_score + epss_percentile
        return base
        # include_epss_feature=True  → 1+1+22+26+1+1+1+1+1+2 = 57
        # include_epss_feature=False → 1+1+22+26+1+1+1+1+1   = 55

    def encode(self, record: dict) -> np.ndarray:
        """Encode a single CVE record into a tabular feature vector.

        Handles two dataset formats transparently (same 57-dim output):

        Format A — NVD pipeline (labeled_cves.json from DataCollector):
            cvss3_score, cvss3_vector, cwe_ids, references,
            published, epss_score, epss_percentile,
            has_public_exploit, num_exploits

        Format B — Sec4AI4Aec CSV (via csv_adapter.py):
            cvss3_score  (mapped from cvss_score)
            cvss3_vector (reconstructed from CVSS component columns)
            cwe_ids = []          → zero-filled (not in dataset)
            references = []       → falls back to social_source_count
            published  (mapped from social media date)
            epss_score            ✓
            epss_percentile = 0   → zero (not in dataset)
            has_public_exploit    (mapped from code_available)
            num_exploits          (mapped from social_source_count as proxy)

        Args:
            record: CVE record dict from labeled_cves.json.

        Returns:
            numpy array of shape (feature_dim,) = (57,).
        """
        features = []

        # 1. CVSS v3 base score (normalized 0-10 → 0-1)
        #    Format A: "cvss3_score" | Format B: also "cvss3_score" (adapter maps it)
        cvss_score = record.get("cvss3_score") or record.get("cvss_score")
        if cvss_score is not None:
            features.append(float(cvss_score) / 10.0)
            features.append(1.0)  # has_cvss = True
        else:
            features.append(0.0)
            features.append(0.0)  # has_cvss = False

        # 2. CVSS v3 vector components (one-hot per component)
        cvss_vector = record.get("cvss3_vector", "")
        cvss_onehot = self._encode_cvss_vector(cvss_vector)
        features.extend(cvss_onehot)

        # 3. CWE IDs (multi-hot for top-K + "other" bucket)
        #    Format B: always empty → all zeros, which is handled gracefully
        cwe_ids = record.get("cwe_ids", [])
        cwe_encoded = self._encode_cwes(cwe_ids)
        features.extend(cwe_encoded)

        # 4. Number of CWE IDs (normalized, cap at 10)
        features.append(min(len(cwe_ids), 10) / 10.0)

        # 5. Signal richness: NVD references count OR social media source count
        #    Format A: number of NVD reference links
        #    Format B: social_source_count (platforms mentioning this CVE)
        num_refs = len(record.get("references", []))
        if num_refs == 0:
            # Fall back to social source count for Format B records
            num_refs = int(record.get("social_source_count", 0))
        features.append(math.log1p(num_refs) / math.log1p(20))

        # 6. Vulnerability age (log-normalized days since publication)
        published = record.get("published", "")
        age_days = self._compute_age_days(published)
        features.append(math.log1p(age_days) / math.log1p(3650))  # ~10 years

        # 7 & 8. EPSS score and percentile (only if include_epss_feature=True)
        # WARNING: Including EPSS as a feature when EPSS is also the training label
        # creates data leakage — the model learns "predict EPSS from EPSS" rather
        # than learning genuine exploitation signals from CVE characteristics.
        # Set include_epss_feature=False and retrain for a leakage-free model.
        if self.include_epss_feature:
            features.append(float(record.get("epss_score", 0.0)))
            features.append(float(record.get("epss_percentile", 0.0)))

        # 9. Has public exploit / PoC code
        #    Format A: ExploitDB flag | Format B: code_available flag
        has_exploit = (
            record.get("has_public_exploit", False)
            or record.get("code_available", False)
        )
        features.append(1.0 if has_exploit else 0.0)

        # 10. Exploit/mention count (log-normalized, cap at 20)
        #     Format A: ExploitDB num_exploits
        #     Format B: social_source_count as a proxy for exploitation interest
        num_exp = int(record.get("num_exploits", 0))
        if num_exp == 0:
            num_exp = int(record.get("social_source_count", 0))
        features.append(math.log1p(num_exp) / math.log1p(20))

        return np.array(features, dtype=np.float32)

    def _encode_cvss_vector(self, vector_str: str) -> List[float]:
        """Parse CVSS:3.1/AV:N/AC:L/... into one-hot encoding."""
        onehot = [0.0] * CVSS_ONEHOT_DIM

        if not vector_str:
            return onehot

        # Parse "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:N"
        parts = {}
        for segment in vector_str.split("/"):
            if ":" in segment:
                key, val = segment.split(":", 1)
                parts[key] = val

        offset = 0
        for component, values in CVSS_COMPONENTS.items():
            val = parts.get(component, "")
            if val in values:
                idx = values.index(val)
                onehot[offset + idx] = 1.0
            offset += len(values)

        return onehot

    def _encode_cwes(self, cwe_ids: List[str]) -> List[float]:
        """Multi-hot encode CWE IDs with top-K + other bucket."""
        encoded = [0.0] * (self.top_k_cwes + 1)  # +1 for "other"

        for cwe in cwe_ids:
            if cwe in self.cwe_to_idx:
                encoded[self.cwe_to_idx[cwe]] = 1.0
            else:
                encoded[-1] = 1.0  # "other" bucket

        return encoded

    def _compute_age_days(self, published_str: str) -> float:
        """Compute days between publication date and reference date."""
        if not published_str:
            return 0.0
        try:
            pub_date = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            # Make ref_date timezone-aware if pub_date is
            ref = self.ref_date
            if pub_date.tzinfo is not None and ref.tzinfo is None:
                ref = ref.replace(tzinfo=pub_date.tzinfo)
            delta = ref - pub_date
            return max(0.0, delta.total_seconds() / 86400)
        except (ValueError, TypeError):
            return 0.0

    def get_feature_names(self) -> List[str]:
        """Return human-readable names for each feature dimension."""
        names = ["cvss3_score", "has_cvss"]

        for component, values in CVSS_COMPONENTS.items():
            for val in values:
                names.append(f"cvss_{component}_{val}")

        for cwe in sorted(self.cwe_to_idx.keys(), key=lambda c: self.cwe_to_idx[c]):
            names.append(f"cwe_{cwe}")
        names.append("cwe_other")

        names.extend(["num_cwes", "num_references", "vulnerability_age_days"])
        if self.include_epss_feature:
            names.extend(["epss_score", "epss_percentile"])
        names.extend(["has_public_exploit", "num_exploits"])
        return names
