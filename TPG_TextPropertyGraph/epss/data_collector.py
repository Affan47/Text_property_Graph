"""
Data Collector — Fetch CVE descriptions, CISA KEV labels, and EPSS scores
==========================================================================
Three data sources:
    1. NVD (National Vulnerability Database) — CVE descriptions via REST API 2.0
    2. CISA KEV (Known Exploited Vulnerabilities) — binary exploitation labels
    3. FIRST EPSS — daily exploitation probability scores (soft labels)

Usage:
    collector = DataCollector(output_dir="data/epss")
    collector.fetch_all(start_year=2017, end_year=2024)
"""

import json
import csv
import gzip
import time
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from io import StringIO

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects CVE descriptions, CISA KEV labels, and EPSS scores."""

    NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    EPSS_URL = "https://epss.cyentia.com/epss_scores-{date}.csv.gz"
    EPSS_CURRENT_URL = "https://api.first.org/data/v1/epss"

    def __init__(self, output_dir: str = "data/epss", nvd_api_key: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nvd_api_key = nvd_api_key or os.environ.get("NVD_API_KEY")
        self._session = requests.Session()
        if self.nvd_api_key:
            self._session.headers["apiKey"] = self.nvd_api_key

    # ─── CISA KEV ─────────────────────────────────────────────────────

    def fetch_kev(self) -> Dict[str, dict]:
        """Fetch CISA Known Exploited Vulnerabilities catalog.

        Returns:
            dict mapping CVE-ID -> {dateAdded, dueDate, product, vendor, ...}
        """
        cache = self.output_dir / "cisa_kev.json"
        if cache.exists():
            age_hours = (time.time() - cache.stat().st_mtime) / 3600
            if age_hours < 24:
                logger.info("Using cached CISA KEV (%.1f hours old)", age_hours)
                with open(cache) as f:
                    return json.load(f)

        logger.info("Fetching CISA KEV catalog...")
        resp = self._session.get(self.KEV_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        kev = {}
        for vuln in data.get("vulnerabilities", []):
            cve_id = vuln.get("cveID", "")
            if cve_id:
                kev[cve_id] = {
                    "dateAdded": vuln.get("dateAdded", ""),
                    "dueDate": vuln.get("dueDate", ""),
                    "vendor": vuln.get("vendorProject", ""),
                    "product": vuln.get("product", ""),
                    "shortDescription": vuln.get("shortDescription", ""),
                    "knownRansomwareCampaignUse": vuln.get("knownRansomwareCampaignUse", ""),
                }
        with open(cache, "w") as f:
            json.dump(kev, f, indent=2)

        logger.info("Fetched %d KEV entries", len(kev))
        return kev

    # ─── EPSS Scores ──────────────────────────────────────────────────

    def fetch_epss_current(self, cve_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Fetch current EPSS scores from FIRST API.

        Args:
            cve_ids: specific CVEs to query (batched in groups of 100).
                     If None, fetches all scores.

        Returns:
            dict mapping CVE-ID -> EPSS probability (0.0 to 1.0)
        """
        cache = self.output_dir / "epss_scores.json"
        if cache.exists() and cve_ids is None:
            age_hours = (time.time() - cache.stat().st_mtime) / 3600
            if age_hours < 24:
                logger.info("Using cached EPSS scores (%.1f hours old)", age_hours)
                with open(cache) as f:
                    return json.load(f)

        scores = {}

        if cve_ids is None:
            # Fetch all scores via paginated API
            logger.info("Fetching all EPSS scores (paginated)...")
            offset = 0
            limit = 1000
            while True:
                resp = self._session.get(
                    self.EPSS_CURRENT_URL,
                    params={"envelope": "true", "limit": limit, "offset": offset},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                items = data.get("data", [])
                if not items:
                    break
                for item in items:
                    scores[item["cve"]] = float(item["epss"])
                offset += limit
                total = data.get("total", 0)
                logger.info("  fetched %d / %d", offset, total)
                if offset >= total:
                    break
                time.sleep(0.5)
        else:
            # Batch query specific CVEs
            batch_size = 100
            n_batches = (len(cve_ids) + batch_size - 1) // batch_size
            logger.info("Fetching EPSS scores for %d CVEs (%d batches)...",
                        len(cve_ids), n_batches)
            for i in tqdm(range(0, len(cve_ids), batch_size),
                          total=n_batches, desc="EPSS scores", unit="batch"):
                batch = cve_ids[i : i + batch_size]
                cve_param = ",".join(batch)
                try:
                    resp = self._session.get(
                        self.EPSS_CURRENT_URL,
                        params={"cve": cve_param},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for item in data.get("data", []):
                        scores[item["cve"]] = float(item["epss"])
                except Exception as e:
                    logger.warning("EPSS batch %d failed: %s", i // batch_size, e)
                time.sleep(0.3)

        with open(cache, "w") as f:
            json.dump(scores, f)
        logger.info("Fetched EPSS scores for %d CVEs", len(scores))
        return scores

    # ─── NVD CVE Descriptions ─────────────────────────────────────────

    def fetch_nvd_cves(
        self,
        start_year: int = 2017,
        end_year: int = 2024,
        cve_ids: Optional[List[str]] = None,
    ) -> Dict[str, dict]:
        """Fetch CVE records from NVD API 2.0.

        If cve_ids is provided, fetches only those CVEs.
        Otherwise, fetches all CVEs published in [start_year, end_year].

        Returns:
            dict mapping CVE-ID -> {description, published, cvss3_score, cvss3_vector,
                                     cwe_ids, references}
        """
        cache = self.output_dir / "nvd_cves.json"
        if cache.exists():
            with open(cache) as f:
                existing = json.load(f)
            logger.info("Loaded %d cached NVD CVE records", len(existing))
        else:
            existing = {}

        if cve_ids is not None:
            missing = [c for c in cve_ids if c not in existing]
            if not missing:
                return {c: existing[c] for c in cve_ids if c in existing}
            logger.info("Fetching %d specific CVEs from NVD...", len(missing))
            for cve_id in tqdm(missing, desc="NVD CVEs", unit="cve"):
                record = self._fetch_single_cve(cve_id)
                if record:
                    existing[cve_id] = record
                time.sleep(0.6 if self.nvd_api_key else 6.0)
        else:
            for year in range(start_year, end_year + 1):
                year_count = sum(1 for v in existing.values()
                                 if v.get("published", "").startswith(str(year)))
                if year_count > 100:
                    logger.info("Year %d: %d CVEs already cached, skipping", year, year_count)
                    continue
                self._fetch_year(year, existing)

        with open(cache, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info("Total NVD CVE records: %d", len(existing))
        return existing

    def _fetch_single_cve(self, cve_id: str) -> Optional[dict]:
        """Fetch a single CVE from NVD API."""
        try:
            resp = self._session.get(
                self.NVD_API,
                params={"cveId": cve_id},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            vulns = data.get("vulnerabilities", [])
            if vulns:
                return self._parse_nvd_record(vulns[0].get("cve", {}))
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", cve_id, e)
        return None

    def _fetch_year(self, year: int, results: dict, max_retries: int = 3):
        """Fetch all CVEs for a given year from NVD (paginated).

        NVD API 2.0 limits date ranges to 120 days, so each year is
        split into quarterly windows.
        """
        # Quarterly date ranges (each ≤ 120 days)
        quarters = [
            (f"{year}-01-01T00:00:00", f"{year}-03-31T23:59:59"),
            (f"{year}-04-01T00:00:00", f"{year}-06-30T23:59:59"),
            (f"{year}-07-01T00:00:00", f"{year}-09-30T23:59:59"),
            (f"{year}-10-01T00:00:00", f"{year}-12-31T23:59:59"),
        ]
        results_per_page = 2000
        delay = 0.6 if self.nvd_api_key else 6.0

        logger.info("Fetching NVD CVEs for year %d (4 quarters)...", year)

        for q_idx, (start_date, end_date) in enumerate(quarters, 1):
            start_idx = 0
            retries = 0

            while True:
                params = {
                    "pubStartDate": start_date,
                    "pubEndDate": end_date,
                    "startIndex": start_idx,
                    "resultsPerPage": results_per_page,
                }
                try:
                    resp = self._session.get(self.NVD_API, params=params, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    retries = 0  # reset on success
                except Exception as e:
                    retries += 1
                    logger.error("NVD API error for year %d Q%d offset %d (attempt %d/%d): %s",
                                 year, q_idx, start_idx, retries, max_retries, e)
                    if retries >= max_retries:
                        logger.error("Max retries reached for year %d Q%d, skipping", year, q_idx)
                        break
                    time.sleep(30)
                    continue

                vulns = data.get("vulnerabilities", [])
                total = data.get("totalResults", 0)

                for item in vulns:
                    cve_data = item.get("cve", {})
                    cve_id = cve_data.get("id", "")
                    if cve_id:
                        results[cve_id] = self._parse_nvd_record(cve_data)

                start_idx += results_per_page
                logger.info("  Year %d Q%d: %d / %d", year, q_idx, min(start_idx, total), total)

                if start_idx >= total:
                    break
                time.sleep(delay)

    def _parse_nvd_record(self, cve: dict) -> dict:
        """Parse a single NVD CVE JSON record into a flat dict."""
        # Description (English preferred)
        description = ""
        for desc in cve.get("descriptions", []):
            if desc.get("lang") == "en":
                description = desc.get("value", "")
                break
        if not description:
            descs = cve.get("descriptions", [])
            if descs:
                description = descs[0].get("value", "")

        # CVSS v3.x
        cvss3_score = None
        cvss3_vector = ""
        metrics = cve.get("metrics", {})
        for key in ["cvssMetricV31", "cvssMetricV30"]:
            metric_list = metrics.get(key, [])
            if metric_list:
                cvss_data = metric_list[0].get("cvssData", {})
                cvss3_score = cvss_data.get("baseScore")
                cvss3_vector = cvss_data.get("vectorString", "")
                break

        # CWE IDs
        cwe_ids = []
        for weakness in cve.get("weaknesses", []):
            for desc in weakness.get("description", []):
                val = desc.get("value", "")
                if val.startswith("CWE-"):
                    cwe_ids.append(val)

        # References
        refs = [r.get("url", "") for r in cve.get("references", [])[:10]]

        return {
            "description": description,
            "published": cve.get("published", ""),
            "lastModified": cve.get("lastModified", ""),
            "cvss3_score": cvss3_score,
            "cvss3_vector": cvss3_vector,
            "cwe_ids": cwe_ids,
            "references": refs,
        }

    # ─── Label Assembly ───────────────────────────────────────────────

    def build_labels(
        self,
        cve_records: Dict[str, dict],
        kev: Dict[str, dict],
        epss_scores: Dict[str, float],
        epss_threshold: float = 0.5,
    ) -> Dict[str, dict]:
        """Build unified label set for each CVE.

        Label strategy:
            - binary_label = 1 if CVE is in CISA KEV (confirmed exploited)
            - binary_label = 0 otherwise
            - epss_score = EPSS probability (soft label, 0.0–1.0)
            - high_epss = 1 if epss_score >= threshold

        Returns:
            dict mapping CVE-ID -> {binary_label, epss_score, high_epss,
                                     in_kev, description, ...}
        """
        labeled = {}
        kev_count = 0
        epss_count = 0

        for cve_id, record in cve_records.items():
            desc = record.get("description", "")
            if not desc or desc.startswith("** REJECT") or desc.startswith("Rejected reason:"):
                continue

            in_kev = cve_id in kev
            epss = epss_scores.get(cve_id, 0.0)

            labeled[cve_id] = {
                "cve_id": cve_id,
                "description": desc,
                "published": record.get("published", ""),
                "cvss3_score": record.get("cvss3_score"),
                "cvss3_vector": record.get("cvss3_vector", ""),
                "cwe_ids": record.get("cwe_ids", []),
                # Labels
                "binary_label": 1 if in_kev else 0,
                "epss_score": epss,
                "high_epss": 1 if epss >= epss_threshold else 0,
                "in_kev": in_kev,
            }
            if in_kev:
                kev_count += 1
            if epss > 0:
                epss_count += 1

        logger.info(
            "Built labels: %d CVEs, %d in KEV (%.1f%%), %d with EPSS scores",
            len(labeled), kev_count,
            100 * kev_count / max(len(labeled), 1),
            epss_count,
        )

        # Save labeled dataset
        outfile = self.output_dir / "labeled_cves.json"
        with open(outfile, "w") as f:
            json.dump(labeled, f, indent=2)
        logger.info("Saved labeled dataset to %s", outfile)

        return labeled

    # ─── High-Level Orchestration ─────────────────────────────────────

    def fetch_all(
        self,
        start_year: int = 2017,
        end_year: int = 2024,
        epss_threshold: float = 0.5,
    ) -> Dict[str, dict]:
        """Fetch all data sources and build unified labeled dataset.

        Returns:
            labeled dataset dict (CVE-ID -> record with labels)
        """
        kev = self.fetch_kev()
        cves = self.fetch_nvd_cves(start_year=start_year, end_year=end_year)
        epss = self.fetch_epss_current(cve_ids=list(cves.keys()))
        return self.build_labels(cves, kev, epss, epss_threshold=epss_threshold)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collector = DataCollector()
    labeled = collector.fetch_all(start_year=2020, end_year=2024)
    print(f"\nDataset: {len(labeled)} CVEs")
    exploited = sum(1 for v in labeled.values() if v["binary_label"] == 1)
    print(f"Exploited (KEV): {exploited} ({100*exploited/len(labeled):.1f}%)")
