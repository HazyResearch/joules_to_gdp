"""Recover versioned, public AWS prices for one fully specified reference node.

Stores the exact matching CSV rows and source version metadata, not current
marketplace quotes, reserved-capacity placeholders, or third-party estimates.
"""
from __future__ import annotations
import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/glm52_reference_rental_20260919/sources"
BASE = "https://pricing.us-east-1.amazonaws.com"


def get_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def fetch_version(item):
    date, version = item
    path = OUT / f"aws_{date}.json"
    if path.exists():
        return json.loads(path.read_text())
    region_url = BASE + version["offerVersionUrl"].replace("/index.json", "/region_index.json")
    regions = get_json(region_url)
    source_url = BASE + regions["regions"]["us-east-1"]["currentVersionUrl"].replace(".json", ".csv")
    records = []
    with urllib.request.urlopen(source_url, timeout=60) as response:
        stream = io.TextIOWrapper(response, encoding="utf-8")
        metadata = [next(stream).strip() for _ in range(5)]
        for row in csv.DictReader(stream):
            # Capacity-block and reservation zero-price records are NOT on-demand rental rates.
            compute = (row.get("Instance Type") == "p5en.48xlarge"
                       and row.get("Operating System") == "Linux"
                       and row.get("Tenancy") == "Shared"
                       and row.get("Pre Installed S/W") == "NA"
                       and row.get("MarketOption") == "OnDemand"
                       and row.get("CapacityStatus") == "Used"
                       and row.get("operation") == "RunInstances")
            root_disk = (row.get("usageType") == "EBS:VolumeUsage.gp3"
                         and row.get("Volume API Name") == "gp3")
            if row.get("TermType") == "OnDemand" and (compute or root_disk):
                records.append(row)
    data = {"price_period": date, "version": version,
            "source_url": source_url, "source_metadata": metadata, "rows": records,
            "retrieval_date": "2026-09-19",
            "note": "Exact matching rows from AWS versioned CSV; full regional CSV streamed, not retained"}
    payload = json.dumps(records, sort_keys=True).encode()
    data["selected_rows_sha256"] = hashlib.sha256(payload).hexdigest()
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "aws_version_index.json"
    if not index_path.exists():
        index_path.write_text(json.dumps(get_json(BASE + "/offers/v1.0/aws/AmazonEC2/index.json"), indent=2))
    index = json.loads(index_path.read_text())
    items = []
    for date in ["2024-12", "2025-03", "2025-05", "2025-06", "2025-09",
                 "2025-12", "2026-03", "2026-06", "2026-07", "2026-09"]:
        target = date + "-15T12:00:00Z"
        selected = [v for v in index["versions"].values()
                    if v["versionEffectiveBeginDate"] <= target < (v.get("versionEffectiveEndDate") or "9999")]
        if len(selected) != 1:
            raise ValueError((date, selected))
        items.append((date, selected[0]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch_version, items))
    for item in results:
        print(item["price_period"], [(r["usageType"], r["PricePerUnit"], r["Unit"], r["EffectiveDate"])
                                     for r in item["rows"]], flush=True)


if __name__ == "__main__":
    main()
