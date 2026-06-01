"""Bexar County BCAD parcels — paginated full-county pull.

Source: Bexar County GIS / BCAD — https://maps.bexar.org/arcgis/rest/services/Parcels/MapServer/0
License: not explicitly stated on the dataset page; operated under Bexar County
GIS / BCAD; attribution cited in DATA_INVENTORY catalog entry.

**Two-phase design** so the long-running pull is resumable + completeness-verifiable:

  Phase A — fetch (this script when invoked as `--fetch`):
    Paginated pull, 1000 records/page, ~600 pages county-wide. Each page
    written to a separate file in PAGES_DIR. Per-page log records the attempt
    + outcome. Resumable: if PAGES_DIR/page_<offset>.geojson already exists
    with a non-empty payload, the page is skipped.

  Phase B — verify+classify+rasterize (when invoked as `--finish`):
    Scans the fetch log for gaps or retry-exhausted pages, re-fetches any
    missing ones, then classifies (per the locked rules in
    docs/research/ownership/PHASE_0_INVESTIGATION.md), reprojects to
    EPSG:5070, and rasterizes is_public + is_vacant onto the SA grid.

Run as:
  python scripts/data/download_bexar_parcels.py --fetch
  python scripts/data/download_bexar_parcels.py --finish

Output (after --finish):
  data/sa/sa_ownership.gpkg            — polygons in EPSG:5070 with
                                         owner_class, is_public, is_vacant
  data/sa/sa_public_vacant_30m.tif     — int8 raster on the SA grid:
                                         0=neither, 1=public, 2=vacant,
                                         3=both, -1=outside
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

QUERY_URL = "https://maps.bexar.org/arcgis/rest/services/Parcels/MapServer/0/query"
PAGE_SIZE = 1000          # maxRecordCount per service metadata
MAX_OFFSET = 1_200_000    # safety bound — way past the ~600K county expected
MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 30
TARGET_CRS = "EPSG:5070"

# Pages persisted between phases. Outside the repo by design — they're transient
# bulk data; only the final classified .gpkg lands in the repo.
PAGES_DIR = Path("/tmp/bexar_pages")
FETCH_LOG = PAGES_DIR / "fetch.log"

REPO_ROOT = Path(__file__).resolve().parents[2]
POLY_OUT   = REPO_ROOT / "data" / "sa" / "sa_ownership.gpkg"
RASTER_OUT = REPO_ROOT / "data" / "sa" / "sa_public_vacant_30m.tif"
# Finer ownership classes (OWNERSHIP_FINER_CLASSES_SPEC.md) — two-band raster.
# Band 1: ownership class enum (0-5); Band 2: is_vacant (0/1). nodata=-1.
RASTER_OUT_2BAND = REPO_ROOT / "data" / "sa" / "sa_ownership_2band_30m.tif"
REF_RASTER = REPO_ROOT / "data" / "sa" / "flood" / "land_use_compound_sa.tif"

# Locked classifier (docs/research/ownership/PHASE_0_INVESTIGATION.md).
PUBLIC_GOVERNMENT_CLASSES = {"city", "county", "state", "federal", "isd", "river_auth"}

# Finer Ownership Classes (OWNERSHIP_FEASIBILITY_PROFILING.md → 99.9% public
# acreage classified cleanly). Class enum used in Band 1 of the new two-band
# raster. Order matters — match the rule precedence in _classify_six_way:
# school catches before state (Texas A&M Regents); HOA filter rescues
# "BEXAR COUNTY X HOMEOWNERS" → private before the county branch matches.
OWNERSHIP_CLASS_ENUM = {
    "private":           0,
    "city":              1,
    "county":            2,
    "state_federal":     3,
    "school_university": 4,
    "unknown":           5,
}

_HOA_RE = re.compile(r"\b(HOMEOWNERS|ASSOCIATION|HOA|LLC|LTD|INC|TRUST)\b")
_SCHOOL_RE = re.compile(
    r"\b(ISD|INDEPENDENT SCHOOL|SCHOOL DISTRICT|UNIVERSITY|COLLEGE|REGENTS"
    r"|BOARD OF TRUSTEES.*SCHOOL|BOARD OF TRUSTEES.*ISD)\b"
)
_CITY_RE = re.compile(
    r"\b(CITY OF|HOUSING AUTHORITY|PUBLIC SERVICE BOARD|CITY PUBLIC SERVICE"
    r"|WATER SYSTEM)\b"
)
_COUNTY_RE = re.compile(r"\bCOUNTY\b")
_STATE_FED_RE = re.compile(
    # State (TX) + federal (US) government — patterns refined after the
    # full-parcel spot-check surfaced false positives from business names.
    # DROPPED from the feasibility doc's regex:
    #   - standalone \bUSA\b → caught "BORALIS USA INC", "FORESTAR (USA)
    #     REAL ESTATE GROUP", "HOME DEPOT USA INC" (~1.5k ac of private
    #     companies with USA in their corporate name). Federal-gov owners
    #     instead use "UNITED STATES" or "U S GOVERNMENT" — both covered.
    #   - standalone \bFEDERAL\b → caught "SECURITY SERVICE FEDERAL CREDIT
    #     UNION" (~76 ac; credit unions aren't federal gov). The real
    #     federal patterns are caught by "UNITED STATES" / "US GOVERNMENT".
    r"\b(STATE OF TEXAS|TX DEPT|TEXAS DEPT|TEXAS PARKS|TEXAS A&M"
    r"|TEXAS HIGHWAY|UNITED STATES|U\.S\.|US GOVERNMENT|U S GOVERNMENT"
    r"|RIVER AUTHORITY)\b"
)


def _classify_six_way(owner: str) -> str:
    """Apply the locked OWNERSHIP_FEASIBILITY_PROFILING.md rules; return one
    of the six class keys from OWNERSHIP_CLASS_ENUM. Owner-field None or
    empty → 'unknown'."""
    if owner is None:
        return "unknown"
    o = str(owner).strip().upper()
    if not o:
        return "unknown"
    if _SCHOOL_RE.search(o):
        return "school_university"
    if _CITY_RE.search(o):
        return "city"
    if _COUNTY_RE.search(o):
        if _HOA_RE.search(o):
            return "private"
        return "county"
    if _STATE_FED_RE.search(o):
        return "state_federal"
    return "private"


def _log(offset: int, status: str, n: int, err: str = "") -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    with FETCH_LOG.open("a") as f:
        f.write(json.dumps({
            "offset": offset, "status": status, "n": n,
            "err": err, "t": time.time(),
        }) + "\n")


def _fetch_page(offset: int) -> bytes:
    params = {
        "where": "1=1",
        "outFields": "Owner,State_cd,ImprVal,Exempts,Acres",
        "returnGeometry": "true",
        "f": "geojson",
        "resultOffset": offset,
        "resultRecordCount": PAGE_SIZE,
        "outSR": "4326",
    }
    r = requests.get(QUERY_URL, params=params, timeout=120)
    r.raise_for_status()
    return r.content


def _page_path(offset: int) -> Path:
    return PAGES_DIR / f"page_{offset:08d}.geojson"


def _page_record_count(path: Path) -> int:
    """Count GeoJSON features without parsing the full geometry."""
    try:
        data = json.loads(path.read_bytes())
        return len(data.get("features", []))
    except Exception:
        return -1


def fetch_all() -> int:
    """Phase A — paginate until an empty page is returned. Resumable."""
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    offset = 0
    fetched_pages = 0
    skipped_pages = 0
    failed_pages = 0
    start_t = time.time()
    while offset < MAX_OFFSET:
        path = _page_path(offset)
        # Resume: skip if a non-empty page is already on disk.
        if path.exists() and path.stat().st_size > 0:
            n = _page_record_count(path)
            if n > 0:
                print(f"  offset={offset:>7} skip (already have {n} records)", flush=True)
                _log(offset, "skip", n)
                skipped_pages += 1
                # If we know it's a partial last page, stop here.
                if n < PAGE_SIZE:
                    print(f"  ...short page (n={n} < {PAGE_SIZE}); fetch complete.")
                    break
                offset += PAGE_SIZE
                continue
            elif n == 0:
                # Recorded as the terminal empty page on a prior run.
                print(f"  offset={offset:>7} terminal empty page (cached); done.")
                break

        # Try with retries.
        success = False
        last_err = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.time()
                content = _fetch_page(offset)
                # Quick parse check
                data = json.loads(content)
                feats = data.get("features", [])
                n = len(feats)
                path.write_bytes(content)
                dt = time.time() - t0
                print(f"  offset={offset:>7} got {n:>4} in {dt:4.1f}s "
                      f"(attempt {attempt})", flush=True)
                _log(offset, "ok", n)
                fetched_pages += 1
                success = True
                if n == 0:
                    print(f"  ...empty page; fetch complete.")
                    return fetched_pages + skipped_pages
                if n < PAGE_SIZE:
                    print(f"  ...short page (n={n} < {PAGE_SIZE}); fetch complete.")
                    return fetched_pages + skipped_pages
                break
            except Exception as e:
                last_err = repr(e)
                print(f"  offset={offset:>7} FAIL attempt {attempt}/{MAX_RETRIES}: "
                      f"{e!r}", flush=True)
                _log(offset, f"retry_{attempt}", 0, last_err)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SEC)

        if not success:
            print(f"  offset={offset:>7} RETRIES_EXHAUSTED", flush=True)
            _log(offset, "exhausted", 0, last_err)
            failed_pages += 1
            # Don't stop the run; keep going so we have all the other pages
            # and can re-fetch the failures in a verify pass.

        offset += PAGE_SIZE

    dt = time.time() - start_t
    print(f"\nFetch loop done in {dt/60:.1f} min. "
          f"fetched={fetched_pages} skipped={skipped_pages} "
          f"failed={failed_pages}")
    return fetched_pages + skipped_pages


def verify_completeness() -> tuple[list[int], bool]:
    """Phase B step 1 — scan the log for gaps / retry-exhausted offsets AND
    confirm the fetch reached BCAD's genuine end-of-data.

    Returns (needs_refetch, reached_eod).
    - needs_refetch: offsets to (re-)fetch — empty if pages 0..max_off are
      contiguous and intact.
    - reached_eod: True iff there's a confirmed terminal marker — either the
      highest on-disk page has n<PAGE_SIZE (terminal partial), OR the log has
      an entry beyond max_off with status='ok' and n<PAGE_SIZE (the terminal
      empty page; n=0 pages aren't written to disk, only logged).
      reached_eod=False means "we don't know if BCAD has more records past
      max_off" — caller should extend via fetch_all() until eod is reached.
    """
    print(f"\nVerifying completeness against {FETCH_LOG}...")
    if not FETCH_LOG.exists():
        print("  No fetch log — nothing to verify.")
        return [], False

    # Build a map of offset -> (status, n) from the log (last record wins).
    log_records: dict[int, tuple[str, int]] = {}
    for line in FETCH_LOG.read_text().splitlines():
        try:
            rec = json.loads(line)
            log_records[int(rec["offset"])] = (rec["status"], int(rec.get("n", 0)))
        except Exception:
            continue
    statuses = {off: st for off, (st, _) in log_records.items()}

    # Inventory disk
    on_disk = {}
    for p in sorted(PAGES_DIR.glob("page_*.geojson")):
        m = re.match(r"page_(\d+)\.geojson$", p.name)
        if not m:
            continue
        off = int(m.group(1))
        on_disk[off] = _page_record_count(p)

    # Find the highest offset for which we have a successful page.
    if not on_disk:
        print("  No pages on disk.")
        return [], False
    max_off = max(on_disk)

    # Identify gaps: every offset 0..max_off step PAGE_SIZE should be present
    # AND have n==PAGE_SIZE (except the last, which can be partial). Anything
    # missing, empty (with status != 'ok' on its last line meaning genuine
    # end-of-data), SHORT (truncated mid-write), or marked exhausted in the
    # log gets re-fetched. The SHORT check catches the SIGTERM-during-write
    # case the resume logic ("skip if on disk + non-empty") otherwise misses.
    expected = list(range(0, max_off + PAGE_SIZE, PAGE_SIZE))
    needs_refetch: list[int] = []
    for off in expected:
        if off not in on_disk:
            needs_refetch.append(off)
            print(f"  MISSING offset={off:>7}")
            continue
        n = on_disk[off]
        is_last = (off == max_off)
        if n == 0:
            # Empty page — only valid if it's the terminal one AND the log
            # confirms an 'ok' empty (rather than a malformed write).
            if is_last and statuses.get(off) == "ok":
                continue
            needs_refetch.append(off)
            print(f"  EMPTY   offset={off:>7} (status={statuses.get(off)})")
            continue
        # Truncation check: BCAD returns exactly PAGE_SIZE per page until the
        # genuine final page, so any non-terminal page with n<PAGE_SIZE is the
        # smoking gun for a SIGTERM-during-write. Re-fetch it.
        if n < PAGE_SIZE and not is_last:
            needs_refetch.append(off)
            print(f"  SHORT   offset={off:>7} (n={n}, expected {PAGE_SIZE}; "
                  f"likely truncated mid-write)")
            continue
        last_status = statuses.get(off, "?")
        if last_status not in ("ok", "skip"):
            needs_refetch.append(off)
            print(f"  BAD     offset={off:>7} status={last_status}")

    # Also surface any retry-exhausted offsets not in the expected sweep.
    for off, status in statuses.items():
        if status == "exhausted" and off not in needs_refetch:
            needs_refetch.append(off)
            print(f"  EXHAUSTED offset={off:>7}")

    # End-of-data check: did the fetch confirm BCAD has no more records past
    # max_off? Terminal marker is either (a) the on-disk highest page being
    # short (n<PAGE_SIZE = genuine partial final page), or (b) a log entry at
    # offset>max_off with status='ok' and n<PAGE_SIZE (the empty terminal —
    # n=0 pages aren't written to disk so they show up only in the log).
    last_n = on_disk[max_off]
    reached_eod = False
    eod_note = ""
    if last_n < PAGE_SIZE:
        reached_eod = True
        eod_note = f"last on-disk page (offset={max_off}, n={last_n}) is partial"
    else:
        for off in sorted(log_records):
            if off <= max_off:
                continue
            st, n = log_records[off]
            if st == "ok" and n < PAGE_SIZE:
                reached_eod = True
                eod_note = f"log shows terminal {'empty' if n == 0 else f'short n={n}'} at offset={off}"
                break

    if not reached_eod:
        print(f"  EARLY STOP — last on-disk page (offset={max_off}) is full "
              f"({last_n} records) and no terminal marker beyond it. "
              f"Fetch did not confirm end-of-data; must extend.")
    else:
        print(f"  EOD confirmed: {eod_note}")

    if not needs_refetch and reached_eod:
        total = sum(n for n in on_disk.values() if n > 0)
        print(f"  Completeness OK: {len(on_disk)} pages, {total:,} total records.")
    elif needs_refetch:
        print(f"  {len(needs_refetch)} offsets need (re-)fetching: "
              f"{needs_refetch[:10]}{'...' if len(needs_refetch) > 10 else ''}")

    return needs_refetch, reached_eod


def refetch(offsets: list[int]) -> int:
    """Phase B step 2 — re-fetch the offsets returned by verify_completeness."""
    if not offsets:
        return 0
    print(f"\nRe-fetching {len(offsets)} offsets...")
    recovered = 0
    for off in offsets:
        path = _page_path(off)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                content = _fetch_page(off)
                data = json.loads(content)
                n = len(data.get("features", []))
                path.write_bytes(content)
                _log(off, "ok", n)
                print(f"  offset={off:>7} recovered ({n} records)")
                recovered += 1
                break
            except Exception as e:
                _log(off, f"retry_{attempt}", 0, repr(e))
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SEC)
        else:
            print(f"  offset={off:>7} STILL FAILED after {MAX_RETRIES} retries")
    return recovered


def classify_and_rasterize() -> None:
    """Phase B step 3 — load all pages, classify, reproject, rasterize."""
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    import rasterio
    from rasterio.features import rasterize

    print(f"\nLoading pages from {PAGES_DIR}...")
    pages = []
    for p in sorted(PAGES_DIR.glob("page_*.geojson")):
        try:
            gp = gpd.read_file(p)
        except Exception as e:
            print(f"  WARN: failed to read {p.name}: {e!r}; skip")
            continue
        if len(gp) > 0:
            pages.append(gp)
    if not pages:
        sys.exit("No pages on disk — run --fetch first.")
    print(f"  loaded {len(pages)} pages")

    g = pd.concat(pages, ignore_index=True)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=pages[0].crs)
    print(f"  total records: {len(g):,}")
    print(f"  source CRS as reported: {g.crs}")
    print(f"  source bounds: {g.total_bounds.tolist()}")

    # Portal CRS quirk — same as council districts. If geometry values look
    # like meter-scale magnitudes (|x| > 1000), the data is actually 3857
    # mis-declared as 4326.
    minx, miny, maxx, maxy = g.total_bounds
    if abs(minx) > 1000:
        print("  CRS override: portal-declared CRS is wrong; forcing EPSG:3857")
        g = g.set_crs("EPSG:3857", allow_override=True)

    g_5070 = g.to_crs(TARGET_CRS)
    print(f"  reprojected to {TARGET_CRS}; bounds {g_5070.total_bounds.tolist()}")

    print("\nClassifying...")

    def classify_owner(owner, exempts):
        if owner is None:
            return "unknown"
        O = str(owner).upper()
        if re.match(r"CITY OF SAN ANTONIO", O):                                  return "city"
        if re.search(r"\bSAN ANTONIO.*HOUSING AUTH", O):                         return "city"
        if re.match(r"(BEXAR COUNTY|COUNTY OF BEXAR)", O):                       return "county"
        if re.search(r"\b[A-Z]+ ISD\b|\bINDEPENDENT SCHOOL", O):                 return "isd"
        if re.match(r"STATE OF TEXAS", O):                                       return "state"
        if re.match(r"(UNITED STATES|U S |U\.S\.)", O):                          return "federal"
        if re.search(r"\bSAN ANTONIO RIVER AUTH", O):                            return "river_auth"
        if re.search(r"\bCHURCH\b|\bDIOCESE\b|\bCATHOLIC\b|\bBAPTIST\b|\bMETHODIST\b", O):
            return "church"
        if re.search(r"\bUNIV(ERSITY|\.)\b|\bCOLLEGE\b", O):                     return "university"
        if exempts and ("EX-" in str(exempts) or str(exempts) == "EX"):          return "tax_exempt_other"
        return "private"

    g_5070["owner_class"] = g_5070.apply(
        lambda r: classify_owner(r.get("Owner"), r.get("Exempts")), axis=1
    )
    g_5070["is_public"] = g_5070["owner_class"].isin(PUBLIC_GOVERNMENT_CLASSES).astype(bool)
    state_cd = g_5070["State_cd"].astype(str)
    impr_val = g_5070["ImprVal"].fillna(0).astype(float)
    # Vacancy keys on tax-exemption status, not gov-vs-private:
    #   totally-exempt parcels (any Exempts token matches EX-X[A-Z]) carry
    #     ImprVal==0 because improvements aren't assessed, NOT because the
    #     land is empty. Use C* alone for these. Captures gov + church +
    #     university + tax_exempt_other in one rule.
    #   non-totally-exempt (taxed) parcels are assessed, so the union
    #     C* OR ImprVal==0 correctly catches commercial-vacant + raw
    #     undeveloped private land.
    # NB: Partial exemptions (HS / OV65 / DV* / DP / LIH / HT / etc.) leave
    # the parcel ASSESSED — they must NOT trigger the exempt branch.
    # Empirically verified: EX-X* tokens cluster at 15-30% built-but-zero
    # ("improvements unassessed"); partial-exemption tokens cluster at <0.5%.
    # See PHASE_0_INVESTIGATION.md cross-tab.
    def _is_totally_exempt(ex):
        if ex is None or (isinstance(ex, float) and ex != ex):  # NaN
            return False
        for tok in str(ex).split(","):
            t = tok.strip()
            if len(t) >= 4 and t[:4] == "EX-X" and t[4:5].isalpha():
                return True
        return False
    is_totally_exempt = g_5070["Exempts"].apply(_is_totally_exempt)
    vacant_c    = state_cd.str.startswith("C")
    vacant_zero = (impr_val == 0)
    g_5070["is_totally_exempt"] = is_totally_exempt.astype(bool)
    g_5070["is_vacant"] = (vacant_c | (~is_totally_exempt & vacant_zero)).astype(bool)

    print("\n=== Classifier coverage (full-county) ===")
    print(g_5070["owner_class"].value_counts().to_string())
    print()
    total = len(g_5070)
    n_public = int(g_5070["is_public"].sum())
    n_vacant = int(g_5070["is_vacant"].sum())
    n_both = int((g_5070["is_public"] & g_5070["is_vacant"]).sum())
    print(f"Total parcels:       {total:>9,}")
    print(f"Public (gov-owned):  {n_public:>9,} ({100*n_public/total:.2f}%)")
    print(f"Vacant:              {n_vacant:>9,} ({100*n_vacant/total:.2f}%)")
    print(f"Public AND vacant:   {n_both:>9,} ({100*n_both/total:.2f}%)")

    # Compare against Phase 0 sample (88% gov-owned coverage on 2,000 parcels).
    # The Phase 0 figure was the regex catching the public-bucket; at full
    # scale we expect the same patterns to dominate.
    print(f"\nPhase 0 sample reported government-owned acreage at ~88% via the "
          f"simple-pattern regex. Full-county echo above.")

    print("\nWriting polygon GPKG...")
    POLY_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_cols = [
        "Owner", "State_cd", "ImprVal", "Exempts", "Acres",
        "owner_class", "is_public", "is_totally_exempt", "is_vacant", "geometry",
    ]
    # GeoPackage drops unknown extension types; coerce booleans to int8 for safety.
    out = g_5070[out_cols].copy()
    out["is_public"] = out["is_public"].astype("int8")
    out["is_totally_exempt"] = out["is_totally_exempt"].astype("int8")
    out["is_vacant"] = out["is_vacant"].astype("int8")
    out.to_file(POLY_OUT, driver="GPKG")
    print(f"  wrote {POLY_OUT} ({POLY_OUT.stat().st_size / 1024 / 1024:.1f} MB)")

    print("\nRasterizing to SA grid (EPSG:5070, 30 m)...")
    with rasterio.open(REF_RASTER) as src:
        ref_shape = src.shape
        ref_transform = src.transform
    # Write the runtime raster with explicit EPSG:5070 — the reference
    # GeoTIFF carries the CRS as a `LOCAL_CS` WKT string (not canonical
    # EPSG), which the app's `_assert_raster_crs` correctly rejects.
    codes = np.where(out["is_public"].values & out["is_vacant"].values, 3,
            np.where(out["is_public"].values, 1,
            np.where(out["is_vacant"].values, 2, 0))).astype(np.int8)
    raster = rasterize(
        ((geom, int(c)) for geom, c in zip(g_5070.geometry, codes)),
        out_shape=ref_shape, transform=ref_transform,
        fill=-1, dtype=np.int8,
    )
    with rasterio.open(
        RASTER_OUT, "w",
        driver="GTiff", height=ref_shape[0], width=ref_shape[1],
        count=1, dtype=np.int8, crs="EPSG:5070",
        transform=ref_transform, nodata=-1, compress="deflate",
    ) as dst:
        dst.write(raster, 1)

    unique, counts = np.unique(raster, return_counts=True)
    print(f"  wrote {RASTER_OUT} ({RASTER_OUT.stat().st_size / 1024:.1f} KB)")
    print(f"  shape={raster.shape}, dtype=int8, nodata=-1")
    print(f"  pixel counts: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Finer Ownership Classes Pass — also emit the two-band raster.
    print("\nApplying six-way classifier + rasterizing two-band TIF...")
    g_5070["owner_class_6"] = g_5070["Owner"].map(_classify_six_way)
    _rasterize_two_band(g_5070, RASTER_OUT_2BAND, REF_RASTER)


def _rasterize_two_band(g_5070, out_path, ref_raster_path) -> None:
    """Write a two-band int8 TIF on the SA grid:
      Band 1 = ownership class enum (OWNERSHIP_CLASS_ENUM); nodata=-1.
      Band 2 = is_vacant (0/1); nodata=-1.

    Caller must have populated `owner_class_6` and `is_vacant` columns on
    the GeoDataFrame. Band 1's nodata=-1 distinguishes outside-AOI from
    `private` (which is class code 0).

    Per-class priority rasterization — public classes overwrite private and
    unknown when polygons abut or overlap at the 30m grid edge. A single-
    pass rasterize with the geometries in DataFrame order produced a
    ~67% undercount on city + ~815% overcount on unknown, because rasterio
    is last-write-wins and the natural DataFrame order put private /
    unknown polygons after city/state ones. The per-class build below is
    explicit about who wins each pixel."""
    import numpy as np
    import rasterio
    from rasterio.features import rasterize as _rasterize_fn

    with rasterio.open(ref_raster_path) as src:
        ref_shape = src.shape
        ref_transform = src.transform

    # Priority order — lowest priority first, so higher priority overwrites.
    # Public classes (city, county, state-federal, school-university) take
    # precedence over private and unknown. Within public, order isn't
    # load-bearing — overlap between public classes is negligible (BCAD
    # parcels are largely disjoint) — but we use enum order for stability.
    _PRIORITY_LOW_TO_HIGH = [
        "unknown", "private",
        "state_federal", "county", "school_university", "city",
    ]
    band1 = np.full(ref_shape, -1, dtype=np.int8)
    for cls in _PRIORITY_LOW_TO_HIGH:
        sub = g_5070[g_5070["owner_class_6"] == cls]
        if len(sub) == 0:
            continue
        mask = _rasterize_fn(
            ((geom, 1) for geom in sub.geometry if geom is not None and not geom.is_empty),
            out_shape=ref_shape, transform=ref_transform,
            fill=0, dtype=np.uint8,
        )
        band1[mask.astype(bool)] = OWNERSHIP_CLASS_ENUM[cls]

    # Band 2 — vacancy is orthogonal; single pass is fine since the value
    # written is the parcel's is_vacant flag, not a class enum.
    vacant_codes = g_5070["is_vacant"].astype(bool).astype(np.int8)
    band2 = _rasterize_fn(
        ((geom, int(c)) for geom, c in zip(g_5070.geometry, vacant_codes)
         if geom is not None and not geom.is_empty),
        out_shape=ref_shape, transform=ref_transform,
        fill=-1, dtype=np.int8,
    )
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=ref_shape[0], width=ref_shape[1],
        count=2, dtype=np.int8, crs="EPSG:5070",
        transform=ref_transform, nodata=-1, compress="deflate",
    ) as dst:
        dst.write(band1, 1)
        dst.write(band2, 2)
        dst.set_band_description(1, "ownership_class_6way")
        dst.set_band_description(2, "is_vacant")

    # Per-class pixel counts + acreage (sanity check; reconciliation
    # against OWNERSHIP_FEASIBILITY_PROFILING.md happens in
    # verify_baselines.py).
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"  shape={band1.shape}, dtype=int8 × 2 bands, nodata=-1")
    pixel_area_acres = 0.2224  # 30m × 30m × 1/4046.86, matches app.py:32
    inv_enum = {v: k for k, v in OWNERSHIP_CLASS_ENUM.items()}
    unique, counts = np.unique(band1[band1 != -1], return_counts=True)
    print("  Band 1 per-class pixel counts × acres:")
    for c, n in sorted(zip(unique.tolist(), counts.tolist())):
        name = inv_enum.get(int(c), f"code_{int(c)}")
        print(f"    {name:>20s}: {int(n):>9,} px  ({n * pixel_area_acres:>10,.0f} ac)")
    vac_in_aoi = int((band2 == 1).sum())
    print(f"  Band 2 vacant pixels (in AOI): {vac_in_aoi:,} "
          f"({vac_in_aoi * pixel_area_acres:,.0f} ac)")


def reclassify_from_gpkg(gpkg_path: str) -> None:
    """Read an existing classified GPKG (e.g. the archived BCAD output) and
    apply the new six-way classifier; rasterize as two bands.

    The archived GPKG must carry `Owner` and `is_vacant` columns. CRS is
    coerced to EPSG:5070 if not already in it. No re-fetch from BCAD; uses
    the polygons already in the GPKG."""
    import geopandas as gpd
    import pandas as pd

    print(f"\nReading {gpkg_path}...")
    g = gpd.read_file(gpkg_path)
    print(f"  {len(g):,} polygons; CRS={g.crs}")
    if g.crs is None or str(g.crs) != TARGET_CRS:
        print(f"  reprojecting to {TARGET_CRS}...")
        g = g.to_crs(TARGET_CRS)

    print("\nApplying six-way classifier...")
    g["owner_class_6"] = g["Owner"].map(_classify_six_way)

    if "is_vacant" not in g.columns:
        raise RuntimeError(
            "Archived GPKG missing `is_vacant` column — re-run a full "
            "--finish to regenerate, or extend reclassify_from_gpkg to "
            "re-derive is_vacant from State_cd + ImprVal + Exempts."
        )

    # Acreage report — sanity check before rasterization.
    g["Acres"] = pd.to_numeric(g.get("Acres", 0), errors="coerce").fillna(0)
    total = g["Acres"].sum()
    breakdown = (g.groupby("owner_class_6")["Acres"]
                  .agg(["sum", "count"])
                  .sort_values("sum", ascending=False))
    breakdown.columns = ["acres", "parcels"]
    print(f"\nPer-class acreage breakdown (polygon-area, pre-rasterization):")
    print(f"  total: {total:,.0f} ac across {len(g):,} parcels")
    for cls, row in breakdown.iterrows():
        pct = row["acres"] / total * 100
        print(f"    {cls:>20s}: {row['acres']:>10,.0f} ac  "
              f"({row['parcels']:>7,} parcels, {pct:>5.2f}% of total)")

    print(f"\nRasterizing to {RASTER_OUT_2BAND}...")
    _rasterize_two_band(g, RASTER_OUT_2BAND, REF_RASTER)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true",
                        help="Phase A — paginated BCAD pull")
    parser.add_argument("--verify", action="store_true",
                        help="Phase B step 1 — completeness scan (read-only)")
    parser.add_argument("--finish", action="store_true",
                        help="Phase B steps 2-3 — re-fetch + classify + rasterize")
    parser.add_argument("--reclassify-from", metavar="GPKG",
                        help="Skip fetching; apply the six-way classifier to "
                             "an existing classified GPKG and emit the new "
                             "two-band raster (additive — does not touch the "
                             "legacy single-band raster).")
    args = parser.parse_args()

    if args.fetch:
        fetch_all()
    if args.verify:
        verify_completeness()
    if args.finish:
        # Loop until both completeness AND end-of-data checks pass. Each
        # iteration either re-fetches gap offsets or extends the fetch into
        # fresh territory past the current max_off; bounded by MAX_FINISH_LOOPS
        # to fail loud if something pathological keeps it from converging.
        MAX_FINISH_LOOPS = 5
        for loop_i in range(1, MAX_FINISH_LOOPS + 1):
            missing, reached_eod = verify_completeness()
            if missing:
                print(f"\n[finish loop {loop_i}] Re-fetching {len(missing)} "
                      "gap offsets...")
                refetch(missing)
                continue  # re-verify
            if not reached_eod:
                print(f"\n[finish loop {loop_i}] Extending fetch past current "
                      "max_off until BCAD terminal marker...")
                fetch_all()
                continue  # re-verify
            print(f"\n[finish loop {loop_i}] Completeness + EOD confirmed.")
            break
        else:
            print(f"\nABORT: {MAX_FINISH_LOOPS} finish loops without "
                  "convergence. Investigate.")
            return 1
        classify_and_rasterize()
    if args.reclassify_from:
        reclassify_from_gpkg(args.reclassify_from)
    if not (args.fetch or args.verify or args.finish or args.reclassify_from):
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
