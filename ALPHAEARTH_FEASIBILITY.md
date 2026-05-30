# AlphaEarth Foundations Feasibility Research

**Audience:** Research
**Status:** Research — forward-looking, not acted on
**Use this for:** The feasibility assessment of AlphaEarth embeddings as a future LULC source; agenda for the Yingjie conversation
**Do not use this for:** Current data the app uses (→ DATA_INVENTORY.md) — AlphaEarth is not integrated
**Source of truth for:** AlphaEarth feasibility findings

---

A research note investigating Google DeepMind's AlphaEarth Foundations
satellite embeddings as a potential future data source for the Ecosystem
Explorer. Informs (but does not commit to) future integration work.

**Date:** 2026-05-20
**Scope:** Read-only documentation review. No data fetched, no app code changed.

**Sources:**

- DeepMind blog announcement — https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/
- AlphaEarth Foundations paper (Brown et al. 2025), arXiv:2507.22291 — https://arxiv.org/abs/2507.22291
- Earth Engine catalog entry, "Satellite Embedding V1" — https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- Earth Engine tutorial, "Introduction to the Satellite Embedding Dataset" — https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction
- Google Earth blog, "Now Available on Google Cloud Storage" — https://medium.com/google-earth/alphaearth-foundations-satellite-embeddings-now-available-on-google-cloud-storage-f9ab0f7252d6
- Earth Engine noncommercial tiers — https://developers.google.com/earth-engine/guides/noncommercial_tiers
- Element 84, "Exploring AlphaEarth Embeddings" (independent assessment) — https://element84.com/machine-learning/exploring-alphaearth-embeddings/

---

## Summary

AlphaEarth Foundations is publicly accessible **today**, well-documented, and
released under a permissive license (CC-BY 4.0) that covers a free public tool
like the Ecosystem Explorer. Minneapolis and San Antonio are both within
coverage, and the 2017–2024 annual cadence includes a **2021 layer that lines
up exactly with the app's current NLCD 2021 inputs**. The friction is not
access — it is *shape*. AlphaEarth ships a 64-dimensional continuous embedding
per 10 m pixel, **not** discrete land-cover classes. The entire app is built on
NLCD lucodes (curve-number tables, cooling biophysical tables, conversion
targets), so using AlphaEarth would mean first training an
embeddings→land-cover classifier — genuine upstream work with no NatCap
precedent yet. **Recommendation: do not integrate into the shipped app now, but
the data is ready enough that a small read-only sample pull is a cheap,
worthwhile de-risking step.**

---

## Findings by question

### 1. Access mechanism

Three independent paths, all functional today:

- **Google Earth Engine** — `ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")`.
  Requires a Google account and a Google Cloud project registered for Earth
  Engine. Free for noncommercial/academic use; a one-time research export is
  unambiguously noncommercial. (From April 27, 2026, noncommercial projects
  carry a monthly compute quota across three tiers — Community / Contributor /
  Partner — but a single small export sits well inside the default Community
  tier.)
- **Google Cloud Storage** — public bucket `gs://alphaearth_foundations`, served
  as Cloud-Optimized GeoTIFFs. **Requester-pays**: the data is free, you pay
  egress. Accessible via `gsutil`, the GCS API, or GDAL. No Earth Engine
  registration needed for this path.
- **Source Cooperative** — a community-hosted free mirror (years 2018–2024)
  surfaced in late 2025; avoids both EE registration and requester-pays egress.

For the Ecosystem Explorer the important point is that **none of this is a
runtime dependency**. Any access would be a one-time, offline export; the
shipped Streamlit app would never call Earth Engine or GCS.

### 2. Data format

Confirmed against the Earth Engine catalog entry and the GCS announcement:

- **Resolution:** 10 m per pixel (3× finer than the app's 30 m NLCD grid).
- **Dimensionality:** 64 bands, `A00`–`A63` — a 64-D unit-length embedding
  vector per pixel. (The proposal's "64-dimensional" figure is correct.)
- **Temporal:** annual snapshots, one image per calendar year, **2017–2024**
  (v1.1, Nov 2025, regenerated the 2017 layer). Not a time series within a
  year; not cumulative. A **2021 layer exists** — direct match for the app's
  NLCD 2021 baseline.
- **File format:** Earth Engine `Image` per year/UTM-zone; on GCS, Cloud-
  Optimized GeoTIFF tiles of 8192×8192 px, 64 channels, with overviews.
- **Encoding:** signed 8-bit integers (−127…127; −128 = nodata). De-quantize
  with: divide by 127.5, square, re-apply sign.
- **CRS:** UTM, zone per tile (carried in a `UTM_ZONE` property); WGS84 datum.
  Minneapolis (~93°W) falls in UTM zone 15 — near-identical to the app's
  EPSG:26915 (NAD83 / UTM 15N), differing only by the NAD83-vs-WGS84 datum
  (~1 m). San Antonio (~98°W) falls in UTM zone 14 and would need reprojection
  to the app's EPSG:5070 (Conus Albers) — a standard one-line GDAL/rasterio
  warp.

### 3. Geographic and temporal coverage

- **Minneapolis, MN (~93°W, 45°N):** covered. Global terrestrial coverage spans
  ±82° latitude; 45°N is well inside.
- **Bexar County / San Antonio, TX (~98°W, 29°N):** covered.
- **Years:** 2017–2024, annual, stable versioned releases (currently v1.1).
  Updates are versioned re-releases, not a silent rolling feed. The 2021 layer
  matches the app's current NLCD 2021 vintage.

### 4. License and use terms

- The **dataset** is released under **Creative Commons Attribution 4.0
  (CC-BY 4.0)**. CC-BY explicitly permits derivative works *and* commercial use,
  so it covers the Ecosystem Explorer (a free public tool) and would still
  cover it if the project ever became commercial.
- **Attribution is mandatory**, with prescribed wording: *"The AlphaEarth
  Foundations Satellite Embedding dataset is produced by Google and Google
  DeepMind."* The paper (Brown et al. 2025, arXiv:2507.22291) should also be
  cited.
- **Caveat — two separate licenses.** CC-BY 4.0 governs the *data*. The Earth
  Engine *platform* has its own Terms of Service with noncommercial-use
  restrictions (no fee-for-service). This only matters if you run EE compute;
  it does not constrain a one-time data export, and it does not touch the GCS
  or Source Cooperative paths at all.
- The **AlphaEarth model itself and its training code are not open source** —
  consumers depend on Google for future releases. Relevant to long-term
  reliance, not to using the published embeddings.

### 5. Integration complexity

Mechanically straightforward; conceptually a real lift.

- **Python clients exist:** `earthengine-api` (the `ee` package), plus
  `google-cloud-storage` / `gsutil` / GDAL for the GCS path. No exotic
  tooling.
- **No GPU required.** AlphaEarth's whole value proposition is that the
  expensive deep-learning step is already done — the embeddings are
  "analysis-ready." Documented and independently-confirmed downstream
  workflows train *light* classifiers (logistic regression, random forest,
  k-means) on top of the 64-D vectors. Element 84 reports "remarkably good"
  results from a plain logistic-regression classifier on a tiny label set.
  This runs comfortably on the app's existing CPU infrastructure — and it is
  the same `RandomForestClassifier`-class tooling already in `surrogate.py`.
- **Data volume:** the embeddings are large but only offline. MN downtown
  (~10.8 × 10.7 km) at 10 m × 64 bands ≈ 75 MB as int8 (~300 MB float32) —
  fine in memory. The San Antonio AOI (~51 × 60 km) at 10 m × 64 bands ≈
  2 GB int8 (~8 GB float32) — would blow the app's 1 GB Streamlit Cloud
  ceiling **if loaded raw**. It would not be: classification happens offline,
  and the app would ship only the derived land-cover raster — same size and
  role as today's NLCD GeoTIFFs.
- **No pre-trained AlphaEarth→NLCD classifier exists.** This is the crux.
  AlphaEarth is deliberately task-agnostic; it gives you features, not labels.
  To feed the app's NLCD-class machinery you would have to train an
  embeddings→NLCD-class model yourself (needs NLCD or other land-cover labels
  as training targets) and validate it. That is a self-contained research
  project, not a data-swap.

### 6. Alignment with NatCap's direction

**No public documentation connects NatCap / InVEST to AlphaEarth.** Searches
across the InVEST docs, NatCap's site, and the literature surfaced nothing —
no blog post, no InVEST release note, no joint paper. AlphaEarth is publicly
framed as a *general-purpose, task-agnostic geospatial embedding model*, with
land-cover classification as one of several documented downstream uses
(alongside crop mapping, change detection, biophysical regression). Adopters
named by Google are mapping organizations (FAO, MapBiomas, Harvard Forest,
Stanford), not ecosystem-services modelers.

This absence is itself a finding: **the AlphaEarth ↔ InVEST connection is not
public.** The proposal Yingjie shared frames it as a future direction, which
private team conversations may already be exploring — but nothing public
confirms an active NatCap pipeline. Treat it as a genuinely upstream research
direction, not a near-term standard.

---

## Honest gaps

What this read-only review **could not** answer:

- **NatCap's actual timeline and intent.** Whether NatCap is actively building
  an AlphaEarth pipeline, and on what horizon, is not public — only Yingjie /
  the NatCap team can say.
- **Whether an embeddings→NLCD classifier reaches usable accuracy for this
  app.** Published results report ~5-point accuracy gains for *coarse*
  land-cover schemes (Urban / Water / Bare / Vegetation). The app needs ~13
  fine NLCD classes (21/22/23/24 developed-intensity tiers, 41/42/43 forest
  types, etc.). Whether AlphaEarth resolves those distinctions well enough to
  drive curve-number and cooling tables is unknown without actually training
  and validating a classifier.
- **Real-world access friction.** The docs say EE registration and the
  requester-pays bucket work; only an actual export attempt confirms it for
  our specific AOIs, and reveals the true egress cost.
- **Long-term licensing posture.** CC-BY 4.0 is unambiguous for the *current*
  release. Whether future releases stay CC-BY, and any license nuance for a
  *commercial* future of the tool, is a question for the AlphaEarth/DeepMind
  team.
- **What AlphaEarth adds over NLCD for *this* app.** AlphaEarth's headline
  strengths — 10 m resolution, annual cadence, cloud-gap-free — are real, but
  the app's metrics are built on NLCD semantics. The actual analytical gain
  from switching is unquantified and would need a side-by-side comparison.

These are exactly the questions worth raising with Yingjie at the Symposium:
"I looked into AlphaEarth — here's what's accessible and licensed, and here's
what I couldn't answer without your team's input."

---

## Implications for the Ecosystem Explorer

**Do not integrate AlphaEarth into the shipped app in the near term.** The
blocker is not access or licensing — both are clean. It is that AlphaEarth
delivers *features*, while every downstream calculation in this app
(`evaluate_scenario`, the CN tables, the cooling biophysical tables, the
conversion-target logic) consumes discrete *NLCD classes*. Bridging that gap
means building and validating an embeddings→NLCD classifier — a research
project upstream of where the app is, with no NatCap precedent to lean on.

But the data is mature and the license is favorable, so the project is not
blocked from *preparing*. The high-value, low-cost move is a small read-only
sample pull: it de-risks the access path, lets a classifier prototype be
spiked offline (entirely outside the app, no leakage into `evaluate_scenario`
or `verify_baselines.py`), and turns the Symposium conversation with Yingjie
from abstract into concrete.

This is the "feasible but genuinely upstream — prepare, don't commit"
conclusion, not a hedge: a clear no on app integration now, a clear yes on a
bounded sample-pull spike.

---

## Recommended next steps

1. **Authorize the follow-up sample brief.** Pull the **2021 Minneapolis
   downtown** Satellite Embedding tile via Earth Engine (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`,
   2021, clipped to the existing NLCD template). MN is the right test bed: its
   UTM-15 tile needs essentially no reprojection to EPSG:26915, and at ~75 MB
   (int8) it is trivially small. Keep it entirely under a research/ scratch
   path — no `data/` writes, no app wiring.
2. **Offline classifier spike.** With the sample in hand, train a quick
   logistic-regression or random-forest classifier from AlphaEarth embeddings
   to NLCD 2021 classes and measure per-class accuracy. This is the single
   experiment that answers "is this worth it" — and it reuses tooling already
   in `surrogate.py`.
3. **Take the open questions to the Symposium.** Use the "Honest gaps" list as
   the agenda for the Yingjie conversation — especially NatCap's timeline and
   whether they expect to publish an AlphaEarth→InVEST pipeline the app could
   adopt rather than build.
4. **Revisit integration only after** either the classifier spike clears an
   accuracy bar or NatCap publishes a pipeline. Until then, NLCD 2021 remains
   the right input for the app.
