# Contributing notes

**Audience:** Developer
**Status:** Current
**Use this for:** Environment setup and running the canonical-InVEST validation harness (the two-env pattern, PROJ_DATA/GDAL_DATA)
**Do not use this for:** Architecture overview (→ ../internal/ARCHITECTURE.md) or metric methodology (→ ../../REFERENCE.md)
**Source of truth for:** How to set up and validate the project

---

## Canonical-InVEST validation environments

The prototype reimplements several InVEST models in numpy (Urban Cooling,
Urban Flood Risk, Urban Nature Access, Urban Mental Health, Carbon) — it does
**not** import `natcap.invest` at runtime (it isn't in `requirements.txt`).
Each reimplementation is validated offline against canonical
`natcap.invest.*.execute()` by a `compare_*_invest.py` harness, which reports
MAE / Pearson r.

There are **two** validation environments, because no single interpreter has
every InVEST model the prototype reimplements:

| Env | Python | natcap.invest | Has | Used by |
|-----|--------|---------------|-----|---------|
| anaconda **base** | 3.13 | 3.16.2 | UCM, UFR, UNA, Carbon | `compare_ucm_invest.py`, `compare_una_invest.py`, `compare_carbon_invest.py` |
| **`natcap_umh_validation`** (conda) | 3.12 | 3.19.0 | + Urban Mental Health | `compare_umh_invest.py` |

Urban Mental Health was added in InVEST **3.18/3.19**, which require **Python
≥ 3.10** — the app's `.venv` (Python 3.9) cannot host it, and anaconda base's
3.16.2 has no `urban_mental_health` module. So UMH validation lives in its own
isolated env. **Do not upgrade anaconda base or the app `.venv`** for this — it
would risk the other (MAE≈0) harnesses and the app's own runtime stack.

### Recreate the UMH validation env

```bash
conda create -y -n natcap_umh_validation -c conda-forge python=3.12 "natcap.invest=3.19.0"
conda install  -y -n natcap_umh_validation -c conda-forge pyproj   # geopandas CRS support
# sanity check:
conda run -n natcap_umh_validation python -c \
  "from natcap.invest import urban_mental_health as u; print(u.__name__)"
```

### The decoupled two-environment harness pattern

`compare_umh_invest.py` can't `import app` in the isolated env (the env
deliberately lacks app's rasterio/sklearn/scikit-image stack). Instead it
bridges the two environments on disk:

1. **EXPORT** (run in the app `.venv`, which has app's full stack):
   ```bash
   PROJ_DATA=.venv/lib/python3.9/site-packages/rasterio/proj_data \
   GDAL_DATA=.venv/lib/python3.9/site-packages/rasterio/gdal_data \
   .venv/bin/python compare_umh_invest.py export
   ```
   Writes shared inputs (NDVI base/alt, population) and the prototype's
   per-pixel output rasters to `tests/umh_fixtures/<city>/` (gitignored,
   ~67 MB — regenerate, don't commit).

2. **COMPARE** (run in the isolated env):
   ```bash
   conda run -n natcap_umh_validation python compare_umh_invest.py compare
   ```
   Feeds the *same* inputs into canonical `execute()` and reports MAE +
   Pearson r vs the prototype's rasters.

This pattern generalizes to any future canonical-InVEST validation where the
canonical model needs a newer interpreter than the app `.venv`: keep the
prototype side in `.venv`, the canonical side in an isolated env, and compare
exported rasters. See `../internal/DESIGN_NOTES.md` "UMH validation against canonical
InVEST 3.19.0" for the result and interpretation.

> The `PROJ_DATA` / `GDAL_DATA` override on the EXPORT command works around the
> app `.venv` picking up anaconda's stale `proj.db`; the isolated conda env sets
> its own PROJ data via activation and needs no override.
