#!/usr/bin/env python3
"""Vocabulary guard — fail if a retired term reappears on a user-facing surface.

Wired into `verify_baselines.py` so it rides the regression gate (Relay 26).

Scope (intentional): `app.py` + the current user/collaborator-facing docs
(`REFERENCE.md`, `CAPABILITIES.md`, `README.md`). Internal rationale/history docs
(`docs/internal/*`, `docs/archive/*`) are EXCLUDED on purpose — they record
superseded decisions and legitimately name retired terms when explaining how the
vocabulary got here. Guarding them would create false pressure to rewrite history.

Honest limit: "surrogate" is deliberately NOT guarded. It has legitimate live uses
(code identifiers like `surrogate.py` / `SURROGATE_TREES`, the one "How this
prototype works" tooltip, and deep methodology paragraphs). It stays a human
review item, not a machine check — pretending the guard covers it would be
dishonest.

Per-line opt-out: a line containing the marker ``vocab-allow`` is skipped. This is
used by the canonical glossary's "Retired:" lines and by sentences that name a
removed card to explain its removal (e.g. "the earlier 'Flood Volume Reduction'
card is removed").

Exit status: 0 when clean, 1 when any retired term is found (matching is
case-sensitive — the listed phrases are the unambiguous retired forms).
"""
import sys
from pathlib import Path

# Unambiguous retired phrases. Case-sensitive on purpose: the capitalized metric
# labels ("Flood Retention", "Flood Volume Reduction") must not false-positive on
# generic lowercase prose, and these are the exact forms that were retired.
RETIRED_TERMS = [
    "uncertainty band",
    "canonical engine",
    "Flood Retention",
    "flood-reduction index",
    "Flood Volume Reduction",
    "mental-health proxy",
    "mental-health effects",
    # Relay 31: the visible evaluator noun is now "InVEST-aligned evaluator".
    # "full evaluator" stays legal in internal docs (those are out of scope).
    "full evaluator",
    "full-evaluator",
    # Relay 39: the overlay control is "Urban intensity overlay" now. Guard the
    # DISTINCTIVE phrase only (both casings of the old label) — bare "heat" /
    # "heat proxy" stay legal in code comments / internal docs.
    "development-intensity heat proxy",
    "Development-intensity heat proxy",
    # Relay 47: visible evaluator synonyms are retired in favor of
    # "InVEST-aligned evaluator". "raster engine" covers "full raster engine"
    # too; no legitimate use found on the scanned surfaces.
    "raster engine",
    "full raster engine",
    # Relay 52: the hyphenated "full-raster" descriptor too — cleared from all
    # scanned surfaces (it stays legal in internal docs, which aren't scanned).
    "full-raster",
    # Relay 53: distinctive variant forms the bare-term guards miss (all
    # confirmed absent from scanned surfaces). Hyphenated / badge / "-driven"
    # forms ONLY — bare "surrogate" (legit tooltip + code use) and the space
    # forms "canonical engine" / "uncertainty band" / "Flood Retention" are
    # handled separately, so these add no false positives.
    "canonical-engine",
    "Surrogate-suggested",
    "surrogate-suggested",
    "surrogate-driven",
    "uncertainty-band",
    "flood-retention",
]

ALLOW_MARKER = "vocab-allow"

# User-facing surfaces only. See module docstring for why internal docs are out.
SCANNED_FILES = [
    "app.py",
    "REFERENCE.md",
    "CAPABILITIES.md",
    "README.md",
]


def find_hits_in_text(text):
    """Return ``[(lineno, term, line)]`` for retired terms in ``text``, honoring
    the per-line allow marker. Shared by ``scan()`` and ``selftest()`` so the
    meta-test exercises the real detection path, not a parallel reimplementation.
    """
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARKER in line:
            continue
        for term in RETIRED_TERMS:
            if term in line:
                hits.append((lineno, term, line.strip()))
    return hits


def scan():
    root = Path(__file__).resolve().parent.parent
    hits = []
    for rel in SCANNED_FILES:
        path = root / rel
        if not path.exists():
            continue
        for lineno, term, line in find_hits_in_text(path.read_text(encoding="utf-8")):
            hits.append((rel, lineno, term, line))
    return hits


def selftest():
    """Prove the guard has teeth — no vacuous pass.

    A seeded retired term MUST be caught, the allow marker MUST suppress it, and
    canonical copy MUST stay clean. Returns 0 when detection works, 1 otherwise.
    Wired into ``verify_baselines.py`` as a meta-test so the guard's teeth ride
    the gate (same discipline as the Assertion-C swap test).
    """
    seeded = "An accidental Flood Retention card slipped into the copy."
    suppressed = f"Retired: Flood Retention {ALLOW_MARKER}"
    clean = "The Flood Index is computed by the InVEST-aligned evaluator."
    eval_seeded = "Apply one to recompute it with the full evaluator."
    raster_seeded = "Cards reflect a full-raster evaluation on apply."
    seeded_hits = find_hits_in_text(seeded)
    eval_hits = find_hits_in_text(eval_seeded)
    raster_hits = find_hits_in_text(raster_seeded)
    ok = (
        len(seeded_hits) == 1
        and seeded_hits[0][1] == "Flood Retention"
        and find_hits_in_text(suppressed) == []
        and find_hits_in_text(clean) == []
        # Relay 31 entry has teeth: the retired evaluator noun is caught.
        and len(eval_hits) == 1
        and eval_hits[0][1] == "full evaluator"
        # Relay 52 entry has teeth: the "full-raster" descriptor is caught.
        and len(raster_hits) == 1
        and raster_hits[0][1] == "full-raster"
    )
    # Relay 53 — each distinctive variant form must be caught, AND the legit
    # bare-"surrogate" tooltip use must NOT be flagged (no false positive).
    variant_seeds = {
        "canonical-engine":    "Computed by the canonical-engine-verified models.",
        "Surrogate-suggested": "Provenance flips to Surrogate-suggested.",
        "surrogate-suggested": "an applied surrogate-suggested scenario",
        "surrogate-driven":    "the surrogate-driven optimizer is reframed",
        "uncertainty-band":    "10th-90th uncertainty-band shading on the chart",
        "flood-retention":     "strongest per-pixel flood-retention effect",
    }
    variants_ok = all(
        v in {t for (_l, t, _ln) in find_hits_in_text(seed)}
        for v, seed in variant_seeds.items()
    )
    bare_surrogate_ok = find_hits_in_text(
        "Fast machine-learning model: a surrogate trained on precomputed runs."
    ) == []
    return 0 if (ok and variants_ok and bare_surrogate_ok) else 1


def main():
    hits = scan()
    if hits:
        print("Vocabulary guard FAILED — retired term(s) on user-facing surfaces:")
        for rel, lineno, term, line in hits:
            print(f"  {rel}:{lineno}: [{term}] {line[:120]}")
        print(
            f"\n{len(hits)} hit(s). Replace with the canonical term (REFERENCE.md "
            f"§ \"Vocabulary (canonical terms)\"), or append the '{ALLOW_MARKER}' "
            "marker if this is a deliberate historical mention."
        )
        return 1
    print(
        f"Vocabulary guard OK — 0 retired terms across {', '.join(SCANNED_FILES)}."
    )
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _rc = selftest()
        print("selftest OK — guard catches a seeded retired term and honors the "
              "allow marker." if _rc == 0 else
              "selftest FAILED — guard is vacuous (seeded term not caught).")
        sys.exit(_rc)
    sys.exit(main())
