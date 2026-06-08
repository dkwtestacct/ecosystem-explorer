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
]

ALLOW_MARKER = "vocab-allow"

# User-facing surfaces only. See module docstring for why internal docs are out.
SCANNED_FILES = [
    "app.py",
    "REFERENCE.md",
    "CAPABILITIES.md",
    "README.md",
]


def scan():
    root = Path(__file__).resolve().parent.parent
    hits = []
    for rel in SCANNED_FILES:
        path = root / rel
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if ALLOW_MARKER in line:
                continue
            for term in RETIRED_TERMS:
                if term in line:
                    hits.append((rel, lineno, term, line.strip()))
    return hits


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
    sys.exit(main())
