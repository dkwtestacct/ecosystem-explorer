# Demo and Collaboration Notes

**Audience:** Internal
**Status:** Demo / collaboration runbook
**Use this for:** Spoken demos, screenshots, meeting prep, README language
**Do not use this for:** Metric definitions (→ `REFERENCE.md`) or validation status / source of truth (→ `NATCAP_ALIGNMENT.md`)
**Source of truth for:** How to present and talk about the app

---

## 1. One-sentence framing

Ecosystem Explorer is a scenario exploration and discovery layer for urban InVEST analyses — built on a model engine validated against canonical InVEST where comparable inputs exist.

Shorter, spoken: *validated scenario exploration for urban ecosystem tradeoffs.*

## 2. Thirty-second explanation

NatCap's urban InVEST work (e.g. the San Antonio study) produces a fixed set of project scenarios. Planners often want to ask "what about a slightly different design?" — but re-running full InVEST for every variation is slow. Ecosystem Explorer is a fast layer in front of that workflow: its model engine is validated against canonical InVEST for the core urban models, so it can evaluate *new* scenarios with the same methodology, show NatCap's published values alongside as reference points, and export any promising candidate back to canonical InVEST for a full run.

## 3. The honest claim set

The spine of every conversation. Say these plainly — don't overclaim, don't undersell the real result.

**Validated (measured):**
- The core model engine is validated against canonical `natcap.invest` where comparable inputs exist — UCM, UNA, and UMH match per-pixel at MAE ≈ 0. Other metrics (flood, carbon, food, cost, monetized outputs) carry their own validation / prototype badges rather than this measured-parity claim.

**Displayed, not reproduced:**
- NatCap's published San Antonio project-scenario values, where available, are shown as labeled reference points. We surface NatCap's own numbers; we do not independently reproduce all of them, because the exact scenario rasters, aggregation scripts, and model arguments behind some published values weren't available.

**Exploratory:**
- Scenarios the user builds, and scenarios the optimizer suggests, are computed by the same engine — validated against InVEST for the core models, InVEST-aligned for the others — with prototype-only metrics (e.g. food, NDVI) labeled as such. They are for exploring new designs; they are not NatCap-published scenarios.

The framing that makes the data gap read as care, not limitation: *"The app can't reproduce every historical NatCap scenario without the original intermediate rasters — but it uses a validated engine to explore the design space around and beyond those published scenarios, and exports candidates for full canonical validation."*

## 4. Three-minute demo path

Each step: what to click · what to say · which value-ladder rung (STRATEGY §3).

1. **Select San Antonio.** "A real NatCap project city — the San Antonio urban-agriculture study." (L1)
2. **Load a NatCap project scenario.** Sidebar → Scenario source → "NatCap project scenario." Point at the green "NatCap published value" badges. "These are NatCap's own published outcomes, shown as reference points — theirs, not ours." (L1 → L2)
3. **Open the validation status.** "Where we compute, we've verified the engine against canonical InVEST per-pixel — UCM, UNA, UMH at essentially zero error — so it's trustworthy even where no NatCap anchor exists." (the credibility core)
4. **Compare a project scenario with an Explorer one.** Build a comparable scenario with the sliders; open the comparison table. "The Source and Validation columns make clear which is which — NatCap-published reference vs engine-validated exploration." (L2 → L3)
5. **Run Find Best Scenario.** "The optimizer searches the design space for promising candidates worth validating further — discovery, not a decision." (L4)
6. **Export one for InVEST.** Apply a suggestion → full-raster evaluation → Export for InVEST. "Any candidate exports as a runnable canonical InVEST bundle — exploration feeds back into full validation." (L5)

This walks the value ladder L1 → L5 in order, and every claim along the way is honest.

## 5. What to emphasize

- NatCap scenarios are the anchors.
- Explorer scenarios expand the design space.
- The optimizer discovers promising alternatives.
- Full InVEST runs validate candidates — the app hands off, it doesn't replace.

## 6. What not to overclaim

- Not replacing InVEST.
- Not final planning output — it surfaces options for stakeholder conversations.
- Not an exact reproduction of all NatCap published values where the intermediates are missing.
- The engine isn't uniformly canonical — UCM / UNA / UMH are measured at MAE ≈ 0; other metrics are aligned or prototype, per their badges.
- The optimizer suggests candidates; it does not choose the answer.

## 7. Likely questions & answers

**Q: Is this InVEST?**
A: It's an InVEST-aligned exploration layer. The model engine is validated against canonical InVEST for the core urban models where inputs are available, and selected scenarios can be exported for canonical InVEST runs.

**Q: What does the optimizer do?**
A: It searches many plausible scenario combinations quickly to surface options worth testing more rigorously. It's a surrogate — a fast approximation — so promising suggestions should be re-evaluated at full resolution before any weight is put on them.

**Q: Why don't all NatCap published values reproduce exactly?**
A: Some published numbers depend on intermediate rasters or aggregation scripts that weren't available. Where those are missing, the app displays NatCap's published values as references rather than claiming to reproduce them.

**Q: What does this add over just viewing NatCap's InVEST outputs?**
A: Three things a static viewer can't: build and evaluate arbitrary new scenarios with a validated engine; use the optimizer to discover promising designs beyond the fixed set; export any candidate as a runnable canonical InVEST bundle.

**Q: Can I trust the numbers for scenarios with no NatCap anchor?**
A: They're produced by the same engine that matches canonical InVEST per-pixel for the core models — so for those metrics they're genuine InVEST-method outputs. Prototype-only metrics (food, NDVI) are labeled separately; treat those as directional.

**Q: Which metrics are solid vs provisional?**
A: UCM / UNA / UMH are validated against canonical InVEST. San Antonio carbon uses NatCap's four-pool methodology (adopted, not independently reproduced). Food and NDVI are prototype proxies. The badges say which is which on every card.

## 8. Screenshots to use

- Main dashboard: a balanced San Antonio scenario
- The cross-source comparison table (hover the Validation header to confirm the tooltip renders)
- Optimizer / scenario discovery
- An export bundle

## 9. Collaboration asks

The live, canonical list is in **`OPEN_QUESTIONS.md`** — this is a talking-points snapshot, not the source of truth. Current headline asks:

- Missing per-scenario compound LULCs (or the overlay script that built them)
- MN four-pool carbon table
- Per-crop San Antonio yield data, if available
