"""Region-Local Metrics treatment table — pure constants, no logic.

Moved out of `app.py` (Constants Refactor / Task #52). Byte-identical to
the prior app.py definition.

The reconciliation invariant (region_local over the entire AOI == citywide)
is asserted in `verify_baselines.py`; the per-metric decomposition / clip /
caveat fields are consumed in `app.py` for the Selected-region-impact
table and the locked honesty captions surfaced there.
"""

# Region-Local Metrics (`REGION_LOCAL_METRICS_SPEC.md`) — per-metric
# treatment table. Every entry is decomposable under the locked per-model
# treatment from the spec; the field `clip` records which clip (pixel vs
# population) is used, and `caveat` carries the locked honesty caption type
# (`spillover` for UCM reach effects, `routing` for the flood routing
# disclaimer, `cross_boundary` for UNA, `exposure_kernel` for UMH, None
# for clean clips). Reconciliation assertion: for every entry, computing
# region_local over the entire AOI must equal citywide.
_REGION_LOCAL_METRICS = {
    # Pixel-clip, clean (carbon / food / cost / conversions).
    'n_wet':                {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': None},
    'n_for':                {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': None},
    'n_hd':                 {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': None},
    'ff_fellback_pixels':   {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': None},
    'gi_fellback_pixels':   {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': None},
    'hd_fellback_pixels':   {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': None},
    'food_mln_lbs':         {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': None},
    'people_fed':           {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': None},
    'carbon_tons_co2':      {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': None},
    'carbon_value_usd':     {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': None},
    'total_cost_mln':       {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': None},
    'mean_ndvi':            {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': None},
    # Pixel-clip + flood routing caveat (per-pixel runoff retention, not routed hydrology).
    'mean_cn':              {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': 'routing'},
    'flood_reduction':      {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': 'routing'},
    'runoff_acre_feet':     {'decomposable': True,  'clip': 'pixel',      'reach_m': 0,   'caveat': 'routing'},
    'flood_damage_avoided_usd': {'decomposable': True,  'clip': 'pixel',  'reach_m': 0,   'caveat': 'routing'},
    # Pixel-clip + UCM spillover caveat (~600 m reach).
    'mean_hm':              {'decomposable': True,  'clip': 'pixel',      'reach_m': 600, 'caveat': 'spillover'},
    'temp_change_f':        {'decomposable': True,  'clip': 'pixel',      'reach_m': 600, 'caveat': 'spillover'},
    'cooling_energy_savings_usd': {'decomposable': True, 'clip': 'pixel', 'reach_m': 600, 'caveat': 'spillover'},
    # Population-clip + UNA cross-boundary caveat (~800 m reach; supply/access can cross the edge).
    'nature_access_pct':    {'decomposable': True,  'clip': 'population', 'reach_m': 800, 'caveat': 'cross_boundary'},
    'people_with_nature_access': {'decomposable': True, 'clip': 'population', 'reach_m': 800, 'caveat': 'cross_boundary'},
    # Population-clip + UMH exposure-kernel caveat (~300 m reach).
    'preventable_mh_cases': {'decomposable': True,  'clip': 'population', 'reach_m': 300, 'caveat': 'exposure_kernel'},
    'avoided_mh_cost_usd':  {'decomposable': True,  'clip': 'population', 'reach_m': 300, 'caveat': 'exposure_kernel'},
}
