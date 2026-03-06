import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

# Point this to your JSON exported from the XLSX (sheet-centric JSON).
ISO_JSON_PATH = Path("iso286_tables.json")  # change if needed

# If your stored deviations are in micrometres (µm), use 1000.0 to convert -> mm.
UM_PER_MM = 1000.0


def _to_unit(value_mm, unit: str):
    """Convert a value (or array) from mm to the requested unit ('mm' or 'um')."""
    u = (unit or "mm").strip().lower()
    if u in ("mm", "millimeter", "millimetre", "millimeters", "millimetres"):
        return value_mm
    if u in ("um", "µm", "micrometer", "micrometre", "micrometers", "micrometres"):
        return value_mm * UM_PER_MM
    raise ValueError("unit must be 'mm' or 'um'")


def _convert_pair(low_mm, high_mm, unit: str):
    """Convert a (low, high) pair from mm to unit."""
    return _to_unit(low_mm, unit), _to_unit(high_mm, unit)


# =============================================================================
# JSON loading + basic helpers
# =============================================================================

_TABLES: Optional[Dict[str, Any]] = None

def _load_tables(path: Path = ISO_JSON_PATH) -> Dict[str, Any]:
    global _TABLES
    if _TABLES is not None:
        return _TABLES
    if not path.exists():
        raise FileNotFoundError(f"Missing ISO JSON file: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        _TABLES = json.load(f)
    if not isinstance(_TABLES, dict) or "sheets" not in _TABLES:
        raise ValueError("iso286_tables.json must contain a top-level object with key 'sheets'")
    return _TABLES


_CODE_RE = re.compile(r"^([A-Za-z]{1,2})(\d+)$")  # H7, ZA11, js6

def _parse_tol_code(code: str) -> Tuple[str, int]:
    code = code.strip()
    m = _CODE_RE.match(code)
    if not m:
        raise ValueError(f"Bad tolerance code: {code!r} (expected like 'H7', 'ZA11', 'h6', 'js7')")
    zone = m.group(1)
    grade = int(m.group(2))
    return zone, grade

def _is_hole_zone(zone: str) -> bool:
    # JS is a HOLE-zone by convention; js is shaft
    return zone == "JS" or (len(zone) > 0 and zone[0].isupper())

def _is_shaft_zone(zone: str) -> bool:
    return zone == "js" or (len(zone) > 0 and zone[0].islower())

def _band_match(d_mm: float, above: Any, upto: Any) -> bool:
    """
    ISO band rule: (Above, Up to and including] == (lo_excl, hi_incl]
    - Above is strict
    - Up_to_and_including is inclusive
    """
    if upto is None:
        return False
    hi = float(str(upto).replace(" ", "").replace(",", "."))
    if above is None or (isinstance(above, str) and above.strip() in {"—", "-"}):
        return d_mm <= hi
    lo = float(str(above).replace(" ", "").replace(",", "."))
    return (d_mm > lo) and (d_mm <= hi)


def _tol_dev_um_scalar(db: Dict[str, Any], d_mm: float, code: str) -> Tuple[float, float]:
    """
    Look up deviations (lower, upper) in µm for a scalar d_mm and code like H7.
    If code is a shaft (lowercase) and not present, mirrors the hole (uppercase).
    """
    zone, grade = _parse_tol_code(code)

    # Try direct columns first
    col_l = f"{zone}{grade}_l"
    col_u = f"{zone}{grade}_u"

    sheets = db.get("sheets", {})
    if not isinstance(sheets, dict):
        raise ValueError("JSON schema error: 'sheets' must be a dict")

    for sheet_name, sheet in sheets.items():
        rows = sheet.get("rows", [])
        if not isinstance(rows, list):
            continue

        cols = sheet.get("columns") or sheet.get("tolerance_columns") or []
        if (col_l not in cols) or (col_u not in cols):
            continue

        for r in rows:
            if _band_match(d_mm, r.get("Above_mm", None), r.get("Up_to_and_including_mm", None)):
                lo = r.get(col_l, None)
                up = r.get(col_u, None)
                if lo is None or up is None:
                    raise KeyError(f"Found band in sheet {sheet_name!r} but {col_l}/{col_u} missing in that row")
                return float(lo), float(up)

    # If not found and it's a shaft, mirror from hole
    if _is_shaft_zone(zone):
        hole_code = f"{zone.upper()}{grade}"
        hole_lo, hole_up = _tol_dev_um_scalar(db, d_mm, hole_code)
        return -hole_up, -hole_lo  # mirror: (lower, upper) = (-upper_hole, -lower_hole)

    raise KeyError(f"Could not find tolerance columns for {code!r} at d={d_mm} mm in any sheet")


def _maybe_scalarize(x):
    if np is None:
        return x
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x


# =============================================================================
# IT computation (ISO 286-1 style formulas)
# =============================================================================

# Size steps used for computing D (geometric mean) for the "i" unit.
# Intervals are (lo_excl, hi_incl]
_SIZE_STEPS = [
    (0.0, 3.0),
    (3.0, 6.0),
    (6.0, 10.0),
    (10.0, 18.0),
    (18.0, 30.0),
    (30.0, 50.0),
    (50.0, 80.0),
    (80.0, 120.0),
    (120.0, 180.0),
    (180.0, 250.0),
    (250.0, 315.0),
    (315.0, 400.0),
    (400.0, 500.0),
    (500.0, 630.0),
    (630.0, 800.0),
    (800.0, 1000.0),
    (1000.0, 1250.0),
    (1250.0, 1600.0),
    (1600.0, 2000.0),
    (2000.0, 2500.0),
    (2500.0, 3150.0),
]

# Multipliers for IT5..IT18 in terms of i (µm)
_IT_MULT = {
    5: 7, 6: 10, 7: 16, 8: 25, 9: 40, 10: 64, 11: 100, 12: 160,
    13: 250, 14: 400, 15: 640, 16: 1000, 17: 1400, 18: 2000,
}

def _geom_mean_D(d_mm: float) -> float:
    """Return geometric mean D for the size step containing d_mm."""
    if d_mm <= 0:
        raise ValueError("nominal_mm must be > 0")
    for lo, hi in _SIZE_STEPS:
        if (d_mm > lo) and (d_mm <= hi):
            lo_eff = lo if lo > 0 else 1.0  # conventional handling for first step
            return math.sqrt(lo_eff * hi)
    raise ValueError("nominal_mm out of supported ISO 286 range (0 < d ≤ 3150 mm)")

def it(nominal_mm, grade: int) -> Any:
    """
    Compute IT tolerance width for nominal size(s) in mm.

    Returns IT width in **mm**.

    Supports grades: 01, 0, 1..18 (pass 1,2,... and for 01 pass grade=1 with flag?).
    Practical: accepts grade as int:
      - use grade=0 for IT0
      - use grade=1 for IT1
      - IT01 is not representable as int cleanly; if you need it, call it01().
    """
    g = int(grade)

    def _it_um_scalar(d: float, g_int: int) -> float:
        D = _geom_mean_D(d)
        if g_int in _IT_MULT:
            i = 0.45 * (D ** (1/3)) + 0.001 * D  # µm
            return _IT_MULT[g_int] * i
        # IT0..IT4 use linear formulas (µm). (Common ISO 286-1 formulation.)
        if g_int == 0:
            return 0.5 + 0.012 * D
        if g_int == 1:
            return 0.8 + 0.02 * D
        if g_int == 2:
            return 1.2 + 0.03 * D
        if g_int == 3:
            return 2.0 + 0.05 * D
        if g_int == 4:
            return 3.0 + 0.08 * D
        raise ValueError(f"Unsupported IT grade: {g_int}")

    if np is None:
        um = _it_um_scalar(float(nominal_mm), g)
        return um / UM_PER_MM

    d = np.asarray(nominal_mm, dtype=float)
    if np.any(d <= 0):
        raise ValueError("nominal_mm must be > 0")

    out_um = np.empty_like(d, dtype=float)
    # vectorize by looping elements (still fast enough for typical use)
    it_scalar = np.vectorize(lambda x: _it_um_scalar(float(x), g), otypes=[float])
    out_um = it_scalar(d)
    return _maybe_scalarize(out_um / UM_PER_MM)


# Optional helper if you ever need IT01 (since it's not an int nicely)
def it01(nominal_mm) -> Any:
    """Compute IT01 in mm."""
    def _it01_um_scalar(d: float) -> float:
        D = _geom_mean_D(d)
        return 0.3 + 0.008 * D  # µm
    if np is None:
        return _it01_um_scalar(float(nominal_mm)) / UM_PER_MM
    d = np.asarray(nominal_mm, dtype=float)
    it_scalar = np.vectorize(lambda x: _it01_um_scalar(float(x)), otypes=[float])
    return _maybe_scalarize(it_scalar(d) / UM_PER_MM)


# =============================================================================
# iso_tol: deviations or absolute limits
# =============================================================================

def iso_tol(nominal_mm, tol: str, mode: str = "abs", unit: str = "mm") -> Tuple[Any, Any]:
    """
    Compute ISO tolerance limits for a hole/shaft designation like 'H7', 'h6', 'JS7'.

    Parameters
    ----------
    nominal_mm : float or array-like
    tol : str
        e.g. 'H7', 'h6', 'ZA11', 'js7'
    mode : 'abs' or 'dev'
        - 'dev' returns (lower_dev_mm, upper_dev_mm)
        - 'abs' returns (lower_limit_mm, upper_limit_mm) [default]
    unit : 'mm' or 'um'
        Output unit. Default 'mm'.

    Notes
    -----
    - For 'JS'/'js' (symmetric about nominal), deviations are ±IT/2.
    - For lowercase (shaft) codes not present in JSON, values are mirrored from hole (uppercase).
    """
    if mode not in ("abs", "dev"):
        raise ValueError("mode must be 'abs' or 'dev'")

    zone, grade = _parse_tol_code(tol.strip())

    # JS/js: symmetric around nominal via IT/2 (in mm)
    if zone in ("JS", "js"):
        IT = it(nominal_mm, grade)
        low_dev = -IT / 2
        high_dev = IT / 2
        if mode == "dev":
            return _convert_pair(low_dev, high_dev, unit)
        d = float(nominal_mm)
        return _convert_pair(d + low_dev, d + high_dev, unit)

    db = _load_tables()

    if np is None:
        lo_um, up_um = _tol_dev_um_scalar(db, float(nominal_mm), tol)
        lo = lo_um / UM_PER_MM
        up = up_um / UM_PER_MM
        if mode == "dev":
            return _convert_pair(lo, up, unit)
        d = float(nominal_mm)
        return _convert_pair(d + lo, d + up, unit)

    d = np.asarray(nominal_mm, dtype=float)
    if np.any(d <= 0):
        raise ValueError("nominal_mm must be > 0")

    # elementwise lookups (sheet JSON means we can't do pure vectorized search easily)
    lo = np.empty_like(d, dtype=float)
    up = np.empty_like(d, dtype=float)

    it_flat = np.nditer(d, flags=["multi_index"])
    for x in it_flat:
        idx = it_flat.multi_index
        lo_um, up_um = _tol_dev_um_scalar(db, float(x), tol)
        lo[idx] = lo_um / UM_PER_MM
        up[idx] = up_um / UM_PER_MM

    if mode == "dev":
        lo_out, up_out = _convert_pair(lo, up, unit)
        return _maybe_scalarize(lo_out), _maybe_scalarize(up_out)

    lo_abs, up_abs = d + lo, d + up
    lo_out, up_out = _convert_pair(lo_abs, up_abs, unit)
    return _maybe_scalarize(lo_out), _maybe_scalarize(up_out)


# =============================================================================
# fit_check
# =============================================================================

def fit_check(nominal_mm, tol1: str, tol2: str, unit: str = "mm") -> Dict[str, Any]:
    """
    Compute fit between a hole tolerance and a shaft tolerance.

    Works regardless of argument order: determines hole vs shaft from case:
      - Hole: uppercase (e.g. H7, ZA11) or 'JS'
      - Shaft: lowercase (e.g. h6, za11? no — shafts are lowercase letters) or 'js'

    Returns dict:
      hole_limits: (hole_low, hole_high)
      shaft_limits: (shaft_low, shaft_high)
      clearance: (min_clearance, max_clearance) [mm]
      overlap: overlap_length [mm]
      fit_type: 'clearance'|'transition'|'interference'

    Note
    ----
    The `unit` parameter controls the output units ('mm' by default, 'um' allowed).
    """
    z1, _ = _parse_tol_code(tol1.strip())
    z2, _ = _parse_tol_code(tol2.strip())

    # Determine which is hole and which is shaft
    is1_hole = _is_hole_zone(z1)
    is2_hole = _is_hole_zone(z2)
    is1_shaft = _is_shaft_zone(z1)
    is2_shaft = _is_shaft_zone(z2)

    if (is1_hole and is2_hole) or (is1_shaft and is2_shaft):
        raise ValueError(
            f"fit_check needs one hole and one shaft tolerance. Got: {tol1!r} and {tol2!r}"
        )

    hole_tol = tol1 if is1_hole else tol2
    shaft_tol = tol2 if is1_hole else tol1

    hole_low, hole_high = iso_tol(nominal_mm, hole_tol, mode="abs", unit="mm")
    shaft_low, shaft_high = iso_tol(nominal_mm, shaft_tol, mode="abs", unit="mm")

    if np is None:
        min_clear = hole_low - shaft_high
        max_clear = hole_high - shaft_low
        overlap = max(0.0, min(hole_high, shaft_high) - max(hole_low, shaft_low))

        if min_clear > 0:
            fit_type = "clearance"
        elif max_clear < 0:
            fit_type = "interference"
        else:
            fit_type = "transition"

        hole_limits = _convert_pair(hole_low, hole_high, unit)
        shaft_limits = _convert_pair(shaft_low, shaft_high, unit)
        clearance = _convert_pair(min_clear, max_clear, unit)
        overlap_out = _to_unit(overlap, unit)

        return {
            "hole_limits": hole_limits,
            "shaft_limits": shaft_limits,
            "clearance": clearance,
            "overlap": overlap_out,
            "fit_type": fit_type,
        }

    # numpy path
    hole_low = np.asarray(hole_low, dtype=float)
    hole_high = np.asarray(hole_high, dtype=float)
    shaft_low = np.asarray(shaft_low, dtype=float)
    shaft_high = np.asarray(shaft_high, dtype=float)

    min_clear = hole_low - shaft_high
    max_clear = hole_high - shaft_low
    overlap = np.maximum(0.0, np.minimum(hole_high, shaft_high) - np.maximum(hole_low, shaft_low))

    fit_type = np.full(overlap.shape, "transition", dtype=object)
    fit_type[min_clear > 0] = "clearance"
    fit_type[max_clear < 0] = "interference"

    hole_limits = _convert_pair(hole_low, hole_high, unit)
    shaft_limits = _convert_pair(shaft_low, shaft_high, unit)
    clearance = _convert_pair(min_clear, max_clear, unit)
    overlap_out = _to_unit(overlap, unit)

    return {
        "hole_limits": (_maybe_scalarize(hole_limits[0]), _maybe_scalarize(hole_limits[1])),
        "shaft_limits": (_maybe_scalarize(shaft_limits[0]), _maybe_scalarize(shaft_limits[1])),
        "clearance": (_maybe_scalarize(clearance[0]), _maybe_scalarize(clearance[1])),
        "overlap": _maybe_scalarize(overlap_out),
        "fit_type": _maybe_scalarize(fit_type),
    }