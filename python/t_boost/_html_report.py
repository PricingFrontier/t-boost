"""Render a model report (`pricing_report`, version 2) as one self-contained HTML page.

`report_html` is a pure view of the report dict: it computes nothing that is not already in the
report, so a page can be rebuilt from the JSON logged next to the model. The page has no scripts
and loads nothing: charts are inline SVG (hover values from SVG ``<title>``), the tabs and the
dataset, class, factor and table selectors are CSS-only radio buttons, and detail sits in
``<details>``. It opens offline, in MLflow's artifact viewer (sandboxed iframe) and prints.

Layout: a sticky bar of tabs (Overview, Performance, Factors, Interactions, Tables, Build, Checks)
with one page-wide dataset switch. Every chart is drawn at the pixel width of the card that holds
it, so its text keeps its size and panels that share an x-axis line up. Colour roles: blue is
observed data, orange is the model (predicted values, relativities, tables); the selected
dataset is drawn in colour and the others in grey.

Every label, feature name, level and metadata value is HTML-escaped; element ids are built from
indices, never from labels.
"""

from __future__ import annotations

import bisect
import json
import math
import struct
from html import escape
from typing import Any, Callable, Iterable, Protocol, Sequence

_SUPPORTED_VERSIONS = (2,)

# Page geometry (CSS px). A chart is drawn at the inner width of its card (card width less 16 px
# padding and a 1 px border each side), so one SVG unit is one CSS pixel on a desktop screen.
_CONTENT = 1200
_GAP = 16
_INNER = 2 * 16 + 2
_W_FULL = _CONTENT - _INNER
_W_HALF = (_CONTENT - _GAP) // 2 - _INNER
_RAIL = 240
_W_FACTOR = _CONTENT - _RAIL - _GAP - _INNER
_LIST = 560
_H_ROW = 200  # chart height on the Overview: its three rows fit a 1440 x 900 screen
_FONT = 12.0
# Size budget: the Tables tab prints at most `_TABLE_CELLS` cells in `_TABLE_ROWS` tables (all
# of them are in the JSON report).
_TABLE_CELLS = 20_000
_TABLE_ROWS = 300
_RANKED_ROWS = 300
_INTERACTION_ROWS = 40
_OVERVIEW_ISSUES = 3
_TAB_NAMES = ("Overview", "Performance", "Factors", "Interactions", "Tables", "Build", "Checks")
_CHECKS_TAB = _TAB_NAMES.index("Checks")

_LIGHT = (
    "--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;"
    "--grid:#e7e6e0;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--wash:rgba(11,11,11,.045);"
    "--data:#2a78d6;--model:#eb6834;--model-soft:#f6b89e;--other:#c3c2b9;"
    "--band:rgba(42,120,214,.14);--massbar:#d6d5ce;--tol:rgba(11,11,11,.05);"
    "--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--good-ink:#006300;--link:#1c5cab;"
    "--i0:#f2a17c;--i1:#e3703c;--i2:#b44c18;--i3:#6f2c0b;--hi:#c6ecc3;--lo:#fbeba6"
)
_DARK = (
    "--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#8f8d86;"
    "--grid:#2c2c2a;--axis:#3d3d3a;--border:rgba(255,255,255,.10);--wash:rgba(255,255,255,.06);"
    "--data:#3987e5;--model:#d95926;--model-soft:#7a3a20;--other:#55544f;"
    "--band:rgba(57,135,229,.22);--massbar:#45443f;--tol:rgba(255,255,255,.06);"
    "--good-ink:#0ca30c;--link:#86b6ef;"
    "--i0:#b04c20;--i1:#d9662f;--i2:#f08f62;--i3:#f9c3a6;--hi:#2c4f30;--lo:#544a1b"
)

_CSS = """
:root{color-scheme:light;%(light)s}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;%(dark)s}}
:root[data-theme="dark"]{color-scheme:dark;%(dark)s}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.app{max-width:%(content)dpx;margin:0 auto;padding:0 16px 16px;box-sizing:content-box}
.top{padding:10px 0 0}
.top h1{font-size:20px;line-height:26px;margin:0;font-weight:650}
.top .sub{color:var(--ink2);font-size:13px;line-height:18px;margin:2px 0 0}
.bar{position:sticky;top:0;z-index:20;background:var(--page);display:flex;flex-wrap:wrap;
align-items:center;justify-content:space-between;gap:8px 16px;padding:8px 0;margin-bottom:12px;
border-bottom:1px solid var(--grid)}
.tabs{display:flex;flex-wrap:wrap;gap:2px}
.tabs label{display:block;padding:5px 12px;border:1px solid transparent;border-radius:7px;
cursor:pointer;color:var(--ink2);font-size:14px;line-height:20px;user-select:none}
.tabs label:hover{color:var(--ink);background:var(--wash)}
.switches{display:flex;flex-wrap:wrap;gap:8px}
.switch{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden;
background:var(--surface);font-size:13px;line-height:18px}
.switch .cap{padding:5px 10px;color:var(--muted)}
.switch label{padding:5px 12px;cursor:pointer;color:var(--ink2);border-left:1px solid var(--border);
user-select:none}
.switch label:hover{color:var(--ink)}
.sr{position:absolute;opacity:0;width:1px;height:1px;margin:0;pointer-events:none}
section.tab{display:flex;flex-direction:column;gap:%(gap)dpx}
.row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:%(gap)dpx;align-items:stretch}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
padding:12px 16px;min-width:0;display:flex;flex-direction:column}
.card h3{font-size:15px;line-height:20px;margin:0;font-weight:650}
.card .what{font-size:12.5px;line-height:17px;color:var(--ink2);margin:2px 0 8px}
.card.solo>h3,.card.solo>.head{margin-bottom:8px}
.card .head{display:flex;align-items:center;justify-content:space-between;gap:12px}
.card h4{font-size:13px;line-height:18px;margin:14px 0 6px;color:var(--ink2);font-weight:600}
.card>details,.card>div>details{margin-top:8px}
@media (min-width:981px){.card h3,.card .what{white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}}
svg.chart{display:block;max-width:100%%;height:auto;overflow:visible}
svg text{fill:var(--muted);font-size:%(font)dpx;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
svg .lbl{fill:var(--ink2)}
svg .strong{fill:var(--ink);font-weight:600}
svg .gl{stroke:var(--grid);stroke-width:1}
svg .ax{stroke:var(--axis);stroke-width:1}
svg .ref{stroke:var(--ink2);stroke-width:1;opacity:.55}
svg .guide{stroke:var(--axis);stroke-width:1;stroke-dasharray:4 3}
svg .data{stroke:var(--data);stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}
svg .model{stroke:var(--model);stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}
svg .faded{stroke:var(--model);stroke-width:2;fill:none;opacity:.3;stroke-linejoin:round;stroke-linecap:round}
svg .dfaded{stroke:var(--data);stroke-width:2;fill:none;opacity:.3;stroke-linejoin:round;stroke-linecap:round}
svg .ddot{fill:var(--data);stroke:var(--surface);stroke-width:1.5}
svg .mdot{fill:var(--model);stroke:var(--surface);stroke-width:1.5}
svg .dsl{stroke:var(--other);stroke-width:1.5;fill:none;stroke-linejoin:round}
svg .dsw{display:none;fill:var(--band);stroke:none}
svg .mb{fill:var(--massbar)}
svg .own,svg .keep{fill:var(--model)}
svg .via,svg .floor{fill:var(--model-soft)}
svg .drop{fill:var(--other)}
svg .mk{fill:var(--ink);stroke:var(--surface);stroke-width:2}
svg .tol{fill:var(--tol)}
svg .shapband{fill:var(--model);opacity:.14}
svg .ref1{stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 3;fill:none}
svg .ib0,svg .ib1,svg .ib2,svg .ib3{stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}
svg .ib0{stroke:var(--i0)}svg .ib1{stroke:var(--i1)}svg .ib2{stroke:var(--i2)}svg .ib3{stroke:var(--i3)}
svg .dash{stroke-dasharray:4 3}
svg .exl{stroke:var(--ink2);stroke-width:2;fill:none}
svg .acl{stroke:var(--ink2);stroke-width:2;stroke-dasharray:4 3;fill:none}
svg .ibw0{fill:var(--i0);opacity:.14}svg .ibw1{fill:var(--i1);opacity:.14}svg .ibw2{fill:var(--i2);opacity:.14}svg .ibw3{fill:var(--i3);opacity:.14}
svg .fade{opacity:.3}
svg .shapmean{stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 3;fill:none}
svg .hbar{fill:var(--model-soft)}
svg .flagdot{fill:var(--crit);stroke:var(--surface);stroke-width:1.5}
svg .neardot{fill:var(--warn);stroke:var(--surface);stroke-width:1.5}
svg .bandbox{fill:var(--band)}
svg .hit{fill:transparent}
svg .hit:hover{fill:var(--wash)}
table{border-collapse:collapse;width:100%%;font-size:13px;line-height:18px}
th,td{padding:5px 10px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:baseline}
th{color:var(--ink2);font-weight:600;white-space:nowrap}
th:first-child,td:first-child{padding-left:0}
th:last-child,td:last-child{padding-right:0}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
.tw{overflow-x:auto}
.tw.scroll{max-height:440px;overflow-y:auto}
.tw.short{max-height:250px;overflow-y:auto}
.tw.scroll thead th,.tw.short thead th{position:sticky;top:0;background:var(--surface);z-index:1}
.kv td:first-child{color:var(--ink2);width:32%%;padding-right:12px}
.tinted th:last-child,.tinted td:last-child{padding-right:8px}
td.g1{background:color-mix(in srgb,var(--hi) 20%%,transparent)}
td.g2{background:color-mix(in srgb,var(--hi) 40%%,transparent)}
td.g3{background:color-mix(in srgb,var(--hi) 60%%,transparent)}
td.g4{background:color-mix(in srgb,var(--hi) 80%%,transparent)}
td.g5{background:var(--hi)}
td.y1{background:color-mix(in srgb,var(--lo) 20%%,transparent)}
td.y2{background:color-mix(in srgb,var(--lo) 40%%,transparent)}
td.y3{background:color-mix(in srgb,var(--lo) 60%%,transparent)}
td.y4{background:color-mix(in srgb,var(--lo) 80%%,transparent)}
td.y5{background:var(--lo)}
details>summary{cursor:pointer;color:var(--ink2);font-size:13px;line-height:20px}
details[open]>summary{margin-bottom:6px}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:12px;line-height:18px;font-weight:600;
padding:0 8px 0 3px;border-radius:10px;border:1px solid var(--border);white-space:nowrap;color:var(--ink2)}
.badge i{font-style:normal;display:inline-block;width:14px;height:14px;border-radius:7px;color:#fff;
font-size:10px;line-height:14px;text-align:center}
.b-critical i{background:var(--crit)}.b-warning i{background:var(--warn);color:#0b0b0b}
.b-pass i{background:var(--good)}.b-info i{background:var(--muted)}
.counts{display:flex;gap:6px;flex-wrap:wrap}
.issues{list-style:none;margin:0;padding:0}
.issues li{display:grid;grid-template-columns:86px minmax(0,1fr);gap:10px;align-items:baseline;
padding:5px 0;border-top:1px solid var(--grid);font-size:13px;line-height:18px}
.issues li:first-child{border-top:0;padding-top:0}
.issues .badge{justify-self:start}
.issues .t{min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.link{color:var(--link);cursor:pointer;font-size:13px;line-height:18px}
.link:hover{text-decoration:underline}
.more{margin-top:auto;padding-top:6px}
.ok{color:var(--good-ink);font-size:13px;margin:4px 0}
.note{color:var(--ink2);font-size:13px;margin:4px 0}
.muted{color:var(--muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.fxgrid{display:grid;grid-template-columns:%(rail)dpx minmax(0,1fr);gap:%(gap)dpx;align-items:start}
.rail{position:sticky;top:64px;max-height:calc(100vh - 134px);overflow-y:auto;padding:6px}
.rail .rl{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:3px 8px;padding:7px 10px 8px;
border-radius:8px;cursor:pointer;user-select:none}
.rail .rl:hover{background:var(--wash)}
.rail .nm{font-size:13.5px;line-height:18px;font-weight:600;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.rail .pc{font-size:12px;line-height:18px;color:var(--ink2);font-variant-numeric:tabular-nums}
.rail .bars{grid-column:1/-1;display:flex;height:5px;border-radius:3px;overflow:hidden;
background:var(--grid)}
.rail .bars i{display:block;height:5px}
.rail .bars .o{background:var(--model)}.rail .bars .v{background:var(--model-soft)}
.rail .w{grid-column:1/-1;font-size:11.5px;line-height:15px;color:var(--muted);white-space:nowrap;
overflow:hidden;text-overflow:ellipsis}
.rail .cap{font-size:11.5px;line-height:15px;color:var(--muted);padding:12px 10px 4px}
.rail .unused .nm{color:var(--muted);font-weight:500}
.rail .pair .nm{white-space:normal;overflow-wrap:anywhere}
.ttgrid{display:grid;grid-template-columns:%(list)dpx minmax(0,1fr);gap:%(gap)dpx;align-items:start}
.tlist .tw{max-height:calc(100vh - 210px);overflow-y:auto}
.tlist .tw thead th{position:sticky;top:0;background:var(--surface);z-index:1}
.tlist td label{display:block;cursor:pointer}.tlist td:first-child{white-space:nowrap}
.tview{position:sticky;top:64px}
@media (max-width:980px){.row,.fxgrid,.ttgrid{grid-template-columns:minmax(0,1fr)}
.rail,.tview{position:static;max-height:none}.tlist .tw{max-height:420px}}
@media print{:root{%(light)s}body{background:#fff}.bar,.rail,.tlist{display:none!important}
section.tab{display:flex!important;break-before:page}.fxp,.tv{display:block!important}
.row,.fxgrid,.ttgrid{grid-template-columns:minmax(0,1fr)}
.card{break-inside:avoid}.tw.scroll,.tw.short,.tlist .tw{max-height:none;overflow:visible}}
"""

# --------------------------------------------------------------------------- formatting -------

def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _num(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return _e(v)
    if not math.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or (ax >= 1e3 and x == round(x)):
        return f"{x:,.0f}"
    if ax >= 1e3:
        return f"{x:,.1f}"
    if ax < 1e-4:
        return f"{x:.{digits - 1}e}"
    return f"{x:.{digits}g}"


def _pct(v: Any, digits: int = 1) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{x * 100:.{digits}f}%" if math.isfinite(x) else "—"


def _fixed(v: Any, digits: int = 3) -> str:
    """Fixed decimals, so 1 prints as 1.000 beside 0.993."""
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{x:.{digits}f}" if math.isfinite(x) else "—"


def _amt(v: Any) -> str:
    """An amount (exposure, expected count, support) with one decimal, so a column aligns."""
    if v is None:
        return "—"
    x = float(v)
    return f"{x:,.1f}" if math.isfinite(x) else "—"


def _column(values: Sequence[Any]) -> list[str]:
    """One format for a column of amounts: whole numbers when any passes 1,000, so the column
    aligns; ``_num`` otherwise."""
    big = any(v is not None and math.isfinite(float(v)) and abs(float(v)) >= 1000 for v in values)
    if not big:
        return [_num(v) for v in values]
    return ["—" if v is None or not math.isfinite(float(v)) else f"{float(v):,.0f}" for v in values]


def _mult(v: Any) -> str:
    """A multiplicative effect: ×0.80, ×1.96, ×12.3."""
    if v is None:
        return "—"
    x = float(v)
    if not math.isfinite(x):
        return "—"
    return f"×{x:.2f}" if x < 10 else f"×{x:.1f}"


def _signed(v: Any) -> str:
    """A z score with a true minus sign."""
    if v is None:
        return "—"
    x = float(v)
    return f"{x:+.1f}".replace("-", "−") if math.isfinite(x) else "—"


def _share_tick(t: float) -> str:
    return f"{t * 100:.0f}%" if t >= 0.01 else f"{t * 100:g}%"


def _border(b: float) -> str:
    x = float(b)
    return f"{x:.0f}" if abs(x) >= 1000 else f"{x:.4g}"


def integer_grid(borders: Sequence[float]) -> bool:
    """Every border a multiple of 0.5, as the cuts of an integer feature are (midpoints of
    adjacent values, or whole numbers between values further apart). Necessary, not sufficient:
    `run_label` also needs the report's ``integer`` flag on the axis."""
    return bool(borders) and all(abs(2 * b - round(2 * b)) < 1e-6 * max(1.0, abs(b))
                                 for b in borders)


def span_label(lo: float | None, hi: float | None, integer: bool) -> str:
    """The values in ``(lo, hi]`` (``None`` for no bound): ``51``, ``22–25``, ``≤ 18`` or
    ``≥ 90`` on an integer feature, else ``(a, b]``, ``≤ b`` or ``> a``."""
    if integer:
        a = None if lo is None else math.floor(round(2 * lo) / 2) + 1
        b = None if hi is None else math.floor(round(2 * hi) / 2)
        if a is None and b is not None:
            return f"≤ {b}"
        if b is None and a is not None:
            return f"≥ {a}"
        if a is not None and b is not None and a <= b:
            return f"{a}" if a == b else f"{a}–{b}"
    if lo is None:
        return "all" if hi is None else f"≤ {_border(hi)}"
    if hi is None:
        return f"> {_border(lo)}"
    return f"({_border(lo)}, {_border(hi)}]"


def run_label(axis: dict[str, Any], first: int, last: int) -> str:
    """The values in a run of numeric cells ``first..last`` of ``axis`` (cell 0 is missing),
    as integers when the axis is flagged ``integer`` (its feature took only whole values)."""
    borders = [float(b) for b in axis.get("borders") or []]
    if not borders:
        return f"cell {first}" if first == last else f"cells {first}–{last}"
    lo = borders[first - 2] if first >= 2 else None
    hi = borders[last - 1] if last - 1 < len(borders) else None
    return span_label(lo, hi, bool(axis.get("integer")) and integer_grid(borders))


def cell_labels(axis: dict[str, Any]) -> list[str]:
    """One label per cell: ``missing`` for cell 0, then the values the cell holds: integer
    ranges (``≤ 18``, ``21``, ``22–25``, ``≥ 90``) on an axis flagged ``integer``, ``≤ b``,
    ``(a, b]`` and ``> b`` on any other numeric one, the member levels on a categorical one."""
    n = int(axis.get("cells") or 0)
    labels = [""] * n
    if n:
        labels[0] = "missing"
    levels = axis.get("levels")
    if levels:
        groups: dict[int, list[str]] = {}
        for lv in levels:
            c = int(lv["cell"])
            name = str(lv["label"])
            if lv.get("synthetic") and lv.get("members"):
                name = f"{name} ({len(lv['members'])} levels)"
            groups.setdefault(c, []).append(name)
        for c in range(1, n):
            labels[c] = ", ".join(groups.get(c, [])) or f"cell {c}"
        return labels
    for c in range(1, n):
        labels[c] = run_label(axis, c, c)
    return labels


def _short(label: str, n: int = 14) -> str:
    return label if len(label) <= n else label[: max(n - 1, 1)] + "…"


def _text_w(s: str, size: float = _FONT) -> float:
    """Approximate rendered width of ``s`` in px (system-ui, digits and lower case)."""
    return len(s) * size * 0.56


# --------------------------------------------------------------------------- scales -----------

class _Scale(Protocol):
    ticks: list[float]

    def __call__(self, v: float) -> float: ...


_Fmt = Callable[[float], str]


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """Nice ticks inside ``[lo, hi]``."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo] if math.isfinite(lo) else [0.0]
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    out: list[float] = []
    t = math.ceil(lo / step - 1e-9) * step
    while t <= hi + 1e-9 * step and len(out) < 50:
        out.append(round(t, 12))
        t += step
    return out


class _Lin:
    """Linear scale ``[lo, hi] -> [a, b]`` padded by ``pad`` of the span; ``zero`` floors
    non-negative data at 0."""

    def __init__(self, lo: float, hi: float, a: float, b: float, *, pad: float = 0.06,
                 zero: bool = False, n: int = 4) -> None:
        if not (math.isfinite(lo) and math.isfinite(hi)):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            lo, hi = lo - (abs(lo) * 0.1 or 0.5), hi + (abs(hi) * 0.1 or 0.5)
        span = hi - lo
        lo2, hi2 = lo - span * pad, hi + span * pad
        if zero and lo >= 0:
            lo2 = 0.0
        self.lo, self.hi, self.a, self.b = lo2, hi2, a, b
        self.ticks = _ticks(lo2, hi2, n)

    def __call__(self, v: float) -> float:
        return self.a + (v - self.lo) / (self.hi - self.lo) * (self.b - self.a)


# Tick sets for a log axis of multipliers, from fine to coarse; the finest set that puts no more
# than `max_ticks` ticks on the axis is used.
_LOG_TICKS = (
    (0.8, 0.9, 1.0, 1.1, 1.25),
    (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0),
    (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0),
    (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
    (0.01, 0.1, 1.0, 10.0, 100.0),
    (1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4),
)


class _Log:
    """Log scale for multipliers. The axis spans at least ``span`` either side of 1, so a small
    effect looks small."""

    def __init__(self, lo: float, hi: float, a: float, b: float, *, span: float = 1.12,
                 max_ticks: int = 5) -> None:
        lo = min(max(lo, 1e-9), 1 / span)
        hi = max(hi, span)
        llo, lhi = math.log(lo), math.log(hi)
        pad = (lhi - llo) * 0.05
        self.llo, self.lhi, self.a, self.b = llo - pad, lhi + pad, a, b
        # The finest tick set that spans at least 40% of the axis in at most `max_ticks` ticks;
        # failing that, the finest one thinned (every k-th tick from 1) to fit.
        whole: list[list[float]] = []
        thinned: list[list[float]] = []
        for tick_set in _LOG_TICKS:
            inside = [t for t in tick_set if self.llo <= math.log(t) <= self.lhi]
            fits = len(inside) <= max_ticks
            if not fits:
                step = math.ceil(len(inside) / max_ticks)
                at = inside.index(1.0) if 1.0 in inside else 0
                inside = [t for i, t in enumerate(inside) if (i - at) % step == 0]
            if len(inside) >= 2 and math.log(inside[-1] / inside[0]) >= 0.4 * (self.lhi - self.llo):
                (whole if fits else thinned).append(inside)
        self.ticks = (whole or thinned or [[1.0]])[0]

    def __call__(self, v: float) -> float:
        lv = math.log(max(v, 1e-12))
        return self.a + (lv - self.llo) / (self.lhi - self.llo) * (self.b - self.a)


def _fv(values: Iterable[Any]) -> list[float]:
    out = []
    for v in values:
        if v is None:
            continue
        x = float(v)
        if math.isfinite(x):
            out.append(x)
    return out


def _mult_tick(t: float) -> str:
    return f"×{t:g}"


def _fixed2(t: float) -> str:
    return _fixed(t, 2)


def _pct0(t: float) -> str:
    return _pct(t, 0)


# --------------------------------------------------------------------------- SVG --------------

def _d(pts: Sequence[tuple[float, float]]) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)


class _Svg:
    """An SVG drawn in CSS pixels."""

    def __init__(self, w: float, h: float, label: str) -> None:
        self.w, self.h, self.label = w, h, label
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x: float, y: float, s: str, *, anchor: str = "start", cls: str = "") -> None:
        a = "" if anchor == "start" else f' text-anchor="{anchor}"'
        c = f' class="{cls}"' if cls else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}"{a}{c}>{_e(s)}</text>')

    def hline(self, x1: float, x2: float, y: float, cls: str) -> None:
        self.add(f'<line class="{cls}" x1="{x1:.1f}" x2="{x2:.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')

    def vline(self, x: float, y1: float, y2: float, cls: str) -> None:
        self.add(f'<line class="{cls}" x1="{x:.1f}" x2="{x:.1f}" y1="{y1:.1f}" y2="{y2:.1f}"/>')

    def path(self, pts: Sequence[tuple[float, float]], cls: str) -> None:
        if len(pts) > 1:
            self.add(f'<path class="{cls}" d="{_d(pts)}"/>')

    def hit(self, x: float, y: float, w: float, h: float, title: str) -> None:
        self.add(f'<rect class="hit" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}">'
                 f"<title>{_e(title)}</title></rect>")

    def group(self, cls: str, parts: list[str]) -> None:
        if parts:
            self.add(f'<g class="{cls}">' + "".join(parts) + "</g>" if cls else "".join(parts))

    def render(self) -> str:
        return (f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" width="{self.w:.0f}" '
                f'height="{self.h:.0f}" viewBox="0 0 {self.w:.0f} {self.h:.0f}" role="img" '
                f'aria-label="{_e(self.label)}">' + "".join(self.parts) + "</svg>")


def _yaxis(svg: _Svg, scale: _Scale, x0: float, x1: float, fmt: _Fmt) -> None:
    """Gridlines, and tick labels right-aligned 6 px left of the plot."""
    for t in scale.ticks:
        y = scale(t)
        svg.hline(x0, x1, y, "gl")
        svg.text(x0 - 6, y + 4, fmt(t), anchor="end")


def _label_w(scale: _Scale, fmt: _Fmt) -> float:
    return max((_text_w(fmt(t)) for t in scale.ticks), default=20.0)


def _legend(svg: _Svg, x: float, y: float, items: Sequence[tuple[str, str, str]]) -> float:
    """``(kind, class, label)`` items left to right from ``x``; ``kind`` is line, dot or box.
    Returns where the legend ends."""
    for kind, cls, label in items:
        if kind == "line":
            svg.add(f'<line class="{cls}" x1="{x:.1f}" x2="{x + 16:.1f}" y1="{y - 4:.1f}" '
                    f'y2="{y - 4:.1f}"/>')
        elif kind == "dot":
            svg.add(f'<circle class="{cls}" cx="{x + 8:.1f}" cy="{y - 4:.1f}" r="4"/>')
        else:
            svg.add(f'<rect class="{cls}" x="{x + 3:.1f}" y="{y - 9:.1f}" width="10" height="10" '
                    'rx="2"/>')
        svg.text(x + 22, y, label, cls="lbl")
        x += 22 + _text_w(label) + 18
    return x


def _others(scored: list[dict[str, Any]], i: int) -> str:
    """The labels of every dataset but the ``i``-th."""
    return ", ".join(str(d["label"]) for j, d in enumerate(scored) if j != i)


def _ds_legend(svg: _Svg, x: float, y: float, n: int,
               items_for: Callable[[int], list[tuple[str, str, str]]], *,
               fit: float | None = None) -> None:
    """A legend for each of ``n`` datasets that names it and the others, shown with it by the
    page's dataset switch. With ``fit``, trailing items that would overflow that width go."""
    for i in range(n):
        items = items_for(i)
        while fit is not None and len(items) > 1 and (
                sum(40 + _text_w(label) for _, _, label in items) - 18 > fit):
            items.pop()
        g = _Svg(svg.w, svg.h, "")
        _legend(g, x, y, items)
        svg.group(f"ds ds-{i}", g.parts)


def _bar(cls: str, x: float, y_top: float, w: float, y_base: float, r: float = 2.0,
         title: str = "") -> str:
    """A column from the baseline up to ``y_top``, rounded at the data end only."""
    h = y_base - y_top
    if h <= 0.2 or w <= 0:
        return ""
    r = min(r, w / 2, h)
    d = (f"M{x:.1f},{y_base:.1f}V{y_top + r:.1f}Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f}"
         f"H{x + w - r:.1f}Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f}V{y_base:.1f}Z")
    t = f"<title>{_e(title)}</title>" if title else ""
    return f'<path class="{cls}" d="{d}">{t}</path>' if t else f'<path class="{cls}" d="{d}"/>'


def _segments(svg: _Svg, pts: Sequence[tuple[float, float]], faded: Sequence[bool],
              cls: str = "model", faded_cls: str = "faded") -> None:
    """A joined line whose segments touching a faded point are drawn ``faded_cls``."""
    if len(pts) == 1:
        svg.add(f'<circle class="mdot" cx="{pts[0][0]:.1f}" cy="{pts[0][1]:.1f}" r="4"/>')
        return
    run: list[tuple[float, float]] = [pts[0]] if pts else []
    state: bool | None = None
    for j in range(1, len(pts)):
        f = faded[j - 1] or faded[j]
        if state is None or f == state:
            run.append(pts[j])
        else:
            svg.path(run, faded_cls if state else cls)
            run = [pts[j - 1], pts[j]]
        state = f
    svg.path(run, faded_cls if state else cls)


def _xlabels(svg: _Svg, xs: Sequence[float], labels: Sequence[str], y: float, slot: float,
             *, max_chars: int = 16) -> None:
    """Category labels under a plot at a regular step, never overlapping; the last always."""
    n = len(xs)
    if n == 0:
        return
    texts = [_short(s, max_chars) for s in labels]
    widths = [_text_w(t) for t in texts]
    step = max(1, math.ceil((max(widths) + 10) / max(slot, 1e-6)))
    cand = list(range(0, n, step))
    if cand[-1] != n - 1:
        cand.append(n - 1)

    def clear(i: int, j: int) -> bool:
        return xs[j] - widths[j] / 2 >= xs[i] + widths[i] / 2 + 8

    placed: list[int] = []
    for i in cand:
        if i == n - 1:
            while placed and not clear(placed[-1], i):
                placed.pop()
            placed.append(i)
        elif not placed or clear(placed[-1], i):
            placed.append(i)
    for i in placed:
        svg.text(xs[i], y, texts[i], anchor="middle")


def _none(text: str = "No data.") -> str:
    return f'<p class="muted">{_e(text)}</p>'


# --------------------------------------------------------------------------- report access ----

def _scored(r: dict[str, Any]) -> list[dict[str, Any]]:
    return [d for d in r["datasets"] if not d.get("error")]


def _rate_name(r: dict[str, Any]) -> str:
    link = r["model"]["link"]
    if link == "log":
        return "rate per unit exposure"
    return "probability" if link in ("logit", "softmax") else "mean response"


def _actual_name(r: dict[str, Any]) -> str:
    obj = str(r["model"].get("objective") or "")
    if obj == "poisson":
        return "claims"
    return "events" if r["model"]["link"] == "logit" else "actual"


def _base_rate(r: dict[str, Any]) -> float | None:
    banks = r["structure"]["banks"]
    if r["model"]["link"] != "log" or not banks or banks[0].get("f0") is None:
        return None
    return math.exp(float(banks[0]["f0"]))


def _banks(r: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return r["tables"] if r["model"]["multiclass"] else {"": r["tables"]}


def _classes(r: dict[str, Any]) -> list[str]:
    return [str(c) for c in r["model"]["classes"]] if r["model"]["multiclass"] else [""]


def _one_way(r: dict[str, Any], cls: str, feature: str) -> dict[str, Any] | None:
    bank = _banks(r).get(cls) or {}
    return next((t for t in bank.get("tables", []) if t["feature_names"] == [feature]), None)


def _effects(t: dict[str, Any], link: str) -> tuple[list[float], bool, str]:
    """A table's effect per cell, whether it multiplies, and its name."""
    if link == "log" and t.get("relativities"):
        return [float(v) for v in t["relativities"]], True, "relativity"
    if link == "logit":
        return [math.exp(float(v)) for v in t["values"]], True, "odds multiplier"
    if link == "softmax":
        return [float(v) for v in t["values"]], False, "logit contribution"
    return [float(v) for v in t["values"]], False, "effect"


def _thin(support: Sequence[float], share: float) -> set[int]:
    """Cells holding exposure but less than ``share`` of the table's total."""
    total = sum(support)
    return {i for i, x in enumerate(support) if total > 0 and 0 < x < share * total}


def _ae_entry(d: dict[str, Any], feature: str, cls: str | None) -> dict[str, Any] | None:
    return next((e for e in d["actual_vs_expected"] if e["feature"] == feature
                 and (cls is None or str(e.get("class")) == cls)), None)


def _metrics(d: dict[str, Any], k: int | None) -> dict[str, Any]:
    m: dict[str, Any] = d["metrics"]
    return m if k is None else m["per_class"][k]


def _lift(d: dict[str, Any], k: int | None) -> list[dict[str, Any]]:
    return list(_metrics(d, k).get("lift") or [])


def _f32(x: Any) -> float:
    """``x`` rounded to the nearest float32."""
    return float(struct.unpack("f", struct.pack("f", float(x)))[0])


def _table_cells(axis: dict[str, Any], taxis: dict[str, Any]) -> list[int | None]:
    """For each cell of a feature's merged grid, the cell of a table axis that holds it."""
    n = int(axis.get("cells") or 0)
    out: list[int | None] = [None] * n
    if n:
        out[0] = 0
    levels, tlevels = axis.get("levels"), taxis.get("levels")
    if levels and tlevels:
        by_label = {str(lv["label"]): int(lv["cell"]) for lv in tlevels}
        for lv in levels:
            tc = by_label.get(str(lv["label"]))
            if tc is not None:
                out[int(lv["cell"])] = tc
        dc, tdc = axis.get("default_cell"), taxis.get("default_cell")
        if dc is not None and tdc is not None and int(dc) < n and out[int(dc)] is None:
            out[int(dc)] = int(tdc)
        return out
    # Both lists hold the same float32 borders (a table's are a subset of the merged grid's),
    # but `merged_axes` writes them in float32's shortest form and `tables` in float64, so they
    # are compared once rounded back to float32: the cell up to border b is then the table cell
    # after the borders below b, however close two borders are.
    mb = [_f32(b) for b in axis.get("borders") or []]
    tb = [_f32(b) for b in taxis.get("borders") or []]
    for c in range(1, n):
        out[c] = 1 + (bisect.bisect_left(tb, mb[c - 1]) if c - 1 < len(mb) else len(tb))
    return out


def _path_data(r: dict[str, Any]) -> dict[str, Any] | None:
    """The ranked path, if the fit recorded one: ``bank`` is the table count per step."""
    fit = r["fit"]["pruning"]
    rep = fit["value"] if fit["available"] else None
    if not rep:
        return None
    path = next((c for c in (rep.get("path"), rep.get("selection"))
                 if isinstance(c, dict) and c.get("sizes") and c.get("oob_deviance")), None)
    if path is None:
        return None
    scores = {tuple(s["u"]): s for s in rep.get("table_scores") or []}
    mains = sorted(u for u, s in scores.items() if s.get("sticky"))
    total = sum(max(_share(s), 0.0) for s in scores.values()) or 1.0
    devs = path["oob_deviance"]
    gain = path.get("total_gain") or (devs[0] - min(devs))
    best = int(path.get("best", len(devs) - 1))
    return {"rep": rep, "path": path, "scores": scores, "mains": mains, "total": total,
            "bank": [len(mains) + int(s) for s in path["sizes"]], "best": best, "gain": gain,
            "share": [((devs[0] - d) / gain) if gain else 0.0 for d in devs]}


def _share(score: dict[str, Any]) -> float:
    """A candidate table's variance (a share once divided by all the candidates' total)."""
    for key in ("variance_share", "variance", "mean_gain"):
        if score.get(key) is not None:
            return float(score[key])
    return 0.0


def _feature_names(r: dict[str, Any], u: Sequence[int]) -> str:
    names = r["model"]["native_feature_order"]
    return " × ".join(names[i] if 0 <= i < len(names) else str(i) for i in u)


# --------------------------------------------------------------------------- charts -----------

def calibration_chart(r: dict[str, Any], k: int | None, w: float, h: float) -> str:
    """Deciles of predicted rate: actual and predicted rate (top; the selected dataset) and A/E
    for every dataset (bottom; the selected one in colour with its ±2σ band)."""
    scored = _scored(r)
    idx = list(range(len(scored)))
    lifts = {i: _lift(scored[i], k) for i in idx}
    n = max((len(v) for v in lifts.values()), default=0)
    if n == 0:
        return _none()
    rates = _fv(b.get(key) for v in lifts.values() for b in v for key in ("actual", "predicted"))
    aes = _fv(b.get("ae") for v in lifts.values() for b in v)
    two_sd = min(2 * max(_fv(b.get("sd") for v in lifts.values() for b in v), default=0.0), 0.25)
    t0, gap, foot = 24.0, 28.0, 22.0
    usable = h - t0 - gap - foot
    top_h = round(usable * 0.56)
    b0 = t0 + top_h + gap
    b1 = h - foot
    ts = _Lin(0.0, max(rates or [1.0]), t0 + top_h, t0, zero=True, n=4)
    bs = _Lin(min(aes + [1 - two_sd, 0.95]), max(aes + [1 + two_sd, 1.05]), b1, b0, pad=0.06, n=3)

    def rate_fmt(v: float) -> str:
        return _num(v, 3)

    left = 6 + max(_label_w(ts, rate_fmt), _label_w(bs, _fixed2)) + 6
    right = w - 46
    bw = (right - left) / n
    xs = [left + (j + 0.5) * bw for j in range(n)]
    svg = _Svg(w, h, "calibration by decile of predicted rate")
    def items(i: int) -> list[tuple[str, str, str]]:
        label = str(scored[i]["label"])
        out = [("line", "data", f"actual ({label})"), ("line", "model", f"predicted ({label})")]
        if len(scored) > 1:
            out.append(("line", "dsl", f"A/E ({_others(scored, i)})"))
        return out

    _ds_legend(svg, left, 12, len(scored), items)
    _yaxis(svg, ts, left, right, rate_fmt)
    _yaxis(svg, bs, left, right, _fixed2)
    svg.hline(left, right, bs(1.0), "ref")
    svg.hline(left, right, t0 + top_h, "ax")
    svg.text(left, b0 - 8, "A/E by decile", cls="lbl")

    def clip(v: float) -> float:
        return bs(min(max(v, bs.lo), bs.hi))

    for i in idx:
        lift = lifts[i]
        tag = f"ds ds-{i}"
        g = _Svg(w, h, "")
        pre = [(xs[j], ts(b["predicted"])) for j, b in enumerate(lift) if b.get("predicted") is not None]
        act = [(xs[j], ts(b["actual"])) for j, b in enumerate(lift) if b.get("actual") is not None]
        g.path(pre, "model")
        g.path(act, "data")
        for x, y in act:
            g.add(f'<circle class="ddot" cx="{x:.1f}" cy="{y:.1f}" r="3"/>')
        if act:
            g.text(act[-1][0] + 8, act[-1][1] + 4, _num(lift[-1].get("actual"), 3), cls="lbl")
        svg.group(tag, g.parts)
        if any(b.get("sd") for b in lift):
            up = [(xs[j], clip(1 + 2 * (b.get("sd") or 0.0))) for j, b in enumerate(lift)]
            dn = [(xs[j], clip(1 - 2 * (b.get("sd") or 0.0))) for j, b in enumerate(lift)]
            svg.add(f'<path class="dsw dsw-{i}" d="{_d(up + dn[::-1])}Z"/>')
    for i in idx:
        pts = [(xs[j], clip(b["ae"])) for j, b in enumerate(lifts[i]) if b.get("ae") is not None]
        svg.path(pts, f"dsl dsl-{i}")
    svg.hline(left, right, b1, "ax")
    for j in range(n):
        svg.text(xs[j], b1 + 16, str(j + 1), anchor="middle")
    svg.text(right + 8, b1 + 16, "decile", cls="lbl")
    for j in range(n):
        lines = [f"decile {j + 1}"]
        for i in idx:
            if j < len(lifts[i]):
                b = lifts[i][j]
                lines.append(f"{scored[i]['label']}: actual {_num(b.get('actual'), 4)}, predicted "
                             f"{_num(b.get('predicted'), 4)}, A/E {_fixed(b.get('ae'))}, "
                             f"z {_signed(b.get('z'))}")
        svg.hit(left + j * bw, t0, bw, b1 - t0, "\n".join(lines))
    return svg.render()


def importance_chart(r: dict[str, Any], w: float, h: float | None = None, *, top: int = 9,
                     row_h: float = 22.0) -> str:
    """Feature importance as bars: the feature's own one-way table, then its share through
    interactions; the tail folded into one row."""
    feats = [f for f in r.get("features") or [] if f.get("used")]
    rows = feats[:top]
    rest = feats[top:]
    if rest:
        rows = rows + [{"feature": f"{len(rest)} more",
                        "own_share": sum(f["own_share"] for f in rest),
                        "interaction_share": sum(f["interaction_share"] for f in rest),
                        "importance": sum(f["importance"] for f in rest)}]
    if not rows:
        return _none("No deployed tables.")
    head = 24.0
    if h is None:
        h = head + row_h * len(rows)
    row_h = (h - head) / len(rows)
    name_w = min(max(_text_w(str(f["feature"])) for f in rows) + 14, w * 0.3)
    x0, x1 = name_w, w - 52
    peak = max(f["importance"] for f in rows) or 1.0
    svg = _Svg(w, h, "feature importance")
    _legend(svg, x0, 12, [("box", "own", "own one-way table"), ("box", "via", "through interactions")])
    bh = min(14.0, row_h - 6)
    for j, f in enumerate(rows):
        yc = head + (j + 0.5) * row_h
        own = max(f["own_share"], 0.0) / peak * (x1 - x0)
        via = max(f["interaction_share"], 0.0) / peak * (x1 - x0)
        chars = int((name_w - 10) / (_FONT * 0.56))
        svg.text(x0 - 10, yc + 4, _short(str(f["feature"]), chars), anchor="end", cls="lbl")
        if own > 0.4:
            svg.add(f'<rect class="own" x="{x0:.1f}" y="{yc - bh / 2:.1f}" width="{own:.1f}" '
                    f'height="{bh:.1f}" rx="2"/>')
        if via > 0.4:
            gap = 2.0 if own > 0.4 else 0.0
            svg.add(f'<rect class="via" x="{x0 + own + gap:.1f}" y="{yc - bh / 2:.1f}" '
                    f'width="{max(via - gap, 0.6):.1f}" height="{bh:.1f}" rx="2"/>')
        svg.text(x0 + own + via + 6, yc + 4, _pct(f["importance"], 1), cls="lbl")
        tip = (f"{f['feature']}: {_pct(f['importance'], 2)} of the model variance, "
               f"{_pct(f['own_share'], 2)} in its own one-way table and "
               f"{_pct(f['interaction_share'], 2)} through interactions")
        svg.hit(0, yc - row_h / 2, w, row_h, tip)
    return svg.render()


def multiples_chart(r: dict[str, Any], w: float, h: float, *, cls: str = "",
                    count: int = 4) -> str:
    """The one-way tables of the most important features side by side on one shared scale (log
    for a multiplicative link), thin cells faded: the shape of the main effects at a glance."""
    link = r["model"]["link"]
    thin_share = r["thresholds"].get("thin_cell_mass") or 0.0
    panels = []
    for f in r.get("features") or []:
        t = _one_way(r, cls, f["feature"])
        if t is None:
            continue
        eff, mult, _ = _effects(t, link)
        sup = [float(s) for s in t.get("support") or [1.0] * len(eff)]
        cells = [i for i in range(len(eff)) if sup[i] > 0]
        if not cells:
            continue
        if t["axes"][0].get("levels"):
            cells.sort(key=lambda i: eff[i])
        panels.append((str(f["feature"]), eff, cells, cell_labels(t["axes"][0]), _thin(sup, thin_share),
                       mult))
        if len(panels) == count:
            break
    if not panels:
        return _none("No one-way tables.")
    mult = panels[0][5]
    vals = [p[1][i] for p in panels for i in p[2]]
    top, bottom = 44.0, h - 20
    ys: _Scale = (_Log(min(vals), max(vals), bottom, top, span=1.25, max_ticks=4) if mult
                  else _Lin(min(vals), max(vals), bottom, top, n=3))

    def fmt(t: float) -> str:
        return _mult_tick(t) if mult else _num(t, 2)

    lead = _label_w(ys, fmt) + 10
    gap = 18.0
    pw = (w - lead - gap * (len(panels) - 1)) / len(panels)
    svg = _Svg(w, h, "largest one-way tables")
    for p, (name, eff, cells, labels, thin, _) in enumerate(panels):
        left = lead + p * (pw + gap)
        right = left + pw
        for tk in ys.ticks:
            svg.hline(left, right, ys(tk), "gl")
            if p == 0:
                svg.text(left - 6, ys(tk) + 4, fmt(tk), anchor="end")
        svg.hline(left, right, ys(1.0 if mult else 0.0), "ref")
        n = len(cells)
        slot = (right - left) / n
        xs = [left + (j + 0.5) * slot for j in range(n)]
        pts = [(xs[j], ys(eff[c])) for j, c in enumerate(cells)]
        _segments(svg, pts, [c in thin for c in cells])
        live = [eff[c] for c in cells if c not in thin] or [eff[c] for c in cells]
        span = (f"{_mult(min(live))} to {_mult(max(live))}" if mult
                else f"{_num(min(live), 3)} to {_num(max(live), 3)}")
        chars = int(pw / (_FONT * 0.6))
        svg.text(left, 13, _short(name, chars), cls="strong")
        svg.text(left, 29, span, cls="lbl")
        half = max(int((pw / 2 - 6) / (_FONT * 0.56)), 3)
        svg.text(left, bottom + 15, _short(labels[cells[0]], half))
        if n > 1:
            svg.text(right, bottom + 15, _short(labels[cells[-1]], half), anchor="end")
        for j, c in enumerate(cells):
            v = _mult(eff[c]) if mult else _num(eff[c], 4)
            svg.hit(xs[j] - slot / 2, top, slot, bottom - top,
                    f"{name} {labels[c]}: {v}" + (" (thin cell)" if c in thin else ""))
    return svg.render()


def gain_chart(r: dict[str, Any], w: float, h: float) -> str:
    """Share of the out-of-bag deviance gain as the ranked tables join the bank: the fraction
    line, the cutoff and what the variance floor left."""
    pd = _path_data(r)
    if pd is None:
        return _none("No pruning path recorded.")
    path, bank, share, best = pd["path"], pd["bank"], pd["share"], pd["best"]
    devs = path["oob_deviance"]
    frac = path.get("fraction")
    left, right, top, bottom = 46.0, w - 14, 24.0, h - 22
    xs = _Lin(0.0, float(max(bank)), left, right, pad=0.0, n=5)
    ys = _Lin(min(min(share), 0.0), 1.0, bottom, top, pad=0.03, n=4)
    svg = _Svg(w, h, "share of the out-of-bag gain by bank size")
    _yaxis(svg, ys, left, right, _pct0)
    svg.hline(left, right, bottom, "ax")
    for j, t in enumerate(xs.ticks):
        last = j == len(xs.ticks) - 1
        svg.text(xs(t), bottom + 16, f"{t:.0f} tables" if last else f"{t:.0f}",
                 anchor="end" if last and xs(t) > right - 30 else "middle")
    svg.text(left, 12, "share of the out-of-bag deviance gain", cls="lbl")
    if frac is not None and frac < 1:
        svg.hline(left, right, ys(frac), "guide")
        svg.text(right, ys(frac) + 15, f"target: {_pct(frac, 1)} of the gain", anchor="end")
    xb = xs(bank[best])
    svg.vline(xb, top, bottom, "ref")
    svg.path([(xs(b), ys(s)) for b, s in zip(bank, share)], "model")
    svg.add(f'<circle class="mk" cx="{xb:.1f}" cy="{ys(share[best]):.1f}" r="5"/>')
    flip = xb > left + (right - left) * 0.75
    svg.text(xb - 6 if flip else xb + 6, bottom - 8, f"cutoff: {bank[best]} tables",
             anchor="end" if flip else "start", cls="strong")
    for b, s, dv in zip(bank, share, devs):
        svg.add(f'<circle class="hit" cx="{xs(b):.1f}" cy="{ys(s):.1f}" r="7"><title>'
                f"{b} tables: {_pct(s, 1)} of the gain (out-of-bag deviance {_num(dv, 6)})"
                "</title></circle>")
    return svg.render()


def _path_order(pd: dict[str, Any]) -> list[tuple[tuple[int, ...], int]]:
    """``(table, step)`` for every candidate in the order the path adds it (step 0: a sticky
    main effect). Without a recorded sequence: the kept tables, then the rest, each by share
    (step ``best`` for kept, ``best + 1`` for the rest)."""
    scores, best = pd["scores"], pd["best"]

    def by_share(us: Iterable[tuple[int, ...]]) -> list[tuple[int, ...]]:
        return sorted(us, key=lambda u: -_share(scores.get(u, {})))

    order = [(u, 0) for u in by_share(pd["mains"])]
    seq = pd["path"].get("sequence")
    if seq:
        return order + [(tuple(s["u"]), int(s["step"])) for s in seq]
    kept = {tuple(u) for u in pd["rep"].get("kept") or []} - set(pd["mains"])
    rest = set(scores) - kept - set(pd["mains"])
    return order + [(u, best) for u in by_share(kept)] + [(u, best + 1) for u in by_share(rest)]


def bank_chart(r: dict[str, Any], w: float, h: float) -> str:
    """Every candidate table in the order the path adds it, as a column of its share of the
    candidates' variance (log scale): kept, removed by the variance floor, or past the cutoff."""
    pd = _path_data(r)
    if pd is None:
        return _none("No pruning path recorded.")
    path, scores, best, total = pd["path"], pd["scores"], pd["best"], pd["total"]
    order = _path_order(pd)
    if not order:
        return _none("No candidate tables.")
    low = {tuple(u) for u in path.get("dropped_low_share") or []}
    floor = path.get("min_variance_share") or 0.0
    shares = [max(_share(scores.get(u, {})) / total, 1e-7) for u, _ in order]
    lo_e = math.floor(math.log10(min(shares + ([floor] if floor else []))))
    hi_e = math.ceil(math.log10(max(shares)))
    lo_e = max(lo_e, hi_e - 6)
    left, right, top, bottom = 50.0, w - 14, 24.0, h - 22

    def ys(v: float) -> float:
        lv = min(max(math.log10(v), lo_e), hi_e)
        return bottom - (lv - lo_e) / max(hi_e - lo_e, 1) * (bottom - top)

    svg = _Svg(w, h, "candidate tables in path order")
    for e in range(lo_e, hi_e + 1):
        svg.hline(left, right, ys(10.0 ** e), "gl")
        svg.text(left - 6, ys(10.0 ** e) + 4, _share_tick(10.0 ** e), anchor="end")
    n = len(order)
    bw = (right - left) / n
    cut = 0
    for j, ((u, step), sh) in enumerate(zip(order, shares)):
        kept = step <= best
        if kept:
            cut = j + 1
        cls = "floor" if u in low else ("keep" if kept else "drop")
        what = ("removed by the variance floor" if u in low
                else ("kept" if kept else "past the cutoff"))
        svg.add(_bar(cls, left + j * bw + 0.5, ys(sh), max(bw - 1, 0.6), bottom, 1.5,
                     f"{j + 1}. {_feature_names(r, u)}: {_pct(sh, 3)} of the candidates' "
                     f"variance, {what}"))
    if floor:
        svg.hline(left, right, ys(floor), "guide")
        svg.text(right, ys(floor) - 5, f"{_pct(floor, 1)} floor", anchor="end")
    xc = left + cut * bw
    svg.vline(xc, top, bottom, "ref")
    svg.text(xc + 6, top + 12, f"cutoff after {cut}", cls="strong")
    svg.hline(left, right, bottom, "ax")
    svg.text(left, bottom + 16, "1")
    svg.text(right, bottom + 16, f"{n} tables", anchor="end")
    _legend(svg, left, 12, [("box", "keep", "kept"), ("box", "floor", "below the floor"),
                            ("box", "drop", "past the cutoff")])
    return svg.render()


def training_chart(r: dict[str, Any], w: float, h: float) -> str:
    """Mean deviance on the early-stopping rows by boosting round, the band of rounds where the
    bags stopped and the minimum."""
    fit = r["fit"]
    hist = fit["evals_result"]
    if not hist["available"]:
        return f'<p class="note">Training curve: {_e(hist["reason"])}.</p>'
    per_bag = (hist["value"].get("eval") or {}).get("deviance") or []
    label = "early-stopping rows"
    if not per_bag:
        per_bag = (hist["value"].get("train") or {}).get("deviance") or []
        label = "train rows"
    if not per_bag:
        return _none("No training history.")
    rounds = max(len(b) for b in per_bag)
    mean: list[float | None] = []
    for i in range(rounds):
        vals = [b[i] for b in per_bag if i < len(b) and b[i] is not None]
        mean.append(sum(vals) / len(vals) if vals else None)
    pts = [(i + 1, v) for i, v in enumerate(mean) if v is not None]
    best_round, best = min(pts, key=lambda t: t[1])
    tail = [v for i, v in pts if i >= max(5, len(pts) // 20)] or [best]
    left, right, top, bottom = 58.0, w - 14, 24.0, h - 22
    xs = _Lin(1.0, float(rounds), left, right, pad=0.0, n=5)
    ys = _Lin(min(tail), max(tail), bottom, top, pad=0.12, n=4)

    def dev_fmt(v: float) -> str:
        return _num(v, 4)

    svg = _Svg(w, h, "deviance by boosting round")
    kept = fit["n_trees_per_bag"]["value"] or []
    if kept:
        x0, x1 = xs(min(kept)), xs(max(kept))
        svg.add(f'<rect class="tol" x="{x0:.1f}" y="{top:.1f}" width="{max(x1 - x0, 1):.1f}" '
                f'height="{bottom - top:.1f}"/>')
        svg.text((x0 + x1) / 2, top + 14, "bags stop", anchor="middle")
    _yaxis(svg, ys, left, right, dev_fmt)
    svg.hline(left, right, bottom, "ax")
    for j, t in enumerate(xs.ticks):
        last = j == len(xs.ticks) - 1
        svg.text(xs(t), bottom + 16, f"{t:,.0f} rounds" if last else f"{t:,.0f}",
                 anchor="end" if last and xs(t) > right - 40 else "middle")
    svg.text(left, 12, f"mean deviance on the {label}, over the bags still running", cls="lbl")
    svg.path([(xs(i), ys(min(max(v, ys.lo), ys.hi))) for i, v in pts], "data")
    xb, yb = xs(best_round), ys(best)
    svg.vline(xb, yb - 6, yb - 30, "ref")
    svg.add(f'<circle class="mk" cx="{xb:.1f}" cy="{yb:.1f}" r="4"/>')
    svg.text(xb, yb - 36, f"minimum at round {best_round:,}", anchor="middle", cls="lbl")
    return svg.render()


def lorenz_chart(r: dict[str, Any], k: int | None, w: float, h: float) -> str:
    """Lorenz curves for every dataset, the selected one in colour; Gini beside."""
    scored = _scored(r)
    left, top = 46.0, 10.0
    side = min(h - top - 30, w - left - 150)
    right, bottom = left + side, top + side
    svg = _Svg(w, h, "Lorenz curve")
    xs = _Lin(0.0, 1.0, left, right, pad=0.0, n=4)
    ys = _Lin(0.0, 1.0, bottom, top, pad=0.0, n=4)
    _yaxis(svg, ys, left, right, _pct0)
    for t in xs.ticks:
        svg.text(xs(t), bottom + 16, _pct(t, 0), anchor="middle")
    svg.add(f'<line class="ref" x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{top:.1f}"/>')
    for i, d in enumerate(scored):
        lz = _metrics(d, k).get("lorenz")
        if lz:
            svg.path([(xs(x), ys(y)) for x, y in zip(lz["x"], lz["y"])], f"dsl dsl-{i}")
    gx = right + 24
    svg.text(gx, top + 12, "Gini", cls="lbl")
    for j, d in enumerate(scored):
        y = top + 34 + j * 20
        svg.add(f'<line class="dsl dsl-{j}" x1="{gx:.1f}" x2="{gx + 16:.1f}" y1="{y - 4:.1f}" '
                f'y2="{y - 4:.1f}"/>')
        svg.text(gx + 22, y, f"{d['label']} {_fixed(_metrics(d, k).get('gini'), 3)}", cls="lbl")
    svg.text((left + right) / 2, bottom + 32, "share of exposure, riskiest first", anchor="middle")
    return svg.render()


def spread_chart(r: dict[str, Any], w: float) -> str:
    """The spread of the predictions as a histogram: each dataset's share of exposure (or weight)
    by predicted rate on the bins every dataset shares, the selected dataset as bars and the
    others as grey outlines, so a shift between them shows."""
    scored = _scored(r)
    hists = [(i, _metrics(d, None).get("predicted_rate_histogram")) for i, d in enumerate(scored)]
    found = [(i, h) for i, h in hists if h and h.get("mass") and h.get("edges")]
    if not found:
        return _none()
    edges = [float(e) for e in found[0][1]["edges"]]
    use_log = bool(found[0][1].get("log")) and edges[0] > 0
    share = {i: [float(m) / (sum(h["mass"]) or 1.0) for m in h["mass"]] for i, h in found}
    peak = max(max(v) for v in share.values()) or 1.0
    right, top = w - 8, 30.0
    bottom = top + 190.0
    h = bottom + 40
    ys = _Lin(0.0, peak, bottom, top, pad=0.05, zero=True, n=4)
    step = (ys.ticks[1] - ys.ticks[0]) * 100 if len(ys.ticks) > 1 else 1.0
    places = 0 if abs(step - round(step)) < 1e-9 else 1  # 2.5% steps keep their decimal

    def y_label(v: float) -> str:
        return _pct(v, places)

    left = _label_w(ys, y_label) + 14
    if use_log:
        e0, e1 = math.log10(edges[0]), math.log10(edges[-1])

        def xs(v: float) -> float:
            return left + (math.log10(max(v, 1e-300)) - e0) / (e1 - e0) * (right - left)

        cands = [m * 10.0 ** e for e in range(math.floor(e0) - 1, math.ceil(e1) + 1)
                 for m in (1, 2, 5)]
        ticks = [t for t in cands if e0 <= math.log10(t) <= e1]
    else:
        lin = _Lin(edges[0], edges[-1], left, right, pad=0.0, n=6)
        xs = lin
        ticks = [t for t in lin.ticks if edges[0] <= t <= edges[-1]]
    svg = _Svg(w, h, "spread of the predictions")
    basis = "exposure" if r["model"]["link"] == "log" else "weight"
    _ds_legend(svg, left, 12, len(scored), lambda i: [("box", "hbar", str(scored[i]["label"]))] + (
        [("line", "dsl", _others(scored, i))] if len(found) > 1 else []))
    _yaxis(svg, ys, left, right, y_label)
    svg.text(right, 12, f"share of {basis} per bin", anchor="end", cls="lbl")
    for t in ticks:
        svg.text(xs(t), bottom + 16, _num(t, 3), anchor="middle")
    svg.text((left + right) / 2, bottom + 34, f"predicted {_rate_name(r)}"
             + (", log scale" if use_log else ""), anchor="middle", cls="lbl")
    svg.hline(left, right, bottom, "ax")

    def outline(v: list[float]) -> list[tuple[float, float]]:
        return [(xs(edges[j] if e == 0 else edges[j + 1]), ys(s))
                for j, s in enumerate(v) for e in (0, 1)]

    for i, _ in found:
        bars = [_bar("hbar", xs(edges[j]) + 0.5, ys(v), max(xs(edges[j + 1]) - xs(edges[j]) - 1, 0.5),
                     bottom, r=1.0) for j, v in enumerate(share[i])]
        svg.group(f"ds ds-{i}", bars)
    if len(found) > 1:
        for i, _ in found:
            svg.path(outline(share[i]), "dsl")
    for i, _ in found:
        g = _Svg(w, h, "")
        g.path(outline(share[i]), "model")
        svg.group(f"ds ds-{i}", g.parts)
    last = len(edges) - 2
    for j in range(len(edges) - 1):
        lo, hi = _num(edges[j], 3), _num(edges[j + 1], 3)
        span = f"≤ {hi}" if j == 0 else f"≥ {lo}" if j == last else f"{lo}–{hi}"
        lines = [f"predicted {_rate_name(r)} {span}"]
        lines += [f"{scored[i]['label']}: {_pct(share[i][j], 1)} of the {basis}" for i, _ in found]
        svg.hit(xs(edges[j]), top, xs(edges[j + 1]) - xs(edges[j]), bottom - top, "\n".join(lines))
    return svg.render()


def balance_chart(r: dict[str, Any], w: float, *, k: int | None = None) -> str:
    """Every feature's A/E by band as a small chart on one shared scale: the Factors tab's check
    for all features at a glance. Each dataset's A/E steps across the feature's bands (the
    selected dataset in colour with its ±2σ band, the others in grey) between the dashed
    tolerance; on the selected dataset the bands the review flags are red and its near misses
    amber, and the band furthest from balance is marked and named above the chart.

    A band is as wide as its share of the first dataset's exposure, so a thin band's noise takes
    little room, and the scale is set by the bands the review checks (`factor_min_mass`): the
    rest are clipped at its edge."""
    scored = _scored(r)
    feats = [str(f["feature"]) for f in r.get("features") or []]
    if not scored or not feats:
        return _none()
    tol = r["thresholds"].get("factor_ae")
    fmin = r["thresholds"].get("factor_min_mass") or 0.0
    cls = None if k is None else _classes(r)[k]
    bands = {(i, f): [b for b in ((_ae_entry(d, f, cls) or {}).get("bands") or [])
                      if b.get("ae") is not None] for i, d in enumerate(scored) for f in feats}
    share: dict[tuple[int, str], dict[str, float]] = {}
    for key, bs in bands.items():
        total = sum(float(b.get("mass") or 0.0) for b in bs)
        share[key] = {str(b["label"]): float(b.get("mass") or 0.0) / total if total > 0 else 0.0
                      for b in bs}
    checked = [b for key, bs in bands.items() for b in bs if share[key][str(b["label"])] >= fmin]
    checked = checked or [b for bs in bands.values() for b in bs]
    vals = _fv(b["ae"] for b in checked)
    two_sd = min(2 * max(_fv(b.get("sd") for b in checked), default=0.0), 0.3)
    span = (tol or 0.1) * 1.3
    lo = max(min(vals + [1 - span, 1 - two_sd]), 0.4)
    hi = min(max(vals + [1 + span, 1 + two_sd]), 1.8)

    cols = max(1, min(len(feats), int((w + 16) // 186)))
    gap_x, gap_y, legend_h, name_h, plot_h = 16.0, 14.0, 28.0, 32.0, 56.0
    pw = (w - gap_x * (cols - 1)) / cols
    step = name_h + plot_h + gap_y
    rows = -(-len(feats) // cols)
    h = legend_h + rows * step - gap_y + 4
    svg = _Svg(w, h, "A/E by band for every feature")
    def items(i: int) -> list[tuple[str, str, str]]:
        out = [("line", "data", str(scored[i]["label"])), ("box", "bandbox", "±2σ")]
        if tol:
            out.append(("line", "guide", f"±{_pct(tol, 0)}"))
        out += [("dot", "flagdot", "flagged"), ("dot", "neardot", "near miss")]
        if len(scored) > 1:
            out.append(("line", "dsl", _others(scored, i)))
        return out

    _ds_legend(svg, 0, 12, len(scored), items, fit=w)  # a narrow card keeps the leading items
    chars = int(pw / (_FONT * 0.6))
    for n, f in enumerate(feats):
        px = (n % cols) * (pw + gap_x)
        py = legend_h + (n // cols) * step
        top, bottom = py + name_h, py + name_h + plot_h
        ys = _Lin(lo, hi, bottom, top, pad=0.04)

        def y_of(v: float) -> float:
            return ys(min(max(v, ys.lo), ys.hi))

        svg.text(px, py + 12, _short(f, chars), cls="strong")
        first = next((i for i in range(len(scored)) if bands[(i, f)]), 0)
        order = [str(b["label"]) for b in bands[(first, f)]]
        # Widths by the first dataset's exposure; a band too thin to see keeps a sliver.
        raw = [max(share[(first, f)].get(lab, 0.0), 0.004) for lab in order]
        x0: dict[str, float] = {}
        wid: dict[str, float] = {}
        edge = px
        for lab, v in zip(order, raw):
            x0[lab], wid[lab] = edge, pw * v / (sum(raw) or 1.0)
            edge += wid[lab]
        x_of = {lab: x0[lab] + wid[lab] / 2 for lab in order}
        svg.hline(px, px + pw, ys(1.0), "ref")
        if tol:
            for g in (1 - tol, 1 + tol):
                if ys.lo <= g <= ys.hi:
                    svg.hline(px, px + pw, ys(g), "guide")
        for i in range(len(scored)):
            bs = [b for b in bands[(i, f)] if str(b["label"]) in x_of]
            if any((b.get("sd") or 0) > 0 for b in bs):
                up = [(x0[str(b["label"])] + e * wid[str(b["label"])],
                       y_of(1 + 2 * (b.get("sd") or 0.0))) for b in bs for e in (0, 1)]
                dn = [(x0[str(b["label"])] + e * wid[str(b["label"])],
                       y_of(1 - 2 * (b.get("sd") or 0.0))) for b in bs for e in (0, 1)]
                svg.add(f'<path class="dsw dsw-{i}" d="{_d(up + dn[::-1])}Z"/>')
        for i in range(len(scored)):
            svg.path([(x0[str(b["label"])] + e * wid[str(b["label"])], y_of(b["ae"]))
                      for b in bands[(i, f)] if str(b["label"]) in x_of for e in (0, 1)],
                     f"dsl dsl-{i}")
        for i in range(len(scored)):
            bs = [b for b in bands[(i, f)] if str(b["label"]) in x_of]
            if not bs:
                continue
            g = _Svg(w, h, "")
            worst = max(bs, key=lambda b: abs(b.get("z") or 0.0))
            for b in bs:
                mark = {"warning": "flagdot", "near_miss": "neardot"}.get(str(b.get("flag")))
                if mark or b is worst:
                    g.add(f'<circle class="{mark or "ddot"}" cx="{x_of[str(b["label"])]:.1f}" '
                          f'cy="{y_of(b["ae"]):.1f}" r="{4 if mark else 3}"/>')
            tail = f": {_fixed(worst['ae'], 2)}"
            room = max(chars - len(tail) - 6, 4)
            g.text(px, py + 27, "worst " + _short(str(worst["label"]), room) + tail, cls="lbl")
            svg.group(f"ds ds-{i}", g.parts)
        for lab in order:
            lines = [f"{f}, band {lab}"]
            for i, d in enumerate(scored):
                b = next((b for b in bands[(i, f)] if str(b["label"]) == lab), None)
                if b is not None:
                    verdict = {"warning": " (flagged)", "near_miss": " (near miss)"}.get(
                        str(b.get("flag")), "")
                    lines.append(f"{d['label']}: A/E {_fixed(b['ae'])}, z {_signed(b.get('z'))}"
                                 + verdict)
            lines.append(f"{_pct(share[(first, f)].get(lab, 0.0), 1)} of the "
                         f"{scored[first]['label']} exposure")
            svg.hit(x0[lab], top, max(wid[lab], 1.0), plot_h, "\n".join(lines))
    return svg.render()


def factor_chart(r: dict[str, Any], feature: str, cls: str, k: int | None, w: float) -> str:
    """One feature on one x-axis of its levels: the deployed one-way table (log scale for a
    multiplier, thin cells faded) with the feature's Shapley value across the first dataset's
    policies (its mean and the 10th to 90th percentile: the one-way table plus its share of the
    interactions); the selected dataset's actual and expected rate by level (thin levels
    faded); A/E by the review's bands for every dataset (the selected one in colour with its
    ±2σ band); and the train exposure. Categorical levels are ordered by their effect."""
    link = r["model"]["link"]
    scored = _scored(r)
    key = None if k is None else cls
    entries = [_ae_entry(d, feature, key) for d in scored]
    base = entries[0] if entries else None
    if base is None:
        return _none()
    th = r["thresholds"]
    axis = base["axis"]
    labels = cell_labels(axis)
    mass = [float(m) for m in base["mass"]]
    n_cells = len(mass)
    t = _one_way(r, cls, feature)
    eff_cell: list[float | None] = [None] * n_cells
    thin_cell = [False] * n_cells
    mult, kind = link in ("log", "logit"), "effect"
    if t is not None:
        eff, mult, kind = _effects(t, link)
        sup = [float(s) for s in t.get("support") or []]
        thin_t = _thin(sup, th.get("thin_cell_mass") or 0.0) if sup else set()
        for c, tc in enumerate(_table_cells(axis, t["axes"][0])[:n_cells]):
            if tc is not None and tc < len(eff):
                eff_cell[c] = eff[tc]
                thin_cell[c] = tc in thin_t
    shap = base.get("shap") if isinstance(base.get("shap"), dict) else None
    shap_cells: dict[str, list[float | None]] = {}
    if shap is not None:
        for key in ("mean", "p10", "p90"):
            shap_cells[key] = [None if v is None else (math.exp(v) if mult else float(v))
                               for v in (shap.get(key) or [])][:n_cells]
    cells = [c for c in range(n_cells)
             if any(e is not None and c < len(e["rows"]) and e["rows"][c] > 0 for e in entries)]
    if not cells:
        return _none()

    def shap_at(key: str, c: int) -> float | None:
        values = shap_cells.get(key) or []
        return values[c] if c < len(values) else None

    if axis.get("levels"):
        neutral = 1.0 if mult else 0.0
        def effect_of(c: int) -> float:
            v = eff_cell[c]
            return neutral if v is None else v

        cells.sort(key=effect_of)
    band_of: list[dict[int, dict[str, Any]]] = []
    for e in entries:
        mp: dict[int, dict[str, Any]] = {}
        for b in (e or {}).get("bands") or []:
            for c in b["cells"]:
                mp[int(c)] = b
        band_of.append(mp)
    tol = th.get("factor_ae")
    neutral_effect = 1.0 if mult else 0.0
    # A feature no table uses has a Shapley value of no effect everywhere: no panel for it.
    has_shap = any(v is not None and abs(v - neutral_effect) > 1e-9
                   for c in cells for v in (shap_at("p10", c), shap_at("p90", c)))
    has_rel = any(eff_cell[c] is not None for c in cells) or has_shap
    rel = _fv([eff_cell[c] for c in cells] + [shap_at(key, c) for key in ("p10", "p90")
                                               for c in cells])
    aes = _fv(b.get("ae") for mp in band_of for c in cells for b in [mp.get(c)] if b is not None)
    two_sd = min(2 * max(_fv(b.get("sd") for mp in band_of for b in mp.values()), default=0.0), 0.3)
    ae_lo = max(min(aes + [1 - (tol or 0.1) * 1.3, 1 - two_sd]), 0.4)
    ae_hi = min(max(aes + [1 + (tol or 0.1) * 1.3, 1 + two_sd]), 1.8)
    # Actual and expected rate per level, per dataset; a level under the thin-cell share of the
    # dataset's exposure is faded, and its actual does not set the scale.
    thin_share = th.get("thin_cell_mass") or 0.0
    level: list[dict[int, tuple[float, float, bool]]] = []
    for e in entries:
        lv: dict[int, tuple[float, float, bool]] = {}
        if e is not None:
            total = sum(float(m) for m in e["mass"]) or 1.0
            for c in cells:
                m = float(e["mass"][c]) if c < len(e["mass"]) else 0.0
                if m > 0 and e["rows"][c]:
                    lv[c] = (e["actual"][c] / m, e["expected"][c] / m, m / total < thin_share)
        level.append(lv)
    lvl_vals = _fv([v[1] for lv in level for v in lv.values()]
                   + [v[0] for lv in level for v in lv.values() if not v[2]])
    top_h, lvl_h, mid_h, mass_h, gap, t0, foot = 180.0, 120.0, 120.0, 48.0, 30.0, 22.0, 26.0
    r0 = t0
    l0 = (r0 + top_h + gap) if has_rel else t0
    a0 = l0 + lvl_h + gap
    m0 = a0 + mid_h + gap
    h = m0 + mass_h + foot
    lvl_s = _Lin(0.0, max(lvl_vals or [1.0]), l0 + lvl_h, l0, zero=True, n=4)

    def rate_fmt(v: float) -> str:
        return _num(v, 3)
    rel_s: _Scale | None = None
    if has_rel:
        rel_s = (_Log(min(rel), max(rel), r0 + top_h, r0, max_ticks=5) if mult
                 else _Lin(min(rel), max(rel), r0 + top_h, r0, n=4))
    ae_s = _Lin(ae_lo, ae_hi, a0 + mid_h, a0, pad=0.04, n=4)

    def rel_fmt(v: float) -> str:
        return _mult_tick(v) if mult else _num(v, 2)

    lw = max(_label_w(rel_s, rel_fmt) if rel_s is not None else 0.0, _label_w(ae_s, _fixed2),
             _label_w(lvl_s, rate_fmt))
    left, right = 6 + lw + 6, w - 10
    n = len(cells)
    slot = (right - left) / n
    xs = {c: left + (j + 0.5) * slot for j, c in enumerate(cells)}
    svg = _Svg(w, h, f"{feature}: one-way table, actual and expected, A/E by band and exposure")
    if rel_s is not None:
        scale = rel_s
        _yaxis(svg, scale, left, right, rel_fmt)
        svg.hline(left, right, scale(1.0 if mult else 0.0), "ref")
        title = kind + (" (log scale)" if mult else "")
        svg.text(left, r0 - 8, title, cls="lbl")
        keys = [("line", "model", "one-way table")] if t is not None else []
        if has_shap:
            keys += [("line", "shapmean", "SHAP mean, with interactions"),
                     ("box", "shapband", f"80% of {scored[0]['label']} policies")]
        end = _legend(svg, left + _text_w(title) + 18, r0 - 8, keys)
        note = f"faded: under {_pct(th.get('thin_cell_mass'), 1)} of exposure"
        if any(thin_cell[c] for c in cells) and end + _text_w(note) < right:
            svg.text(right, r0 - 8, note, anchor="end")

        def runs(key: str) -> list[list[int]]:
            out: list[list[int]] = [[]]
            for c in cells:
                if shap_at(key, c) is None:
                    out.append([])
                else:
                    out[-1].append(c)
            return [run for run in out if run]

        for run in runs("p10"):
            upper = [(xs[c], scale(shap_at("p90", c) or 0.0)) for c in run]
            lower = [(xs[c], scale(shap_at("p10", c) or 0.0)) for c in reversed(run)]
            if len(run) == 1:
                upper = [(xs[run[0]] - slot / 3, upper[0][1]), (xs[run[0]] + slot / 3, upper[0][1])]
                lower = [(xs[run[0]] + slot / 3, lower[0][1]), (xs[run[0]] - slot / 3, lower[0][1])]
            svg.add(f'<path class="shapband" d="{_d(upper + lower)}Z"/>')
        pts, faded = [], []
        for c in cells:
            v = eff_cell[c]
            if v is not None:
                pts.append((xs[c], scale(v)))
                faded.append(thin_cell[c])
        _segments(svg, pts, faded)
        if n <= 24:
            for (x, y), f in zip(pts, faded):
                svg.add(f'<circle class="mdot" cx="{x:.1f}" cy="{y:.1f}" r="3.5"'
                        + (' style="opacity:.35"' if f else "") + "/>")
        for run in runs("mean"):
            svg.path([(xs[c], scale(shap_at("mean", c) or 0.0)) for c in run], "shapmean")
    _yaxis(svg, lvl_s, left, right, rate_fmt)
    svg.hline(left, right, l0 + lvl_h, "ax")
    for i, lv in enumerate(level):
        g = _Svg(w, h, "")
        title = f"actual and expected {_rate_name(r)} ({scored[i]['label']})"
        g.text(left, l0 - 8, title, cls="lbl")
        _legend(g, left + _text_w(title) + 18, l0 - 8,
                [("line", "data", "actual"), ("line", "model", "expected")])
        at = [c for c in cells if c in lv]
        if len(at) == 1:
            c0 = at[0]
            g.add(f'<circle class="mdot" cx="{xs[c0]:.1f}" cy="{lvl_s(lv[c0][1]):.1f}" r="4"/>')
            g.add(f'<circle class="ddot" cx="{xs[c0]:.1f}" '
                  f'cy="{lvl_s(min(lv[c0][0], lvl_s.hi)):.1f}" r="4"/>')
        elif at:
            g.path([(xs[c], lvl_s(lv[c][1])) for c in at], "model")
            _segments(g, [(xs[c], lvl_s(min(lv[c][0], lvl_s.hi))) for c in at],
                      [lv[c][2] for c in at], cls="data", faded_cls="dfaded")
        svg.group(f"ds ds-{i}", g.parts)
    _yaxis(svg, ae_s, left, right, _fixed2)
    if tol:
        for g_ in (1 - tol, 1 + tol):
            if ae_s.lo <= g_ <= ae_s.hi:
                svg.hline(left, right, ae_s(g_), "guide")
    svg.hline(left, right, ae_s(1.0), "ref")
    for i in range(len(scored)):
        g = _Svg(w, h, "")
        others = f", {_others(scored, i)} in grey" if len(scored) > 1 else ""
        g.text(left, a0 - 8, f"A/E by band (the review check): {scored[i]['label']} in colour"
               + others, cls="lbl")
        svg.group(f"ds ds-{i}", g.parts)
    if tol:
        svg.text(right, a0 - 8, f"dashed: ±{_pct(tol, 0)}; shaded: ±2σ", anchor="end")

    def ae_y(v: float) -> float:
        return ae_s(min(max(v, ae_s.lo), ae_s.hi))

    def steps(mp: dict[int, dict[str, Any]], sign: float) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for c in cells:
            b = mp.get(c)
            if b is None or b.get("ae") is None:
                continue
            v = b["ae"] if sign == 0 else 1 + sign * 2 * (b.get("sd") or 0.0)
            pts += [(xs[c] - slot / 2, ae_y(v)), (xs[c] + slot / 2, ae_y(v))]
        return pts

    for i, mp in enumerate(band_of):
        if any((b.get("sd") or 0) > 0 for b in mp.values()):
            up, dn = steps(mp, 1.0), steps(mp, -1.0)
            if up:
                svg.add(f'<path class="dsw dsw-{i}" d="{_d(up + dn[::-1])}Z"/>')
    for i, mp in enumerate(band_of):
        svg.path(steps(mp, 0.0), f"dsl dsl-{i}")
    total = sum(mass) or 1.0
    shown = sorted((mass[c] for c in cells), reverse=True)
    cap = shown[0] or 1.0
    clipped = len(shown) > 1 and shown[0] > 3 * shown[1] > 0
    if clipped:
        cap = shown[1] * 1.5
    svg.text(left, m0 - 8, f"exposure, {scored[0]['label']}", cls="lbl")
    bar_w = max(min(slot - 2, 26.0), 1.0)
    for c in cells:
        v = min(mass[c], cap)
        svg.add(_bar("mb", xs[c] - bar_w / 2, m0 + mass_h - v / cap * mass_h, bar_w, m0 + mass_h))
        if clipped and mass[c] > cap:
            note = f"{_pct(mass[c] / total, 0)} of exposure"
            far = xs[c] + bar_w / 2 + 6 + _text_w(note) > right
            svg.text(xs[c] - bar_w / 2 - 6 if far else xs[c] + bar_w / 2 + 6, m0 + 12, note,
                     anchor="end" if far else "start", cls="lbl")
    svg.hline(left, right, m0 + mass_h, "ax")
    _xlabels(svg, [xs[c] for c in cells], [labels[c] for c in cells], m0 + mass_h + 16, slot)
    for c in cells:
        lines = [f"{feature}: {labels[c]}"]
        if eff_cell[c] is not None:
            effect = _mult(eff_cell[c]) if mult else _num(eff_cell[c], 4)
            lines.append(f"{kind} {effect}" + (" (thin cell)" if thin_cell[c] else ""))
        if shap_at("mean", c) is not None:
            show = _mult if mult else _num
            lines.append(f"SHAP mean {show(shap_at('mean', c))}; 80% of {scored[0]['label']} "
                         f"policies {show(shap_at('p10', c))} to {show(shap_at('p90', c))}")
        lines.append(f"exposure ({scored[0]['label']}) {_num(mass[c])}, {_pct(mass[c] / total, 2)}")
        for i, mp in enumerate(band_of):
            if c in level[i]:
                act, exp, _ = level[i][c]
                lines.append(f"{scored[i]['label']}: actual {_num(act, 4)}, expected "
                             f"{_num(exp, 4)}")
            b = mp.get(c)
            if b is not None:
                lines.append(f"{scored[i]['label']}: band {b['label']}, A/E {_fixed(b.get('ae'))}, "
                             f"z {_signed(b.get('z'))}")
        svg.hit(xs[c] - slot / 2, t0, slot, m0 + mass_h - t0, "\n".join(lines))
    return svg.render()


def _bands_of_support(mass: Sequence[float], count: int = 4) -> list[list[int]]:
    """Runs of consecutive cells (missing-value cell 0 left out) of roughly equal total mass,
    at most ``count`` of them; cells without mass are dropped."""
    cells = [c for c in range(1, len(mass)) if mass[c] > 0]
    total = sum(mass[c] for c in cells)
    if not cells or total <= 0:
        return []
    if len(cells) <= count:
        return [[c] for c in cells]
    out: list[list[int]] = [[]]
    acc = 0.0
    for i, c in enumerate(cells):
        out[-1].append(c)
        acc += mass[c]
        left = len(cells) - i - 1
        if (acc >= total * len(out) / count and len(out) < count and left > 0):
            out.append([])
    return [b for b in out if b]


def _flat_index(idx: Sequence[int], shape: Sequence[int]) -> int:
    k = 0
    for i, d in zip(idx, shape):
        k = k * d + i
    return k


def interaction_layout(r: dict[str, Any], inter: dict[str, Any],
                       cls: str) -> dict[str, Any] | None:
    """How the Interactions tab draws one table, or None when it cannot: its first feature (the
    one with most levels) on the x-axis, and its partner's bands of equal exposure (at most four,
    as table-axis cells) with their labels. The report's A/E for the table uses the same bands."""
    scored = _scored(r)
    names = list(inter["features"])
    bank = _banks(r).get(cls) or {}
    t = next((t for t in bank.get("tables", []) if list(t["feature_names"]) == names), None)
    if t is None or not scored:
        return None
    key = None if not r["model"]["multiclass"] else cls
    entries = {f: _ae_entry(scored[0], f, key) for f in names}
    if any(e is None for e in entries.values()):
        return None

    def levels(f: str) -> int:
        e = entries[f]
        return sum(1 for n in (e or {}).get("rows") or [] if n)

    order = sorted(range(len(names)), key=lambda i: -levels(names[i]))
    ix, iby = order[0], order[1]
    shape = [int(d) for d in t["shape"]]
    values = [float(v) for v in t["values"]]
    support = [float(v) for v in t.get("support") or [0.0] * len(values)]
    # Σ support and Σ support × value per (x cell, partner cell), over any third feature.
    s_xb: dict[tuple[int, int], float] = {}
    f_xb: dict[tuple[int, int], float] = {}
    for k in range(len(values)):
        idx, rest = [], k
        for d in reversed(shape):
            idx.append(rest % d)
            rest //= d
        idx.reverse()
        cell = (idx[ix], idx[iby])
        s_xb[cell] = s_xb.get(cell, 0.0) + support[k]
        f_xb[cell] = f_xb.get(cell, 0.0) + support[k] * values[k]
    by_mass = [sum(s_xb.get((a, b), 0.0) for a in range(shape[ix])) for b in range(shape[iby])]
    bands = _bands_of_support(by_mass)
    by_axis = t["axes"][iby]
    by_labels = cell_labels(by_axis)

    def band_label(cells: list[int]) -> str:
        if by_axis.get("levels"):
            shown = ", ".join(by_labels[c] for c in cells[:2])
            return shown + (f" +{len(cells) - 2}" if len(cells) > 2 else "")
        return run_label(by_axis, cells[0], cells[-1])

    return {"t": t, "names": names, "key": key, "entries": entries, "ix": ix, "iby": iby,
            "xf": names[ix], "by": names[iby], "support": support, "s_xb": s_xb, "f_xb": f_xb,
            "by_mass": by_mass, "bands": bands, "band_labels": [band_label(b) for b in bands]}


def interaction_chart(r: dict[str, Any], inter: dict[str, Any], cls: str, w: float) -> str:
    """One interaction table on the levels of its first feature (the one with most levels): that
    feature's one-way table alone (dashed) and, for each band of equal exposure of its partner,
    the feature's effect with this table added (the table averaged over the band's cells, and
    over any third feature, by their exposure); under it, when the report has it, the A/E by
    band of the first feature within each partner band; the train exposure below. Thin bands
    and thin cells faded."""
    link = r["model"]["link"]
    scored = _scored(r)
    lay = interaction_layout(r, inter, cls)
    if lay is None:
        return _none()
    t, names, ix, iby = lay["t"], lay["names"], lay["ix"], lay["iby"]
    xf, by, bands = lay["xf"], lay["by"], lay["bands"]
    s_xb, f_xb, by_mass, support = lay["s_xb"], lay["f_xb"], lay["by_mass"], lay["support"]
    base = lay["entries"][xf] or {}
    axis = base["axis"]
    mass = [float(m) for m in base["mass"]]
    labels = cell_labels(axis)
    mult = link in ("log", "logit")
    if not bands:
        return _none("No exposure in this table.")
    band_label = dict(zip(map(tuple, bands), lay["band_labels"]))

    total = sum(support) or 1.0
    thin_share = r["thresholds"].get("thin_cell_mass") or 0.0
    to_table = _table_cells(axis, t["axes"][ix])
    one = _one_way(r, cls, xf)
    to_one = _table_cells(axis, one["axes"][0]) if one is not None else [None] * len(mass)
    main = [0.0] * len(mass)
    for c in range(len(mass)):
        tc = to_one[c] if c < len(to_one) else None
        if one is not None and tc is not None and tc < len(one["values"]):
            main[c] = float(one["values"][tc])
    cells = [c for c in range(len(mass)) if c < len(base["rows"]) and base["rows"][c] > 0]
    if axis.get("levels"):
        cells.sort(key=lambda c: main[c])
    if not cells:
        return _none()

    def shown(v: float) -> float:
        return math.exp(v) if mult else v

    lines: list[list[tuple[int, float, bool]]] = []
    for band in bands:
        line = []
        for c in cells:
            tc = to_table[c] if c < len(to_table) else None
            if tc is None:
                continue
            sw = sum(s_xb.get((tc, b), 0.0) for b in band)
            if sw <= 0:
                continue
            adj = sum(f_xb.get((tc, b), 0.0) for b in band) / sw
            line.append((c, main[c] + adj, sw < thin_share * total))
        lines.append(line)
    vals = [shown(v) for line in lines for _, v, _ in line] + [shown(main[c]) for c in cells]
    top_h, mass_h, foot = 210.0, 40.0, 26.0
    legend_items = [("line", "ref1", f"{xf}, one-way table")] + [
        ("line", f"ib{i}", f"{by} {band_label[tuple(b)]}") for i, b in enumerate(bands)]
    if by_mass and by_mass[0] > 0:
        legend_items.append(("box", "tol", f"{by} missing: not shown"))
    rows_of_legend: list[list[tuple[str, str, str]]] = [[]]
    width = 0.0
    for item in legend_items:
        need = 22 + _text_w(item[2]) + 18
        if rows_of_legend[-1] and width + need > w - 60:
            rows_of_legend.append([])
            width = 0.0
        rows_of_legend[-1].append(item)
        width += need
    # The report's A/E by band of the first feature within each partner band, per dataset; a
    # band pair under `factor_min_mass` of the dataset's exposure is faded and off the scale.
    ae = inter.get("ae") if isinstance(inter.get("ae"), dict) else None
    ae_ds: list[tuple[int, dict[str, Any]]] = []
    if ae is not None and [list(b) for b in ae.get("by_bands") or []] == [list(b) for b in bands]:
        by_label = {str(d["label"]): d for d in ae.get("datasets") or []}
        ae_ds = [(i, by_label[str(d["label"])]) for i, d in enumerate(scored)
                 if str(d["label"]) in by_label]
    x_band_of = {int(c): i for i, run in enumerate((ae or {}).get("x_bands") or []) for c in run}
    fmin = r["thresholds"].get("factor_min_mass") or 0.0
    tol = r["thresholds"].get("factor_ae")

    def thin_pair(d: dict[str, Any], a: int, b: int) -> bool:
        total = sum(sum(row) for row in d["mass"]) or 1.0
        return float(d["mass"][a][b]) < fmin * total

    ae_vals = _fv(d["ae"][a][b] for _, d in ae_ds for a in range(len(d["ae"]))
                  for b in range(len(bands)) if not thin_pair(d, a, b))
    span = (tol or 0.1) * 1.3
    # One strip per partner band, on the x-axis of the chart above and one shared scale.
    # First every partner band's line on one panel (the context), then one strip per band with
    # its ±2σ band (the evidence).
    strip_h, strip_gap, ov_h = 58.0, 22.0, 130.0
    ae_h = len(bands) * (strip_h + strip_gap) - strip_gap if ae_ds else 0.0
    t0 = 12 + 18 * len(rows_of_legend) + 4
    o0 = t0 + top_h + 44
    a0 = o0 + ov_h + 58
    m0 = (a0 + ae_h + 30) if ae_ds else (t0 + top_h + 26)
    h = m0 + mass_h + foot
    ys: _Scale = (_Log(min(vals), max(vals), t0 + top_h, t0, max_ticks=5) if mult
                  else _Lin(min(vals), max(vals), t0 + top_h, t0, n=4))
    ae_lo = max(min(ae_vals + [1 - span]), 0.5)
    ae_hi = min(max(ae_vals + [1 + span]), 1.6)
    # The actual and expected rate per level of the first feature within each partner band; a
    # pair under `thin_cell_mass` of the dataset's exposure has its actual faded, off the scale.
    thin_level = r["thresholds"].get("thin_cell_mass") or 0.0
    rates: dict[int, list[list[tuple[int, float, float, bool]]]] = {}
    for i, d in ae_ds:
        lv = d.get("levels") or {}
        act, exp, ms = lv.get("actual") or [], lv.get("expected") or [], lv.get("mass") or []
        total = sum(sum(row) for row in ms) or 1.0
        rates[i] = [[(c, act[c][b] / ms[c][b], exp[c][b] / ms[c][b], ms[c][b] < thin_level * total)
                     for c in cells if c < len(ms) and ms[c][b] > 0] for b in range(len(bands))]
    rate_vals = _fv(v for per in rates.values() for pts in per for _, a, e, thin in pts
                    for v in ((e,) if thin else (a, e)))
    lv_s = _Lin(0.0, max(rate_vals or [1.0]), o0 + ov_h, o0, zero=True, n=4)

    def rate_fmt(v: float) -> str:
        return _num(v, 3)

    def strip(b: int) -> _Lin:
        y0 = a0 + b * (strip_h + strip_gap)
        return _Lin(ae_lo, ae_hi, y0 + strip_h, y0, pad=0.04, n=2)

    def fmt(v: float) -> str:
        return _mult_tick(v) if mult else _num(v, 2)

    left = 6 + max(_label_w(ys, fmt), _label_w(lv_s, rate_fmt) if ae_ds else 0.0,
                   _text_w("0.90") if ae_ds else 0.0) + 6
    right = w - 10
    n = len(cells)
    slot = (right - left) / n
    xs = {c: left + (j + 0.5) * slot for j, c in enumerate(cells)}
    svg = _Svg(w, h, f"{' × '.join(names)} by band of {by}")
    for row_i, row in enumerate(rows_of_legend):
        _legend(svg, left, 12 + 18 * row_i, row)
    _yaxis(svg, ys, left, right, fmt)
    svg.hline(left, right, ys(1.0 if mult else 0.0), "ref")
    svg.path([(xs[c], ys(shown(main[c]))) for c in cells], "ref1")
    for i, line in enumerate(lines):
        pts = [(xs[c], ys(shown(v))) for c, v, _ in line]
        _segments(svg, pts, [thin for _, _, thin in line], cls=f"ib{i}", faded_cls=f"ib{i} fade")
    if ae_ds:
        _yaxis(svg, lv_s, left, right, rate_fmt)
        svg.hline(left, right, o0 + ov_h, "ax")
        keys = [("line", "exl", "expected"), ("line", "acl", "actual")]
        _legend(svg, right - sum(40 + _text_w(k[2]) for k in keys) + 18, o0 - 8, keys)
        if tol:
            svg.text(right, a0 - 24, f"dashed: ±{_pct(tol, 0)}; shaded: ±2σ", anchor="end")
        for i, d in ae_ds:
            g = _Svg(w, h, "")
            g.text(left, o0 - 8, f"actual and expected {_rate_name(r)} by {xf}, a line per {by} "
                   f"band ({d['label']})", cls="lbl")
            for b, band_pts in enumerate(rates[i]):
                g.path([(xs[c], lv_s(e)) for c, _, e, _ in band_pts], f"ib{b}")
                _segments(g, [(xs[c], lv_s(min(a, lv_s.hi))) for c, a, _, _ in band_pts],
                          [thin for _, _, _, thin in band_pts], cls=f"ib{b} dash",
                          faded_cls=f"ib{b} dash fade")
            svg.group(f"ds ds-{i}", g.parts)
        for b in range(len(bands)):
            sc = strip(b)
            _legend(svg, left, sc.b - 6, [("line", f"ib{b}", f"{by} {band_label[tuple(bands[b])]}")])
            marks = [1.0] + ([1 - tol, 1 + tol] if tol else [])
            for v in marks:
                if sc.lo <= v <= sc.hi:
                    svg.hline(left, right, sc(v), "ref" if v == 1.0 else "guide")
                    svg.text(left - 6, sc(v) + 4, _fixed2(v), anchor="end")
        for i, d in ae_ds:
            g = _Svg(w, h, "")
            g.text(left, a0 - 24, f"each {by} band on its own ({d['label']})", cls="lbl")
            for b in range(len(bands)):
                sc = strip(b)

                def y_of(v: float) -> float:
                    return sc(min(max(v, sc.lo), sc.hi))

                runs: list[list[tuple[int, float, float, bool]]] = [[]]
                for c in cells:
                    a = x_band_of.get(c)
                    v = d["ae"][a][b] if a is not None and a < len(d["ae"]) else None
                    if v is None or a is None:
                        runs.append([])
                        continue
                    sd = d.get("sd", [[None]])[a][b] if "sd" in d else None
                    runs[-1].append((c, float(v), float(sd or 0.0), thin_pair(d, a, b)))
                for run in (r_ for r_ in runs if r_):
                    up = [(xs[c] + e * slot / 2, y_of(1 + 2 * sd)) for c, _, sd, _ in run
                          for e in (-1, 1)]
                    dn = [(xs[c] + e * slot / 2, y_of(1 - 2 * sd)) for c, _, sd, _ in run
                          for e in (-1, 1)]
                    g.add(f'<path class="ibw{b}" d="{_d(up + dn[::-1])}Z"/>')
                    _segments(g, [(xs[c] + e * slot / 2, y_of(v)) for c, v, _, _ in run
                                  for e in (-1, 1)],
                              [thin for _, _, _, thin in run for _ in (0, 1)],
                              cls=f"ib{b}", faded_cls=f"ib{b} fade")
            svg.group(f"ds ds-{i}", g.parts)
    svg.text(left, m0 - 8, f"exposure, {scored[0]['label']}", cls="lbl")
    cap = max((mass[c] for c in cells), default=0.0) or 1.0
    bar_w = max(min(slot - 2, 26.0), 1.0)
    for c in cells:
        svg.add(_bar("mb", xs[c] - bar_w / 2, m0 + mass_h - mass[c] / cap * mass_h, bar_w,
                     m0 + mass_h))
    svg.hline(left, right, m0 + mass_h, "ax")
    _xlabels(svg, [xs[c] for c in cells], [labels[c] for c in cells], m0 + mass_h + 16, slot)
    show = _mult if mult else _num
    for c in cells:
        tips = [f"{xf} {labels[c]}: one-way {show(shown(main[c]))}"]
        for i, line in enumerate(lines):
            hit = next((v for cc, v, _ in line if cc == c), None)
            if hit is not None:
                tips.append(f"{band_label[tuple(bands[i])]} {show(shown(hit))}")
        a = x_band_of.get(c)
        for i, d in ae_ds:
            here = [(b, act, exp) for b, band_pts in enumerate(rates[i])
                    for cc, act, exp, _ in band_pts if cc == c]
            if here:
                tips.append(f"{d['label']} actual / expected: " + ", ".join(
                    f"{band_label[tuple(bands[b])]} {_num(act, 4)} / {_num(exp, 4)}"
                    for b, act, exp in here))
            if a is not None and a < len(d["ae"]):
                tips.append(f"{d['label']} A/E: " + ", ".join(
                    f"{band_label[tuple(bands[b])]} {_fixed(d['ae'][a][b], 2)}"
                    for b in range(len(bands))))
        svg.hit(xs[c] - slot / 2, t0, slot, m0 + mass_h - t0, "\n".join(tips))
    return svg.render()


# --------------------------------------------------------------------------- takeaways --------

def _lift_takeaway(r: dict[str, Any], i: int, k: int | None) -> str:
    d = _scored(r)[i]
    lift = _lift(d, k)
    aes = _fv(b.get("ae") for b in lift)
    if len(lift) < 2 or not aes or not ((lift[0].get("actual") or 0) > 0
                                       and (lift[-1].get("actual") or 0) > 0):
        return "Calibration by decile of predicted rate"
    spread = lift[-1]["actual"] / lift[0]["actual"]
    return (f"Top decile {spread:.1f}× the bottom; {d['label']} decile A/E "
            f"{min(aes):.2f}–{max(aes):.2f}")


def _importance_takeaway(r: dict[str, Any]) -> str:
    feats = [f for f in r.get("features") or [] if f.get("used")]
    if not feats:
        return "No deployed tables"
    n = min(3, len(feats))
    words = {1: "One factor carries", 2: "Two factors carry", 3: "Three factors carry"}
    return f"{words[n]} {_pct(sum(f['importance'] for f in feats[:n]), 0)} of the model"


def _gain_takeaway(r: dict[str, Any]) -> str:
    pd = _path_data(r)
    if pd is None:
        return "Pruning"
    rep, best = pd["rep"], pd["best"]
    n = rep.get("n_candidates") or len(pd["scores"])
    text = (f"Cutoff at {pd['bank'][best]} of {n} tables, "
            f"{_pct(pd['share'][best], 1)} of the gain")
    if pd["path"].get("dropped_low_share"):
        text += f"; {r['structure']['deployed_tables']} deployed"
    return text


def _balance_rule(r: dict[str, Any]) -> str:
    """The factor check's rule, with the thresholds in force."""
    th = r["thresholds"]
    tol, fz = th.get("factor_ae"), th.get("factor_z")
    if tol is None:
        return "A/E across each feature's bands; the factor check is off"
    rule = f"outside ±{_pct(tol, 0)}" + (f" and ≥ {fz:g}σ from balance" if fz else "")
    return f"A/E by band, width by exposure; flagged {rule}"


def _balance_takeaway(r: dict[str, Any], i: int, k: int | None = None) -> str:
    d = _scored(r)[i]
    cls = None if k is None else _classes(r)[k]
    worst: tuple[str, dict[str, Any]] | None = None
    for e in d["actual_vs_expected"]:
        if cls is not None and str(e.get("class")) != cls:
            continue
        for b in e.get("bands") or []:
            if (b.get("z") is not None and b.get("ae") is not None
                    and (worst is None or abs(b["z"]) > abs(worst[1]["z"]))):
                worst = (str(e["feature"]), b)
    # One warning per feature with a band out of balance (near misses are notes, not counted).
    off = sum(1 for f in r["review"] if f["check"] == "factor_ae" and f["status"] == "warning"
              and f.get("dataset") == d["label"] and (cls is None or str(f.get("class")) == cls))
    if worst is None:
        return "A/E by factor band"
    return (f"{off} factor{'s' if off > 1 else ''} out of balance on {d['label']}" if off
            else f"No factor out of balance on {d['label']}")


# --------------------------------------------------------------------------- HTML pieces ------

def _table(cols: Sequence[str], rows: Sequence[Sequence[str]], *, numeric: Iterable[int] = (),
           row_class: Sequence[str] | None = None, scroll: str = "", cls: str = "",
           tones: Sequence[Sequence[str]] | None = None) -> str:
    """Rows of already-escaped cell HTML; ``scroll`` is ``scroll`` or ``short`` for a box;
    ``tones`` adds a class per cell (``_tones``)."""
    num = set(numeric)
    head = "".join(f'<th class="{"n" if i in num else ""}">{_e(c)}</th>' for i, c in enumerate(cols))
    body = []
    for r_i, row in enumerate(rows):
        rc = f' class="{row_class[r_i]}"' if row_class and row_class[r_i] else ""
        tone = tones[r_i] if tones is not None and r_i < len(tones) else []
        tds = []
        for i, c in enumerate(row):
            classes = " ".join(x for x in ("n" if i in num else "", tone[i] if i < len(tone) else "")
                               if x)
            tds.append(f'<td class="{classes}">{c}</td>')
        body.append(f"<tr{rc}>" + "".join(tds) + "</tr>")
    tc = f' class="{cls}"' if cls else ""
    return (f'<div class="tw{" " + scroll if scroll else ""}"><table{tc}><thead><tr>{head}</tr>'
            f'</thead><tbody>{"".join(body)}</tbody></table></div>')


def _kv(data: Any, depth: int = 0) -> str:
    """A nested dict as key/value tables; lists of dicts as tables (capped)."""
    if isinstance(data, dict):
        if not data:
            return '<span class="muted">—</span>'
        rows = "".join(f"<tr><td>{_e(k)}</td><td>{_kv(v, depth + 1)}</td></tr>"
                       for k, v in data.items())
        return f'<table class="kv">{rows}</table>'
    if isinstance(data, list):
        if not data:
            return '<span class="muted">[]</span>'
        if all(isinstance(v, dict) for v in data) and depth < 4:
            keys: list[str] = []
            for v in data:
                keys += [k for k in v if k not in keys]
            cells = [[_short_val(v.get(k)) for k in keys] for v in data[:_RANKED_ROWS]]
            return _table(keys, cells)
        text = ", ".join(_short_val(v) for v in data[:60])
        return text + (f" … ({len(data)} values)" if len(data) > 60 else "")
    return _short_val(data)


def _short_val(v: Any) -> str:
    if v is None:
        return '<span class="muted">None</span>'
    if isinstance(v, float):
        return _num(v)
    if isinstance(v, (dict, list)):
        text = json.dumps(v, default=str)
        return f'<span class="mono">{_e(text if len(text) <= 160 else text[:159] + "…")}</span>'
    return _e(v)


def _badge(status: str, text: str | None = None) -> str:
    icon = {"critical": "!", "warning": "!", "pass": "✓", "info": "i"}.get(status, "?")
    word = text if text is not None else {"info": "note"}.get(status, status)
    return (f'<span class="badge b-{_e(status)}"><i aria-hidden="true">{icon}</i>'
            f"{_e(word)}</span>")


def _card(title: str, body: str, *, what: str = "", extra: str = "", cls: str = "",
          titles: list[str] | None = None) -> str:
    """A card; ``titles`` gives one title per dataset, shown with it by the dataset switch
    (``title`` is then the hover text)."""
    shown = ("".join(f'<span class="ds ds-{i}">{_e(t)}</span>' for i, t in enumerate(titles))
             if titles else _e(title))
    head = f'<h3 title="{_e(title)}">{shown}</h3>'
    if extra:
        head = f'<div class="head">{head}{extra}</div>'
    cap = f'<p class="what" title="{_e(what)}">{_e(what)}</p>' if what else ""
    classes = "card" + ("" if what else " solo") + (f" {cls}" if cls else "")
    return f'<div class="{classes}">{head}{cap}{body}</div>'


def _row(*cards: str) -> str:
    return '<div class="row">' + "".join(cards) + "</div>"


class _Radio:
    """A CSS-only radio group. Its inputs sit first in ``.app``; its labels may sit in ``.bar``
    or anywhere in ``.body``; content tagged ``{name} {name}-{i}`` shows only for option i."""

    def __init__(self, name: str, n: int, default: int = 0) -> None:
        self.name, self.n, self.default = name, max(n, 1), default

    def inputs(self) -> str:
        return "".join(f'<input class="sr" type="radio" name="{self.name}" id="{self.name}-{i}"'
                       f'{" checked" if i == self.default else ""}>' for i in range(self.n))

    def label(self, i: int, html: str, cls: str = "", title: str = "") -> str:
        c = f' class="{cls}"' if cls else ""
        t = f' title="{_e(title)}"' if title else ""
        return f'<label for="{self.name}-{i}"{c}{t}>{html}</label>'

    def tag(self, i: int) -> str:
        return f"{self.name} {self.name}-{i}"

    def css(self, selected: str) -> str:
        out = []
        for i in range(self.n):
            on = f"#{self.name}-{i}:checked"
            lab = f"label[for={self.name}-{i}]"
            out.append(f"{on}~.body .{self.name}:not(.{self.name}-{i}){{display:none}}")
            if selected:
                out.append(f"{on}~.bar {lab},{on}~.body {lab}{{{selected}}}")
            out.append(f"#{self.name}-{i}:focus-visible~.bar {lab},#{self.name}-{i}:focus-visible~"
                       f".body {lab}{{outline:2px solid var(--data);outline-offset:-2px}}")
        return "".join(out)


def _dataset_css(n: int) -> str:
    """The selected dataset's series in colour with its ±2σ band; its table rows washed."""
    out = []
    for i in range(n):
        on = f"#ds-{i}:checked~.body"
        out.append(f"{on} .dsl-{i}{{stroke:var(--data);stroke-width:2.5}}{on} .dsw-{i}{{display:inline}}"
                   f"{on} .dsr-{i} td{{background:var(--wash)}}"
                   f"{on} .dsr-{i} td:first-child{{box-shadow:-8px 0 0 var(--wash)}}"
                   f"{on} .dsr-{i} td:last-child{{box-shadow:8px 0 0 var(--wash)}}")
    return "".join(out)


def _per_class(r: dict[str, Any], classes: _Radio | None, render: Callable[[str, int | None], str]
               ) -> str:
    """``render(class label, class index)`` once per class bank, each shown with its class."""
    if classes is None:
        return render("", None)
    return "".join(f'<div class="{classes.tag(k)}">{render(c, k)}</div>'
                   for k, c in enumerate(_classes(r)))


# --------------------------------------------------------------------------- page parts -------

def _open(tab: str) -> str:
    """The opening tag of a tab's section: shown while its radio is checked."""
    return f'<section class="tab tb tb-{_TAB_NAMES.index(tab)}" id="{tab.lower()}">'


def _header(r: dict[str, Any], name: str | None) -> str:
    m = r["model"]
    title = name or r.get("name") or f"{m['class']} ({m['objective']})"
    fitted = m.get("fitted_at")
    when = (f"fitted {str(fitted).replace('T', ' ').replace('+00:00', ' UTC')}" if fitted
            else "fit time unknown")
    dur = m.get("fit_duration_seconds")
    if dur is not None:
        when += f" in {_num(dur, 3)} s"
    st = r["structure"]
    bank0 = st["banks"][0] if st["banks"] else {"by_order": {}}
    words = {"1": "one-way", "2": "two-way", "3": "three-way"}
    by_order = ", ".join(f"{v} {words.get(o, f'order-{o}')}" for o, v in sorted(bank0["by_order"].items()))
    used = sum(1 for f in r.get("features") or [] if f.get("used"))
    tables = (f"{_num(st['deployed_tables'])} tables" + (f" ({by_order})" if by_order else "")
              + (" per class" if m["multiclass"] else "")
              + f" over {used} of {len(m['native_feature_order'])} features")
    sub = [when, f"{m['objective']}, {m['link']} link", tables, f"t-boost {m['t_boost_version']}"]
    return (f'<header class="top"><h1>{_e(title)}</h1>'
            f'<p class="sub">{" · ".join(_e(x) for x in sub)}</p></header>')


def _headline(r: dict[str, Any]) -> str:
    """Datasets by the numbers a reviewer reads first; the selected dataset's row washed."""
    scored = _scored(r)
    m = r["model"]
    log = m["link"] == "log"
    if m["multiclass"]:
        cols = ["dataset", "rows", "deviance explained", "accuracy"]
    else:
        cols = ["dataset", "exposure" if log else "rows", _actual_name(r), "A/E",
                "deviance explained", "Gini"]
    rows, classes = [], []
    mass = _column([d["metrics"].get("mass") if log else d["rows"] for d in scored])
    actual = _column([d["metrics"].get("actual") for d in scored])
    for i, d in enumerate(scored):
        mt = d["metrics"]
        if m["multiclass"]:
            rows.append([_e(d["label"]), _num(d["rows"]), _pct(mt.get("deviance_explained"), 2),
                         _pct(mt.get("accuracy"), 1)])
        else:
            rows.append([_e(d["label"]), mass[i], actual[i], _fixed(mt.get("ae")),
                         _pct(mt.get("deviance_explained"), 2), _fixed(mt.get("gini"))])
        classes.append(f"dsr-{i}")
    for d in r["datasets"]:
        if d.get("error"):
            rows.append([_e(d["label"]), _num(d["rows"])] + ["—"] * (len(cols) - 2))
            classes.append("")
    return _table(cols, rows, numeric=range(1, len(cols)), row_class=classes)


def _scope(f: dict[str, Any]) -> str:
    scope = " · ".join(str(x) for x in (f.get("dataset"), f.get("feature"), f.get("class"))
                       if x is not None)
    return f"{scope}: {f['message']}" if scope else str(f["message"])


def _issue_groups(r: dict[str, Any]) -> list[tuple[str, str, str]]:
    """``(status, short text, full text)`` per check with open findings, most severe first."""
    order = {"critical": 0, "warning": 1, "info": 2}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for f in r["review"]:
        if f["status"] != "pass":
            groups.setdefault((f["status"], f["check"]), []).append(f)
    out = []
    for (status, check), fl in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1])):
        full = "; ".join(_scope(f) for f in fl)
        if len(fl) == 1:
            short = full
        elif check == "missing_level":
            short = ("missing values, with no exposure behind them, would score "
                     + ", ".join(f"{f['feature']} {_mult(f['value'])}" for f in fl))
        elif check == "thin_cells":
            short = (f"{len(fl)} one-way tables have thin cells beyond their supported range: "
                     + ", ".join(str(f["feature"]) for f in fl))
        else:
            short = full
        out.append((status, short, full))
    return out


def _counts(r: dict[str, Any]) -> str:
    n = {s: sum(f["status"] == s for f in r["review"]) for s in ("critical", "warning", "info", "pass")}
    words = (("critical", "critical", "critical"), ("warning", "warning", "warnings"),
             ("info", "note", "notes"), ("pass", "passed", "passed"))
    return '<span class="counts">' + "".join(
        _badge(s, f"{n[s]} {one if n[s] == 1 else many}") for s, one, many in words if n[s]
    ) + "</span>"


def _overview(r: dict[str, Any], tabs: _Radio, classes: _Radio | None) -> str:
    scored = _scored(r)
    last = len(scored) - 1
    groups = _issue_groups(r)
    lines = [f'<li>{_badge(s)}<span class="t" title="{_e(full)}">{_e(short)}</span></li>'
             for s, short, full in groups[:_OVERVIEW_ISSUES]]
    rest = len(groups) - _OVERVIEW_ISSUES
    link = tabs.label(_CHECKS_TAB, (f"{rest} more under Checks →" if rest > 0
                                    else "All checks and thresholds →"), cls="link")
    issues = (f'<ul class="issues">{"".join(lines)}</ul>' if lines
              else '<p class="ok">No open findings.</p>')
    review = _card("Review", issues + f'<div class="more">{link}</div>', extra=_counts(r))
    perf = _card("Performance", _headline(r))

    def cal(c: str, k: int | None) -> str:
        return calibration_chart(r, k, _W_HALF, _H_ROW)

    def one_way(c: str, k: int | None) -> str:
        return multiples_chart(r, _W_HALF, _H_ROW, cls=c)

    k0 = 0 if classes is not None else None
    cal_card = _card(_lift_takeaway(r, last, k0), _per_class(r, classes, cal),
                     titles=[_lift_takeaway(r, i, k0) for i in range(len(scored))],
                     what="Equal-exposure deciles of predicted rate; A/E for every dataset, ±2σ shaded")
    imp_card = _card(_importance_takeaway(r), importance_chart(r, _W_HALF, _H_ROW),
                     what="Share of model variance: own one-way table, then through interactions")
    thin = r["thresholds"].get("thin_cell_mass")
    one_way_card = _card("The largest one-way tables", _per_class(r, classes, one_way),
                         what=("Effects on one shared scale; faded cells hold under "
                               f"{_pct(thin, 1)} of the table's exposure"))
    if _path_data(r) is not None:
        gain_card = _card(_gain_takeaway(r), gain_chart(r, _W_HALF, _H_ROW),
                          what="Out-of-bag gain as ranked tables join the bank; the cutoff and floor")
    else:
        def lorenz(c: str, k: int | None) -> str:
            return lorenz_chart(r, k, _W_HALF, _H_ROW)

        gain_card = _card("Lorenz curve", _per_class(r, classes, lorenz),
                          what="Share of actual against share of exposure, riskiest first")
    return (_open("Overview") + _row(perf, review) + _row(cal_card, imp_card)
            + _row(one_way_card, gain_card) + "</section>")


def _performance(r: dict[str, Any], classes: _Radio | None) -> str:
    scored = _scored(r)
    multi = r["model"]["multiclass"]
    log = r["model"]["link"] == "log"
    if multi:
        cols = ["dataset", "rows", "weight", "mean deviance", "null deviance",
                "deviance explained", "accuracy"]
        rows = [[_e(d["label"]), _num(d["rows"]), _num(d["metrics"].get("weight")),
                 _fixed(d["metrics"].get("deviance"), 5), _fixed(d["metrics"].get("null_deviance"), 5),
                 _pct(d["metrics"].get("deviance_explained"), 2),
                 _pct(d["metrics"].get("accuracy"), 2)] for d in scored]
    else:
        cols = ["dataset", "rows", "exposure" if log else "weight", "actual", "expected", "A/E",
                "mean deviance", "null deviance", "deviance explained", "Gini", "Gini / perfect"]
        rows = []
        cols_fmt = {key: _column([d["metrics"].get(key) for d in scored])
                    for key in ("mass", "actual", "expected")}
        for i, d in enumerate(scored):
            mt = d["metrics"]
            rows.append([_e(d["label"]), _num(d["rows"]), cols_fmt["mass"][i], cols_fmt["actual"][i],
                         cols_fmt["expected"][i], _fixed(mt.get("ae")), _fixed(mt.get("deviance"), 5),
                         _fixed(mt.get("null_deviance"), 5), _pct(mt.get("deviance_explained"), 2),
                         _fixed(mt.get("gini")), _fixed(mt.get("gini_normalised"))])
    errors = "".join(f'<p class="note">{_e(d["label"])} could not be scored: {_e(d["error"])}</p>'
                     for d in r["datasets"] if d.get("error"))
    metrics = _card("Metrics", _table(cols, rows, numeric=range(1, len(cols)),
                                      row_class=[f"dsr-{i}" for i in range(len(scored))]) + errors,
                    what="Deviance explained is against the best constant model on the same rows")
    last = len(scored) - 1

    def cal(c: str, k: int | None) -> str:
        dec_rows, dec_cls = [], []
        for i, d in enumerate(scored):
            for b in _lift(d, k):
                dec_rows.append([_e(d["label"]), str(b["band"]), _num(b["rows"]),
                                 _pct(b.get("mass_share"), 1), _num(b.get("actual"), 4),
                                 _num(b.get("predicted"), 4), _fixed(b.get("ae")), _signed(b.get("z"))])
                dec_cls.append(f"ds ds-{i}")
        table = _table(["dataset", "decile", "rows", "share of exposure", "actual", "predicted",
                        "A/E", "z"], dec_rows, numeric=range(1, 8), row_class=dec_cls)
        suffix = f" · class {c}" if c else ""
        return _card(_lift_takeaway(r, last, k) + suffix, calibration_chart(r, k, _W_FULL, 360)
                     + f"<details><summary>Decile table</summary>{table}</details>",
                     titles=[_lift_takeaway(r, i, k) + suffix for i in range(len(scored))],
                     what=("Equal-exposure deciles of predicted rate: actual and predicted above, "
                           "the A/E of each dataset below with ±2σ shaded"))

    def pair(c: str, k: int | None) -> str:
        lor = _card("Lorenz curve", lorenz_chart(r, k, _W_HALF, 300),
                    what="Share of actual against share of exposure, highest predicted rate first")
        bal = _card(_balance_takeaway(r, last, k), balance_chart(r, _W_HALF, k=k),
                    titles=[_balance_takeaway(r, i, k) for i in range(len(scored))],
                    what=_balance_rule(r))
        return _row(lor, bal)

    parts = [metrics, _per_class(r, classes, cal), _per_class(r, classes, pair)]
    if not multi:
        pct_rows = [[_e(d["label"])] + [_num(v, 4) for v in (_metrics(d, None).get(
            "predicted_rate_percentiles") or {}).values()] for d in scored]
        keys = list((_metrics(scored[0], None).get("predicted_rate_percentiles") or {}).keys())
        parts.append(_card("Spread of the predictions", spread_chart(r, _W_FULL)
                           + "<details><summary>Percentiles</summary>"
                           + _table(["dataset"] + [f"p{q}" for q in keys], pct_rows,
                                    numeric=range(1, len(keys) + 1)) + "</details>",
                           what=(f"Share of {'exposure' if log else 'weight'} by predicted "
                                 f"{_rate_name(r)}, on bins shared by every dataset")))
    else:
        for i, d in enumerate(scored):
            cm = d["metrics"]["class_matrix"]
            labels = [str(x) for x in cm["classes"]]
            num = range(1, len(labels) + 1)
            exp_rows = [[_e(labels[a])] + [_amt(v) for v in row] for a, row in enumerate(cm["expected"])]
            conf_rows = [[_e(labels[a])] + [_amt(v) for v in row] for a, row in enumerate(cm["confusion"])]
            heads = ["actual \\ predicted"] + labels
            parts.append(f'<div class="ds ds-{i}">' + _row(
                _card(f"Expected counts · {d['label']}", _table(heads, exp_rows, numeric=num),
                      what="Σ weight × P(column class), by actual class (row)"),
                _card(f"Arg-max counts · {d['label']}", _table(heads, conf_rows, numeric=num),
                      what="Weighted count by actual class (row) and arg-max class (column)"),
            ) + "</div>")
    return _open("Performance") + "".join(parts) + "</section>"


def _worst_band(r: dict[str, Any], d: dict[str, Any], feature: str, cls: str | None) -> str:
    e = _ae_entry(d, feature, cls)
    bands = [b for b in (e.get("bands") or []) if b.get("z") is not None] if e else []
    if not bands:
        return ""
    b = max(bands, key=lambda b: abs(b["z"]))
    return f"worst band {_short(str(b['label']), 12)}: {_fixed(b['ae'], 2)}, z {_signed(b['z'])}"


def _factors(r: dict[str, Any], fx: _Radio, classes: _Radio | None) -> str:
    feats = r.get("features") or []
    scored = _scored(r)
    cls0 = _classes(r)[0] if classes is not None else None
    used = [f for f in feats if f.get("used")]
    unused = [f for f in feats if not f.get("used")]
    peak = max((f["importance"] for f in used), default=0.0) or 1.0
    rail = ['<div class="rail card">']

    def subs(f: dict[str, Any]) -> str:
        return "".join(f'<span class="w ds ds-{i}">{_e(_worst_band(r, d, f["feature"], cls0))}</span>'
                       for i, d in enumerate(scored))

    for j, f in enumerate(used + unused):
        if j == len(used) and unused:
            rail.append('<div class="cap">Not used by the model</div>')
        if f.get("used"):
            own = max(f["own_share"], 0.0) / peak * 100
            via = max(f["interaction_share"], 0.0) / peak * 100
            inner = (f'<span class="nm">{_e(f["feature"])}</span>'
                     f'<span class="pc">{_pct(f["importance"], 1)}</span>'
                     f'<span class="bars"><i class="o" style="width:{own:.1f}%"></i>'
                     f'<i class="v" style="width:{via:.1f}%"></i></span>' + subs(f))
            rail.append(fx.label(j, inner, cls="rl", title=str(f["feature"])))
        else:
            inner = f'<span class="nm">{_e(f["feature"])}</span><span class="pc">0%</span>' + subs(f)
            rail.append(fx.label(j, inner, cls="rl unused", title=str(f["feature"])))
    rail.append("</div>")
    smoothing_on = bool(r["fit"]["graduation_tables"]["value"])
    flags_by: dict[str, list[dict[str, Any]]] = {}
    for fl in r["review"]:
        if fl["status"] != "pass" and fl.get("feature") and fl["check"] != "factor_ae":
            flags_by.setdefault(str(fl["feature"]), []).append(fl)
    for fl in r["review"]:
        if fl["status"] != "pass" and fl.get("feature") and fl["check"] == "factor_ae":
            flags_by.setdefault(str(fl["feature"]), []).append(fl)
    panes = []
    for j, f in enumerate(used + unused):
        name = str(f["feature"])
        ow = f.get("one_way") or {}
        facts = []
        if f.get("used"):
            facts.append(f"importance {_pct(f['importance'], 1)}: own table "
                         f"{_pct(f['own_share'], 1)}, interactions {_pct(f['interaction_share'], 1)}")
        else:
            facts.append("not used by the model")
        facts.append(f"{f.get('levels', 0)} levels")
        sm = ow.get("smoothing")
        if ow and smoothing_on:
            facts.append("not smoothed" if not sm else f"smoothed, strength {_num(sm)}")
        partners = sorted({p for u in f.get("interactions") or [] for p in u if p != name})
        if partners:
            facts.append("interacts with " + ", ".join(partners))
        found = [(fl["status"], (f"{fl['dataset']}: " if fl.get("dataset") else "") + str(fl["message"]))
                 for fl in flags_by.get(name, [])]
        issues = "".join(f'<li>{_badge(st)}<span class="t" title="{_e(msg)}">{_e(msg)}</span></li>'
                         for st, msg in found)
        body = [f'<ul class="issues" style="margin-bottom:10px">{issues}</ul>' if issues else ""]

        def one(c: str, k: int | None, name: str = name) -> str:
            chart = factor_chart(r, name, c, k, _W_FACTOR)
            lv = []
            for i, d in enumerate(scored):
                e = _ae_entry(d, name, None if k is None else c)
                if e is None:
                    continue
                labels = cell_labels(e["axis"])
                zs = e.get("z") or [None] * len(e["mass"])
                cols = ["level", "rows", "exposure" if r["model"]["link"] == "log" else "weight",
                        "actual", "expected", "A/E", "z"]
                band_rows = [[_e(b["label"]), _num(b["rows"]), _amt(b["mass"]), _amt(b["actual"]),
                              _amt(b["expected"]), _fixed(b.get("ae")), _signed(b.get("z"))]
                             for b in e.get("bands") or []]
                lvl_rows = [[_e(labels[c2]), _num(e["rows"][c2]), _amt(e["mass"][c2]),
                             _amt(e["actual"][c2]), _amt(e["expected"][c2]), _fixed(e["ae"][c2]),
                             _signed(zs[c2])] for c2, n in enumerate(e["rows"]) if n]
                lv.append(f'<div class="ds ds-{i}">'
                          + "<h4>By band</h4>" + _table(cols, band_rows, numeric=range(1, 7))
                          + "<h4>By level</h4>"
                          + _table(cols, lvl_rows, numeric=range(1, 7), scroll="scroll") + "</div>")
            named = "".join(f'<span class="ds ds-{i}">({_e(str(d["label"]))})</span>'
                            for i, d in enumerate(scored))
            return (chart + "<details><summary>The numbers by band and by level "
                    f"{named}</summary>" + "".join(lv) + "</details>")

        body.append(_per_class(r, classes, one))
        panes.append(f'<div class="fxp {fx.tag(j)}">'
                     + _card(name, "".join(body), what=" · ".join(facts)) + "</div>")
    return (_open("Factors") + '<div class="fxgrid">' + "".join(rail)
            + '<div class="fxv">' + "".join(panes) + "</div></div></section>")


def _interactions_tab(r: dict[str, Any], ix: _Radio) -> str:
    """The interaction tables by share of model variance: a list selects one, drawn as its first
    feature's effect by band of its partner."""
    inters = r.get("interactions") or []
    if not inters:
        return (_open("Interactions") + _card("No interaction tables",
                '<p class="note">The model is additive: every table is a one-way table.</p>')
                + "</section>")
    cls = _classes(r)[0]
    more = len(inters) - _INTERACTION_ROWS
    inters = inters[:_INTERACTION_ROWS]
    peak = max((i.get("share") or 0.0) for i in inters) or 1.0

    def span(pair: Any, mult: bool) -> str:
        if not pair:
            return "—"
        return (f"{_mult(pair[0])} to {_mult(pair[1])}" if mult
                else f"{_num(pair[0], 2)} to {_num(pair[1], 2)}")

    rail = ['<div class="rail card">']
    panes = []
    for j, inter in enumerate(inters):
        name = " × ".join(inter["features"])
        mult = bool(inter.get("multiplicative"))
        width = (inter.get("share") or 0.0) / peak * 100
        rail.append(ix.label(j, f'<span class="nm">{_e(name)}</span>'
                             f'<span class="pc">{_pct(inter.get("share"), 2)}</span>'
                             f'<span class="bars"><i class="o" style="width:{width:.1f}%"></i></span>'
                             f'<span class="w">90% of exposure: {_e(span(inter.get("spread"), mult))}'
                             "</span>", cls="rl pair", title=name))
        facts = [f"{_pct(inter.get('share'), 2)} of the model variance",
                 f"effect {span(inter.get('range'), mult)}; "
                 f"{span(inter.get('spread'), mult)} for 90% of exposure"]
        sm = inter.get("smoothing")
        if sm is not None:
            facts.append("not smoothed" if not sm else f"smoothed, strength {_num(sm)}")
        third = inter["features"][2:] if len(inter["features"]) > 2 else []
        note = (f'<p class="note">Averaged over {_e(", ".join(third))} by exposure. ' if third
                else '<p class="note">') + "The numbers are in the Tables tab.</p>"
        if r["model"]["multiclass"]:
            note = note.replace("</p>", f" Class {_e(cls)}.</p>")
        panes.append(f'<div class="fxp {ix.tag(j)}">' + _card(
            name, interaction_chart(r, inter, cls, _W_FACTOR) + note, what=" · ".join(facts))
            + "</div>")
    if more > 0:
        rail.append(f'<div class="cap">{more} smaller interaction tables are in the Tables tab</div>')
    rail.append("</div>")
    return (_open("Interactions") + '<div class="fxgrid">' + "".join(rail) + '<div class="fxv">'
            + "".join(panes) + "</div></div></section>")


def _tones(eff: Sequence[float], support: Sequence[float], mult: bool) -> list[str]:
    """A light tint per cell: green above no effect, yellow below, in five steps of the effect's
    size against the table's largest (on the log scale for a multiplier; steps spaced on a
    square root, so moderate effects show); none for a cell without exposure."""
    dev = [(math.log(v) if v > 0 else 0.0) if mult else v for v in eff]
    live = [abs(d) for d, s in zip(dev, support) if s > 0]
    peak = max(live, default=0.0)
    out = []
    for d, s in zip(dev, support):
        step = (min(5, math.ceil(math.sqrt(abs(d) / peak) * 5 - 1e-9)) if peak > 0 and s > 0
                else 0)
        out.append(("g" if d > 0 else "y") + str(step) if step > 0 else ("muted" if s <= 0 else ""))
    return out


def _table_numbers(t: dict[str, Any], link: str) -> str:
    """A table as numbers, tinted green above no effect and yellow below: one-way tables as rows,
    two-way as a grid (rows the first feature), higher orders as one row per cell; the support
    beside or below."""
    eff, mult, kind = _effects(t, link)
    n_cells = len(eff)
    support = t.get("support") or [0.0] * n_cells
    tone = _tones(eff, support, mult)
    labels = [cell_labels(a) for a in t["axes"]]
    order = len(t["feature_names"])

    def fmt(v: float) -> str:
        return _fixed(v, 3) if mult else _num(v, 4)

    if order == 1:
        rows = [[_e(labels[0][i]), fmt(eff[i]), _amt(support[i])] for i in range(n_cells)]
        return _table(["level", kind, "support"], rows, numeric=(1, 2), scroll="scroll",
                      tones=[["", tone[i], ""] for i in range(n_cells)], cls="tinted")
    if order == 2:
        n0, n1 = int(t["shape"][0]), int(t["shape"][1])
        a0, a1 = t["axes"]
        head = [f"{a0['name']} \\ {a1['name']}"] + labels[1]
        grid = [[_e(labels[0][i])] + [fmt(eff[i * n1 + j]) for j in range(n1)] for i in range(n0)]
        mass = [[_e(labels[0][i])] + [_amt(support[i * n1 + j]) for j in range(n1)] for i in range(n0)]
        cols = range(1, n1 + 1)
        grid_tones = [[""] + [tone[i * n1 + j] for j in range(n1)] for i in range(n0)]
        return (_table(head, grid, numeric=cols, scroll="scroll", tones=grid_tones, cls="tinted")
                + "<details><summary>Support</summary>"
                + _table(head, mass, numeric=cols, scroll="scroll") + "</details>")
    shape = [int(d) for d in t["shape"]]
    rows = []
    for k in range(n_cells):
        idx, rest = [], k
        for d in reversed(shape):
            idx.append(rest % d)
            rest //= d
        idx.reverse()
        rows.append([_e(labels[a][idx[a]]) for a in range(order)] + [fmt(eff[k]), _amt(support[k])])
    return _table([str(a["name"]) for a in t["axes"]] + [kind, "support"], rows,
                  numeric=(order, order + 1), scroll="scroll", cls="tinted",
                  tones=[[""] * order + [tone[k], ""] for k in range(n_cells)])


def _tables_tab(r: dict[str, Any], tt: _Radio) -> str:
    """Every deployed table: a list (share of model variance, effect range, smoothing) whose
    rows select the table shown beside it as numbers."""
    link = r["model"]["link"]
    banks = _banks(r)
    grad = {tuple(g["features"]): g.get("alpha") for g in (r["fit"]["graduation_tables"]["value"] or [])}
    items = [(label, t) for label, bank in banks.items()
             for t in sorted(bank.get("tables", []), key=lambda t: -(t.get("sobol") or 0.0))]
    rows, views, css_rows = [], [], []
    budget = _TABLE_CELLS
    for j, (label, t) in enumerate(items[:_TABLE_ROWS]):
        eff, mult, kind = _effects(t, link)
        sup = t.get("support") or []
        live = [eff[i] for i in range(len(eff)) if i < len(sup) and sup[i] > 0] or eff
        rng = ((f"{_mult(min(live))}–{_mult(max(live))}" if mult
                else f"{_num(min(live), 3)} to {_num(max(live), 3)}") if live else "—")
        sm = grad.get(tuple(t["feature_names"]))
        name = " × ".join(t["feature_names"]) + (f" · class {label}" if label else "")
        n_cells = int(math.prod(int(d) for d in t["shape"]))
        cells = [_e(name), _num(n_cells), _pct(t.get("sobol"), 2), _e(rng),
                 "—" if sm is None else _num(sm)]
        rows.append([tt.label(j, c) for c in cells])
        css_rows.append(f"ttr-{j}")
        what = (f"{_pct(t.get('sobol'), 2)} of the model variance · {kind}, "
                + ("multiplying the base rate" if mult else "added on the link scale"))
        if n_cells > budget:
            body = '<p class="note">Past the page budget; the numbers are in the JSON report.</p>'
        else:
            budget -= n_cells
            body = _table_numbers(t, link)
        views.append(f'<div class="tv {tt.tag(j)}">' + _card(name, body, what=what) + "</div>")
    notes = []
    if len(items) > _TABLE_ROWS:
        notes.append(f"{len(items) - _TABLE_ROWS} more tables are in the JSON report.")
    factored = sum(len(b.get("factored") or []) for b in banks.values())
    if factored:
        notes.append(f"{factored} factored effect(s), which have no cell grid, are in the JSON report.")
    base = _base_rate(r)
    how = (f"Prediction = base rate {_num(base, 5)} × a relativity from each table × exposure"
           if base is not None else "Prediction = intercept + one value from each table, on the "
           "link scale, through the inverse link")
    lst = _card("Deployed tables", _table(["table", "cells", "share", "range", "smoothing"],
                                          rows, numeric=(1, 2, 3, 4), row_class=css_rows)
                + "".join(f'<p class="note">{_e(n)}</p>' for n in notes), what=how, cls="tlist")
    return (_open("Tables") + '<div class="ttgrid">' + lst
            + '<div class="tview">' + "".join(views) + "</div></div></section>")


def _ranked(r: dict[str, Any], pd: dict[str, Any]) -> str:
    rep, path, scores, best, total = pd["rep"], pd["path"], pd["scores"], pd["best"], pd["total"]
    low = {tuple(u) for u in path.get("dropped_low_share") or []}
    floor = path.get("min_variance_share") or 0.0
    deployed = {tuple(u) for u in rep.get("deployed") or rep.get("kept") or []}
    names = ("main effects (always kept)", "kept", "kept, not deployed", "removed: below the floor",
             "dropped: past the cutoff")
    groups: dict[str, list[list[str]]] = {g: [] for g in names}
    shares: dict[str, float] = {g: 0.0 for g in names}
    cum = 0.0

    def add(group: str, rank: str, u: tuple[int, ...], step: str) -> None:
        nonlocal cum
        sh = _share(scores.get(u, {})) / total
        cum += sh
        shares[group] += sh
        note = ("below the floor; kept as part of a larger kept table"
                if floor and group == "kept" and sh < floor else "")
        groups[group].append([rank, _e(_feature_names(r, u)), str(len(u)), _pct(sh, 3),
                              _pct(cum, 2), step, _e(note)])

    seq = [(u, step) for u, step in _path_order(pd) if step > 0]
    for u in pd["mains"]:
        add(names[0], "—", u, "0")
    known = bool(path.get("sequence"))
    for rank, (u, step) in enumerate(seq[:_RANKED_ROWS], start=1):
        group = (names[3] if u in low else names[4] if step > best
                 else names[1] if u in deployed else names[2])
        add(group, str(rank), u, str(step) if known else "—")
    parts = []
    for g in names:
        if groups[g]:
            parts.append(f"<details><summary><b>{_e(g.capitalize())}</b> · {len(groups[g])} "
                         f"table(s), {_pct(shares[g], 2)} of the candidates' variance</summary>"
                         + _table(["rank", "table", "order", "variance share", "cumulative", "step",
                                   "note"], groups[g], numeric=(2, 3, 4, 5)) + "</details>")
    if len(seq) > _RANKED_ROWS:
        parts.append(f'<p class="note">{len(seq) - _RANKED_ROWS} more in the JSON report.</p>')
    stages = {k: rep[k] for k in ("table_budget", "box_budget", "banding") if k in rep}
    if stages:
        parts.append("<details><summary>After selection: " + _e(", ".join(stages)) + "</summary>"
                     + _kv(stages) + "</details>")
    return "".join(parts)


def _build(r: dict[str, Any]) -> str:
    out = [_open("Build")]
    pd = _path_data(r)
    fit = r["fit"]["pruning"]
    if pd is not None:
        best = pd["best"]
        out.append(_row(
            _card(_gain_takeaway(r), gain_chart(r, _W_HALF, 280),
                  what="Share of the out-of-bag gain as ranked tables join the bank"),
            _card("Every candidate table, in the order the path adds it"
                  if pd["path"].get("sequence") else "Every candidate table: kept, then the rest",
                  bank_chart(r, _W_HALF, 280),
                  what="Each column: the table's share of the candidates' variance (log scale)")))
        out.append(_card("Ranked tables", _ranked(r, pd),
                         what=(f"Cutoff: step {best} of {len(pd['path']['sizes']) - 1}, a bank of "
                               f"{pd['bank'][best]} tables")))
    elif fit["available"] and fit["value"]:
        rep = fit["value"]
        body = (f'<p class="note">Selector {_e(rep.get("selector"))}: kept '
                f'{len(rep.get("kept") or [])} tables.</p>'
                + ("<details><summary>Table scores</summary>" + _kv(rep.get("table_scores") or [])
                   + "</details>" if rep.get("table_scores") else ""))
        out.append(_card("Pruning", body))
    else:
        out.append(_card("Pruning", f'<p class="note">{_e(fit.get("reason") or "Not recorded")}.</p>'))
    mono = _monotone(r)
    if mono:
        out.append(mono)
    out.append(_row(_graduation(r), _training(r)))
    out.append("</section>")
    return "".join(out)


def _monotone(r: dict[str, Any]) -> str:
    """One row per constrained feature: how far the deployed tables were moved back onto its
    sign after pruning (empty when the fit has no monotone constraints)."""
    fit = r["fit"]["pruning"]
    rep = (fit["value"] or {}).get("monotone") if fit["available"] else None
    if not rep:
        return ""
    names = list(r["model"]["native_feature_order"])
    rows = []
    for f in rep["features"]:
        j = int(f["feature"])
        name = names[j] if j < len(names) else str(j)
        if not f["tables"]:
            result = "not used by the model"
        elif f["worst_step_before"] >= -1e-9:
            result = "already monotone"
        else:
            clipped = len(f["clipped"])
            result = (f"repaired: main effect moved by up to {_pct(math.expm1(f['main_change']))}"
                      + (f", {clipped} interaction table(s) clipped" if clipped else ""))
        rows.append([_e(name), "increasing" if f["sign"] > 0 else "decreasing", _e(result)])
    return _card("Monotone constraints", _table(["feature", "direction", "result"], rows),
                 what="The trees enforce each sign; after pruning, the deployed tables are "
                      "projected back onto it")


def _graduation(r: dict[str, Any]) -> str:
    grad = r["fit"]["graduation"]
    gt = r["fit"]["graduation_tables"]["value"] or []
    body = []
    if grad["available"] and grad["value"]:
        g = grad["value"]
        smoothed = sum(1 for x in gt if (x.get("alpha") or 0) > 0) if gt else g.get("n_smoothed")
        evidence = ("adopted without holdout evidence" if g.get("evidence") == "none"
                    else f"evidence: {g.get('evidence')}")
        body.append(f'<p class="note">{_e(smoothed)} table(s) smoothed; {_e(evidence)}.</p>')
    if gt:
        ordered = sorted(gt, key=lambda x: (-(x.get("alpha") or 0.0), len(x["features"]), x["features"]))
        rows = [[_e(" × ".join(x["features"])), _num(x.get("alpha")),
                 "rejected" if x.get("rejected") else ("smoothed" if (x.get("alpha") or 0) > 0
                                                       else "as fitted")] for x in ordered]
        body.append(_table(["table", "strength", "result"], rows, numeric=(1,), scroll="short"))
    if not body:
        body.append(f'<p class="note">{_e(grad.get("reason") or "Graduation off")}.</p>')
    return _card("Graduation", "".join(body), what="Whittaker–Henderson smoothing of the deployed tables")


def _training(r: dict[str, Any]) -> str:
    fit = r["fit"]
    per_bag = fit["n_trees_per_bag"]["value"] or []
    reasons = fit["stopping_reason_per_bag"]["value"] or []
    es = fit["early_stopping"]
    rows_used = ("the eval_set" if es["eval_set"]
                 else f"a validation_fraction={es['validation_fraction']} holdout")
    if per_bag:
        what = (f"Trees kept per bag {min(per_bag):,}–{max(per_bag):,} over {len(per_bag)} bags; "
                f"early stopping on {rows_used}, patience {es['early_stopping_rounds']}")
    else:
        what = f"Early stopping on {rows_used}, patience {es['early_stopping_rounds']}"
    bags = ""
    if per_bag:
        rows = [[str(i + 1), _num(n), _e(reasons[i] if i < len(reasons) else "—")]
                for i, n in enumerate(per_bag)]
        bags = ("<details><summary>Bags</summary>"
                + _table(["bag", "trees kept", "stopping reason"], rows, numeric=(1,)) + "</details>")
    return _card("Training", training_chart(r, _W_HALF, 240) + bags, what=what)


def _checks(r: dict[str, Any]) -> str:
    order = {"critical": 0, "warning": 1, "info": 2, "pass": 3}
    flags = sorted(r["review"], key=lambda f: (order.get(f["status"], 9), f["check"]))
    th = r["thresholds"]

    def rows(items: list[dict[str, Any]]) -> str:
        body = [[_badge(f["status"]), _e(f["check"]),
                 _e(" · ".join(str(x) for x in (f.get("dataset"), f.get("feature"), f.get("class"))
                               if x is not None)),
                 _e(f["message"]), _num(f["value"]) if not isinstance(f["value"], str) else _e(f["value"]),
                 _num(f["threshold"])] for f in items]
        return _table(["status", "check", "scope", "finding", "value", "threshold"], body,
                      numeric=(4, 5))

    open_items = [f for f in flags if f["status"] != "pass"]
    passed = [f for f in flags if f["status"] == "pass"]
    th_rows = [[_e(k), "off" if v is None else _num(v), _e(r["threshold_descriptions"].get(k, ""))]
               for k, v in th.items()]
    review = _card("Review", (rows(open_items) if open_items else '<p class="ok">No open findings.</p>')
                   + (f"<details><summary>{len(passed)} passed checks</summary>{rows(passed)}</details>"
                      if passed else "")
                   + "<details><summary>Thresholds in force</summary>"
                   + _table(["threshold", "value", "meaning"], th_rows, numeric=(1,)) + "</details>",
                   what="Flags for a reviewer, not a verdict; set the thresholds with the thresholds "
                   "argument", extra=_counts(r))
    return (_open("Checks") + review
            + _row(_data_card(r), _params_card(r)) + _row(_model_card(r), _how_card(r)) + "</section>")


def _data_card(r: dict[str, Any]) -> str:
    th = r["thresholds"]
    scored = _scored(r)
    out = []
    if scored:
        if all(m["rows"] == 0 for d in scored for m in d["missing"]):
            out.append('<p class="ok">No missing values in any feature.</p>')
        else:
            names = [m["feature"] for m in scored[0]["missing"]]
            mrows = [[_e(n)] + [f'{_num(d["missing"][j]["rows"])} ({_pct(d["missing"][j]["mass_share"], 2)})'
                                for d in scored] for j, n in enumerate(names)]
            out.append("<h4>Missing values: rows (share of exposure)</h4>"
                       + _table(["feature"] + [str(d["label"]) for d in scored], mrows,
                                numeric=range(1, len(scored) + 1)))
    unseen = [(d["label"], u) for d in r["datasets"] for u in d.get("unseen") or []]
    if unseen:
        urows = [[_e(lab), _e(u["feature"]), _e(u["value"]), _num(u["rows"]), _num(u.get("mass"))]
                 for lab, u in unseen[:_RANKED_ROWS]]
        out.append("<h4>Unseen categorical values</h4>"
                   + _table(["dataset", "feature", "value", "rows", "exposure"], urows, numeric=(3, 4)))
    else:
        out.append('<p class="ok">No unseen categorical values.</p>')
    pairs = r.get("drift") or []
    if pairs:
        names = [str(d["feature"]) for d in pairs[0]["features"]]
        worst = {n: max((p["features"][j]["psi"] or 0.0) for p in pairs) for j, n in enumerate(names)}
        top_name, top_psi = max(worst.items(), key=lambda t: t[1])
        warn = th.get("psi_warn")
        cls = "ok" if warn is None or top_psi <= warn else "note"
        out.append(f'<p class="{cls}">Largest PSI {_num(top_psi, 3)} ({_e(top_name)})'
                   + (f"; warning above {_num(warn)}" if warn is not None else "") + ".</p>")
        prow = [[_e(n)] + [_num(p["features"][j]["psi"], 3) for p in pairs]
                for j, n in sorted(enumerate(names), key=lambda t: -worst[t[1]])]
        out.append("<details><summary>Population stability by feature</summary>"
                   + _table(["feature"] + [f"{p['reference']} → {p['compared']}" for p in pairs], prow,
                            numeric=range(1, len(pairs) + 1)) + "</details>")
    return _card("Data", "".join(out), what="PSI compares the supplied datasets; it is not a forecast")


def _params_card(r: dict[str, Any]) -> str:
    m = r["model"]
    set_params = [p for p in m["params"] if p["non_default"]]
    defaults = [p for p in m["params"] if not p["non_default"]]

    def rows(ps: list[dict[str, Any]]) -> list[list[str]]:
        return [[_e(p["name"]), _short_val(p["value"]), _short_val(p["default"])] for p in ps]

    body = (_table(["parameter", "value", "default"], rows(set_params)) if set_params
            else '<p class="note">Every parameter is at its default.</p>')
    body += (f"<details><summary>{len(defaults)} parameters at their default</summary>"
             + _table(["parameter", "value", "default"], rows(defaults)) + "</details>")
    return _card("Parameters set", body)


def _model_card(r: dict[str, Any]) -> str:
    m = r["model"]
    base = _base_rate(r)
    return _card("Model", _kv({
        "columns": ", ".join(str(c) for c in m["columns"]), "classes": m.get("classes"),
        "seed": m.get("seed"),
        "base rate (exp of the intercept)": _num(base, 5) if base is not None else None,
        "exposure column": m.get("exposure_column"), "weight column": m.get("weight_column"),
        "requires": m.get("requires"), "metadata": m.get("metadata"),
        "report version": r["report_version"], "generated": r.get("generated_at"),
        "prediction units": r.get("prediction_units"),
    }))


def _how_card(r: dict[str, Any]) -> str:
    items = [
        ("Deviance", "the objective's mean unit deviance of y against each row's expected value "
         "(exposure and offset included), weighted by sample weight."),
        ("Deviance explained", "1 − deviance / null deviance; the null model is the best constant "
         "on the same rows, with the same exposure and offset."),
        ("Gini", f"the Lorenz Gini of actual against predicted {_rate_name(r)}, exposure on the "
         "x-axis; tied predictions form one segment. Gini / perfect divides by the Gini of ranking "
         "by the actual rate."),
        ("Deciles", "ten bands of equal exposure by ascending predicted rate (tied predictions stay "
         "in one band); σ from the objective's variance function and the Pearson dispersion."),
        ("Base rate", "exp of the intercept: a normalisation (each table averages to no effect), "
         "not the observed mean rate."),
        ("Share of model variance", "a deployed table's Sobol index. Feature importance splits each "
         "interaction's share evenly over its features."),
        ("Factor bands", "up to ten bands per feature of equal train exposure (missing values "
         "alone, the largest categorical levels plus 'other'), the same bands for every dataset; "
         "z = (actual − expected) / σ."),
        ("Thin cells", "one-way cells holding less than thin_cell_mass of the table's exposure."),
        ("SHAP", "a feature's exact Shapley value per policy: its one-way table plus an equal share "
         "of every interaction table it is in (the tables are centred, so this split is exact). The "
         "Factors tab shows its exposure-weighted mean and 10th to 90th percentile per level on the "
         "train data: how much the other factors move the feature's effect. A partial dependence "
         "plot would repeat the one-way table: each interaction averages to no effect over its "
         "partner's exposure."),
        ("Interactions", "a table's effect for 90% of the exposure is the support-weighted 5th to "
         "95th percentile of its cells. The chart adds the table, averaged over bands of equal "
         "exposure of the partner, to the first feature's one-way table."),
        ("Pruning", "the path adds tables in order of variance share (respecting heredity) and "
         "scores each nested bank on out-of-bag rows. The cutoff is the later of the first bank "
         "reaching prune_path_fraction of the gain and the first within prune_path_tolerance of "
         "the minimum; the variance floor then removes tables below it that no kept table is built "
         "on. These are selection statistics; the final model's performance is on Performance."),
        ("Training curve", "each bag runs past its best round by the patience, then keeps its best; "
         "late rounds average fewer bags."),
        ("PSI", "compares the supplied datasets (train against each later one) on at most ten bins "
         "per feature; not a forecast of future drift."),
    ]
    rows = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in items)
    caveat = f'<p class="note">{_e(r.get("caveat"))}</p>' if r.get("caveat") else ""
    return _card("How to read this report", caveat + "<details><summary>Definitions</summary>"
                 f'<table class="kv">{rows}</table></details>')


# --------------------------------------------------------------------------- entry point ------

def report_html(report: dict[str, Any], *, name: str | None = None) -> str:
    """Render a ``pricing_report`` (version 2) as one self-contained HTML page.

    ``name`` overrides the report's own name in the heading. The page holds no scripts and loads
    nothing from the network, so it can be logged as an artifact (for example to MLflow) and
    opened offline. Raises ``ValueError`` for a report of an unsupported version.
    """
    version = report.get("report_version") if isinstance(report, dict) else None
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"report_html renders report_version {_SUPPORTED_VERSIONS}, got {version!r}; "
            "build the report with pricing_report from this t-boost"
        )
    r = report
    scored = _scored(r)
    multi = bool(r["model"]["multiclass"])
    feats = r.get("features") or []
    n_tables = sum(len(b.get("tables", [])) for b in _banks(r).values())
    tabs = _Radio("tb", len(_TAB_NAMES))
    ds = _Radio("ds", len(scored), len(scored) - 1)
    classes = _Radio("cl", len(_classes(r))) if multi else None
    fx = _Radio("fx", len(feats))
    ix = _Radio("ix", min(len(r.get("interactions") or []), _INTERACTION_ROWS))
    tt = _Radio("tt", min(n_tables, _TABLE_ROWS))
    switches = []
    if len(scored) > 1:
        switches.append('<div class="switch" role="group" aria-label="dataset"><span class="cap">'
                        "dataset</span>" + "".join(ds.label(i, _e(d["label"])) for i, d in enumerate(scored))
                        + "</div>")
    if classes is not None:
        switches.append('<div class="switch" role="group" aria-label="class"><span class="cap">class'
                        "</span>" + "".join(classes.label(i, _e(c)) for i, c in enumerate(_classes(r)))
                        + "</div>")
    bar = ('<nav class="bar"><div class="tabs">'
           + "".join(tabs.label(i, _e(t)) for i, t in enumerate(_TAB_NAMES))
           + f'</div><div class="switches">{"".join(switches)}</div></nav>')
    body = "".join([
        _overview(r, tabs, classes), _performance(r, classes), _factors(r, fx, classes),
        _interactions_tab(r, ix), _tables_tab(r, tt), _build(r), _checks(r),
    ])
    switch_on = "background:var(--wash);color:var(--ink);box-shadow:inset 0 -2px 0 var(--data)"
    css = (_CSS % {"light": _LIGHT, "dark": _DARK, "content": _CONTENT, "gap": _GAP,
                   "font": _FONT, "rail": _RAIL, "list": _LIST}
           + tabs.css("background:var(--surface);color:var(--ink);border-color:var(--border)")
           + ds.css(switch_on) + (classes.css(switch_on) if classes is not None else "")
           + fx.css("background:var(--wash);box-shadow:inset 3px 0 0 var(--model)")
           + ix.css("background:var(--wash);box-shadow:inset 3px 0 0 var(--model)")
           + tt.css("") + "".join(f"#tt-{j}:checked~.body .ttr-{j} td{{background:var(--wash)}}"
                                  for j in range(tt.n))
           + _dataset_css(len(scored)))
    inputs = (tabs.inputs() + ds.inputs() + (classes.inputs() if classes is not None else "")
              + fx.inputs() + ix.inputs() + tt.inputs())
    title = name or r.get("name") or f"{r['model']['class']} report"
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{_e(title)}</title><style>{css}</style></head><body>"
            f'<div class="app">{inputs}{_header(r, name)}{bar}<main class="body">{body}</main>'
            "</div></body></html>")
