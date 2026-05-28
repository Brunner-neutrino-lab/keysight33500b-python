"""
ks33500b/gui.py

NiceGUI control panel for the Keysight 33500B series waveform generator
(33511B / 33512B / 33521B / 33522B and the A variants).

Mirrors the Rigol DG1022 panel ([../rigoldg1022-python/dg1022/gui.py]) in
look and tab structure, but uses the 33500B-native parameter set:

  - Pulse: period, width (s), rise/fall edges — not the Rigol's
    duty-cycle.
  - Square: explicit duty-cycle override after APPLy.
  - Ramp:   explicit symmetry override after APPLy.

Each channel card has a live waveform preview — a small matplotlib plot
that regenerates from the on-screen parameters every time the user touches
a field, so the operator sees the *shape* they're about to send before
clicking "apply".  Pure-Python preview; no instrument round-trip.

Standalone-or-embedded pattern is the same as the other instrument GUIs:

  - Standalone (`python -m ks33500b.gui`): a connection card creates and
    owns its own KS33500BController.
  - Embedded (`build_page(get_controller=..., show_connection=False)`):
    a parent app (the DAQ web shell) passes a getter for a shared
    controller.

Tabs: connection (standalone only), ch1, ch2, burst, sweep, arbitrary.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

import numpy as np
from nicegui import ui

from .controller import KS33500BController
from .driver     import DEFAULT_VISA


# ---------------------------------------------------------------------------
# Style — xsphere/DAQ palette (matches the Rigol DG1022 panel)
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg:#11151c; --panel:#1b2230; --panel2:#232c3d;
  --fg:#dde3ee; --mut:#8a93a6;
  --ok:#3fb950; --warn:#d29922; --bad:#f85149; --acc:#58a6ff;
  --line:#2d3648;
}
html, body, .nicegui-content { background:var(--bg) !important; color:var(--fg);
  font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin:0; }
.pill { padding:.15rem .55rem; border-radius:999px; font-size:.78rem;
  font-weight:600; white-space:nowrap; display:inline-flex; align-items:center; gap:.3rem; }
.pill.ok   { background:rgba(63,185,80,.18);  color:var(--ok); }
.pill.bad  { background:rgba(248,81,73,.18);  color:var(--bad); }
.pill.warn { background:rgba(210,153,34,.18); color:var(--warn); }
.pill.mut  { background:rgba(138,147,166,.15);color:var(--mut); }
.q-card, .ks-card {
  background:var(--panel) !important; color:var(--fg) !important;
  border:1px solid var(--line); border-radius:10px;
  box-shadow:none !important; padding:.55rem .85rem .7rem !important;
}
.ks-card h2 { font-size:.92rem; margin:.05rem 0 .45rem; color:var(--acc);
  font-weight:600; letter-spacing:.3px; }
.q-btn { background:var(--panel2) !important; color:var(--fg) !important;
  border:1px solid var(--line) !important; border-radius:6px !important;
  box-shadow:none !important; padding:.18rem .65rem !important;
  min-height:32px !important; text-transform:none !important; }
.q-btn:hover { border-color:var(--acc) !important; }
.q-btn[data-q-color="primary"], .q-btn.bg-primary {
  background:var(--acc) !important; color:#08111f !important;
  border-color:var(--acc) !important; font-weight:600 !important; }
.q-btn[data-q-color="negative"], .q-btn.bg-negative {
  background:transparent !important; color:var(--bad) !important;
  border-color:var(--bad) !important; }
.q-field__control, .q-field--filled .q-field__control {
  background:var(--panel2) !important; border:1px solid var(--line) !important;
  border-radius:6px !important; min-height:32px !important; color:var(--fg) !important; }
.q-field__label, .q-field__native, .q-field input { color:var(--fg) !important; }
.q-field__label { color:var(--mut) !important; }
.q-field--filled .q-field__control:before,
.q-field--filled .q-field__control:after { display:none !important; }
.q-tab { color:var(--mut) !important; text-transform:none !important; }
.q-tab--active { color:var(--acc) !important; }
.q-tab__indicator { background:var(--acc) !important; }
.q-log, .nicegui-log { background:var(--panel2) !important; color:var(--fg) !important;
  border:1px solid var(--line); border-radius:6px;
  font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.82rem; }
.num { font-variant-numeric:tabular-nums; }
"""


async def _in_thread(fn, *a, **kw):
    return await asyncio.to_thread(fn, *a, **kw)


# ---------------------------------------------------------------------------
# Waveform preview — pure-Python shape generator
# ---------------------------------------------------------------------------

def generate_preview(fn: str,
                      freq:        float,
                      amplitude:   float,
                      offset:      float,
                      *,
                      duty_cycle:  float = 50.0,    # square wave, %
                      symmetry:    float = 100.0,   # ramp, %
                      period_s:    Optional[float] = None,
                      width_s:     Optional[float] = None,
                      n_periods:   int = 3,
                      n_samples:   int = 1500,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Generate (t, v) samples for a quick on-screen preview of the
    selected waveform.  Independent of the instrument; uses the same
    Vpp / offset / shape parameters the operator has on screen.
    Returns time in seconds and voltage in volts.
    """
    if fn == "DC":
        t = np.linspace(0.0, 1.0, 100)
        return t, np.full_like(t, offset)
    f = max(freq, 1e-9)
    if fn == "PULS" and period_s and period_s > 0:
        total_T = n_periods * period_s
    else:
        total_T = n_periods / f
    t = np.linspace(0.0, total_T, n_samples)
    A_half = amplitude / 2.0

    if fn == "SIN":
        v = A_half * np.sin(2 * np.pi * f * t)
    elif fn == "SQU":
        duty = max(0.001, min(0.999, duty_cycle / 100.0))
        phase = (t * f) % 1.0
        v = np.where(phase < duty, A_half, -A_half)
    elif fn == "RAMP":
        sym = max(0.001, min(0.999, symmetry / 100.0))
        phase = (t * f) % 1.0
        v = np.where(
            phase < sym,
            A_half * (2.0 * phase / sym - 1.0),
            A_half * (1.0 - 2.0 * (phase - sym) / (1.0 - sym)),
        )
    elif fn == "PULS":
        period = period_s if (period_s and period_s > 0) else (1.0 / f)
        width = width_s if (width_s and 0 < width_s < period) else (period * 0.5)
        phase = t % period
        v = np.where(phase < width, A_half, -A_half)
    elif fn == "NOIS":
        rng = np.random.default_rng(42)
        v = rng.normal(0.0, A_half / 3.0, len(t))
    else:   # "ARB" or anything else — placeholder shape
        v = A_half * np.sin(2 * np.pi * f * t) ** 3
    return t, v + offset


def _style_dark(fig, ax) -> None:
    fig.patch.set_facecolor("#1b2230")
    ax.set_facecolor("#11151c")
    for sp in ax.spines.values():
        sp.set_color("#2d3648")
    ax.tick_params(colors="#8a93a6", labelsize=8)
    ax.grid(True, color="#2d3648", alpha=.4)


def _x_units(T_s: float) -> tuple[float, str]:
    """Return a multiplier and label that scales T_s into a readable unit."""
    if T_s < 1e-5:  return 1e9, "ns"
    if T_s < 1e-2:  return 1e6, "µs"
    if T_s < 1.0:   return 1e3, "ms"
    return 1.0, "s"


# ---------------------------------------------------------------------------
# Per-channel control card
# ---------------------------------------------------------------------------

def _channel_card(ch: int, get_ctrl, log_msg) -> None:
    with ui.card().classes("ks-card w-full"):
        ui.html(f"<h2>ch{ch} waveform</h2>")

        with ui.row().classes("w-full no-wrap gap-3 items-start"):
            # ----- Left column: parameters -----
            with ui.column().classes("gap-1"):
                fn_sel = ui.select(
                    ["SIN", "SQU", "RAMP", "PULS", "NOIS", "DC", "ARB"],
                    value="SIN", label="function",
                ).classes("w-40")
                freq = ui.number(label="frequency (Hz)", value=1000.0,
                                  step=1.0, format="%.6f").classes("w-44 num")
                amp  = ui.number(label="amplitude (Vpp)", value=1.0,
                                  step=0.1, format="%.4f").classes("w-44 num")
                offs = ui.number(label="offset (V)", value=0.0,
                                  step=0.1, format="%.4f").classes("w-44 num")
                phase= ui.number(label="phase (deg)", value=0.0,
                                  step=1.0, format="%.2f").classes("w-44 num")
                load = ui.select(["50", "INF (High-Z)"], value="50",
                                  label="output load").classes("w-44")

                # 33500B-specific: pulse uses width (s) + edges, not duty cycle.
                with ui.expansion("pulse parameters").classes("w-full"):
                    pulse_per  = ui.number(label="period (s)",  value=1e-3,
                                            step=1e-6, format="%.9f").classes("w-44 num")
                    pulse_wid  = ui.number(label="width (s)",   value=5e-4,
                                            step=1e-6, format="%.9f").classes("w-44 num")
                    pulse_rise = ui.number(label="rise time (s)", value=1e-8,
                                            step=1e-9, format="%.9f").classes("w-44 num")
                    pulse_fall = ui.number(label="fall time (s)", value=1e-8,
                                            step=1e-9, format="%.9f").classes("w-44 num")

                with ui.expansion("square duty cycle").classes("w-full"):
                    sq_duty = ui.number(label="duty (%)", value=50.0,
                                         step=1.0, min=0.01, max=99.99) \
                        .classes("w-44 num")

                with ui.expansion("ramp symmetry").classes("w-full"):
                    rm_sym = ui.number(label="symmetry (%)", value=100.0,
                                        step=1.0, min=0.0, max=100.0) \
                        .classes("w-44 num")

            # ----- Right column: live preview plot -----
            with ui.column().classes("flex-1"):
                ui.html('<div style="font-size:.7rem; letter-spacing:.4px; '
                        'text-transform:uppercase; color:var(--mut); '
                        'margin-bottom:.25rem">preview</div>')
                preview = ui.matplotlib(figsize=(5.4, 2.2)).classes("w-full")
                pv_ax = preview.figure.add_subplot(111)
                _style_dark(preview.figure, pv_ax)
                preview.figure.tight_layout()

        out_pill = ui.html(
            f'<span class="pill mut" style="margin-top:.4rem; display:inline-block">'
            f'ch{ch} output: off</span>'
        )

        # ----- Refresh the preview from the current values -----
        def _refresh_preview(*_):
            fn = str(fn_sel.value)
            try:
                t, v = generate_preview(
                    fn        = fn,
                    freq      = float(freq.value or 0.0),
                    amplitude = float(amp.value  or 0.0),
                    offset    = float(offs.value or 0.0),
                    duty_cycle= float(sq_duty.value or 50.0),
                    symmetry  = float(rm_sym.value  or 100.0),
                    period_s  = float(pulse_per.value or 0.0) or None,
                    width_s   = float(pulse_wid.value or 0.0) or None,
                )
            except Exception:
                return
            pv_ax.clear()
            _style_dark(preview.figure, pv_ax)
            mul, unit = _x_units(t[-1] if len(t) else 0.001)
            pv_ax.plot(t * mul, v, lw=1.3, color="#58a6ff")
            pv_ax.set_xlabel(f"time ({unit})", color="#8a93a6", fontsize=9)
            pv_ax.set_ylabel("V", color="#8a93a6", fontsize=9)
            preview.figure.tight_layout()
            preview.update()

        for w in (fn_sel, freq, amp, offs,
                  sq_duty, rm_sym, pulse_per, pulse_wid):
            w.on("update:model-value", _refresh_preview)
        _refresh_preview()

        # ----- Apply / output buttons -----
        def _do_apply():
            c = get_ctrl()
            if c is None:
                log_msg(f"ch{ch} apply: not connected"); return
            try:
                ld = "INF" if "INF" in str(load.value) else 50.0
                c.set_load(ld, channel=ch)
                fn = str(fn_sel.value)
                f = float(freq.value)
                a = float(amp.value)
                o = float(offs.value)
                p = float(phase.value)
                if fn == "SIN":
                    c.apply_sine(f, a, o, p, channel=ch)
                elif fn == "SQU":
                    c.apply_square(f, a, o, p,
                                    duty_cycle=float(sq_duty.value), channel=ch)
                elif fn == "RAMP":
                    c.apply_ramp(f, a, o, p,
                                  symmetry=float(rm_sym.value), channel=ch)
                elif fn == "PULS":
                    c.apply_pulse(f, a, o, p, channel=ch)
                    c.configure_pulse(
                        period_s   = float(pulse_per.value),
                        width_s    = float(pulse_wid.value),
                        rise_time_s= float(pulse_rise.value),
                        fall_time_s= float(pulse_fall.value),
                        channel    = ch,
                    )
                elif fn == "NOIS":
                    c.apply_noise(a, o, channel=ch)
                elif fn == "DC":
                    c.apply_dc(o, channel=ch)
                elif fn == "ARB":
                    c.apply_arbitrary(f, a, o, channel=ch)
                log_msg(f"ch{ch} apply {fn}  f={f}  A={a}  off={o}")
            except Exception as e:
                log_msg(f"ch{ch} apply FAIL: {type(e).__name__}: {e}")

        async def apply():    await _in_thread(_do_apply)

        async def out_on():
            c = get_ctrl()
            if c is None: log_msg(f"ch{ch} output_on: not connected"); return
            try:
                await _in_thread(c.output_on, ch)
                out_pill.content = (f'<span class="pill ok" '
                                     f'style="margin-top:.4rem; display:inline-block">'
                                     f'ch{ch} output: on</span>')
                log_msg(f"ch{ch} output ON")
            except Exception as e:
                log_msg(f"ch{ch} output_on FAIL: {type(e).__name__}: {e}")

        async def out_off():
            c = get_ctrl()
            if c is None: log_msg(f"ch{ch} output_off: not connected"); return
            try:
                await _in_thread(c.output_off, ch)
                out_pill.content = (f'<span class="pill mut" '
                                     f'style="margin-top:.4rem; display:inline-block">'
                                     f'ch{ch} output: off</span>')
                log_msg(f"ch{ch} output OFF")
            except Exception as e:
                log_msg(f"ch{ch} output_off FAIL: {type(e).__name__}: {e}")

        with ui.row().classes("gap-2 mt-1"):
            ui.button(f"apply ch{ch}", on_click=apply).props("color=primary")
            ui.button("output on",     on_click=out_on).props("color=primary")
            ui.button("output off",    on_click=out_off).props("color=negative")


# ===========================================================================
# build_page
# ===========================================================================

def build_page(get_controller: Optional[Callable[[], Optional[KS33500BController]]] = None,
               *, show_connection: Optional[bool] = None) -> None:
    """Render the KS33500B control panel into the current container."""
    if show_connection is None:
        show_connection = (get_controller is None)

    _own: dict = {"ctrl": None, "arb_samples": None}
    if get_controller is None:
        def get_controller():
            return _own["ctrl"]

    log = ui.log(max_lines=120).classes("h-32 w-full")
    def log_msg(s: str): log.push(f"[{time.strftime('%H:%M:%S')}] {s}")

    with ui.tabs().classes("w-full") as tabs:
        t_conn  = ui.tab("connection") if show_connection else None
        t_ch1   = ui.tab("ch1")
        t_ch2   = ui.tab("ch2")
        t_burst = ui.tab("burst")
        t_sweep = ui.tab("sweep")
        t_arb   = ui.tab("arbitrary")

    initial = t_conn if t_conn is not None else t_ch1
    with ui.tab_panels(tabs, value=initial).classes("w-full"):

        # ----- Connection (standalone only) -----
        if t_conn is not None:
            with ui.tab_panel(t_conn):
                with ui.card().classes("ks-card"):
                    ui.html("<h2>instrument connection</h2>")
                    visa_in = ui.input(label="VISA / device path",
                                        value=DEFAULT_VISA).classes("w-96 num")
                    mode_in = ui.select(["simulation", "hardware"],
                                         value="simulation",
                                         label="mode").classes("w-40")
                    conn_pill = ui.html('<span class="pill mut">disconnected</span>')

                    def set_pill(text: str, cls: str):
                        conn_pill.content = f'<span class="pill {cls}">{text}</span>'

                    async def do_connect():
                        c = KS33500BController(visa=visa_in.value.strip(),
                                                mode=mode_in.value)
                        set_pill("connecting…", "warn")
                        try:
                            await _in_thread(c.connect)
                            _own["ctrl"] = c
                            set_pill(f"OK — {c.identify()[:60]}", "ok")
                            log_msg(f"connected: {c.identify()}")
                        except Exception as e:
                            set_pill(f"FAIL: {type(e).__name__}", "bad")
                            log_msg(f"connect FAIL: {type(e).__name__}: {e}")

                    async def do_disconnect():
                        c = _own["ctrl"]
                        if c is None: return
                        try: await _in_thread(c.disconnect)
                        except Exception as e: log_msg(f"disconnect warn: {e}")
                        _own["ctrl"] = None
                        set_pill("disconnected", "mut")
                        log_msg("disconnected")

                    with ui.row().classes("mt-1 gap-2"):
                        ui.button("connect",    on_click=do_connect).props("color=primary")
                        ui.button("disconnect", on_click=do_disconnect).props("color=negative flat")

        # ----- ch1 / ch2 -----
        with ui.tab_panel(t_ch1):
            _channel_card(1, get_controller, log_msg)
        with ui.tab_panel(t_ch2):
            _channel_card(2, get_controller, log_msg)

        # ----- Burst -----
        with ui.tab_panel(t_burst):
            with ui.card().classes("ks-card"):
                ui.html("<h2>burst mode</h2>")
                b_ch   = ui.select(["1", "2"], value="1", label="channel").classes("w-24")
                b_n    = ui.number(label="N cycles", value=1, step=1).classes("w-32 num")
                b_mode = ui.select(["TRIG", "GAT"], value="TRIG",
                                    label="mode").classes("w-32")
                b_ph   = ui.number(label="initial phase (deg)", value=0.0,
                                    step=1.0).classes("w-40 num")
                b_trig = ui.select(["IMM", "EXT", "TIM", "BUS"], value="IMM",
                                    label="trigger source").classes("w-32")

                async def burst_apply():
                    c = get_controller()
                    if c is None: log_msg("burst: not connected"); return
                    try:
                        # controller.enable_burst(ncycles, mode, phase_deg,
                        #                         trigger, channel)
                        await _in_thread(
                            c.enable_burst,
                            int(b_n.value), str(b_mode.value),
                            float(b_ph.value), str(b_trig.value),
                            int(b_ch.value),
                        )
                        log_msg(f"burst enabled ch{b_ch.value} n={b_n.value} "
                                 f"mode={b_mode.value} trig={b_trig.value}")
                    except Exception as e:
                        log_msg(f"burst_apply FAIL: {type(e).__name__}: {e}")

                async def burst_off():
                    c = get_controller()
                    if c is None: log_msg("burst: not connected"); return
                    try:
                        await _in_thread(c.disable_burst, int(b_ch.value))
                        log_msg(f"burst disabled ch{b_ch.value}")
                    except Exception as e:
                        log_msg(f"burst_off FAIL: {type(e).__name__}: {e}")

                async def burst_trg():
                    c = get_controller()
                    if c is None: log_msg("burst: not connected"); return
                    try:
                        await _in_thread(c.trigger, int(b_ch.value))
                        log_msg(f"*TRG sent (ch{b_ch.value})")
                    except Exception as e:
                        log_msg(f"trigger FAIL: {type(e).__name__}: {e}")

                with ui.row().classes("gap-2 mt-1"):
                    ui.button("enable burst",     on_click=burst_apply).props("color=primary")
                    ui.button("disable burst",    on_click=burst_off).props("color=negative")
                    ui.button("trigger now (*TRG)", on_click=burst_trg)

        # ----- Sweep -----
        with ui.tab_panel(t_sweep):
            with ui.card().classes("ks-card"):
                ui.html("<h2>frequency sweep</h2>")
                sw_ch    = ui.select(["1", "2"], value="1", label="channel").classes("w-24")
                sw_start = ui.number(label="start (Hz)", value=100.0,
                                      step=1.0, format="%.3f").classes("w-40 num")
                sw_stop  = ui.number(label="stop (Hz)",  value=10_000.0,
                                      step=1.0, format="%.3f").classes("w-40 num")
                sw_time  = ui.number(label="sweep time (s)", value=1.0,
                                      step=0.1, format="%.3f").classes("w-32 num")
                sw_spc   = ui.select(["LIN", "LOG"], value="LIN",
                                      label="spacing").classes("w-32")
                sw_trig  = ui.select(["IMM", "EXT", "BUS"], value="IMM",
                                      label="trigger source").classes("w-32")
                sw_ret   = ui.number(label="return time (s)", value=0.0,
                                      step=0.1, format="%.3f").classes("w-32 num")
                sw_hold_a= ui.number(label="hold @ start (s)", value=0.0,
                                      step=0.1, format="%.3f").classes("w-32 num")
                sw_hold_b= ui.number(label="hold @ stop (s)",  value=0.0,
                                      step=0.1, format="%.3f").classes("w-32 num")

                async def sweep_apply():
                    c = get_controller()
                    if c is None: log_msg("sweep: not connected"); return
                    try:
                        # controller.enable_sweep(start_hz, stop_hz, time_s,
                        #     spacing, return_time, hold_start, hold_stop,
                        #     trigger, channel)
                        await _in_thread(
                            c.enable_sweep,
                            float(sw_start.value), float(sw_stop.value),
                            float(sw_time.value),  str(sw_spc.value),
                            float(sw_ret.value),
                            float(sw_hold_a.value), float(sw_hold_b.value),
                            str(sw_trig.value),    int(sw_ch.value),
                        )
                        log_msg(f"sweep ch{sw_ch.value} {sw_start.value}→{sw_stop.value} Hz "
                                 f"{sw_spc.value} in {sw_time.value} s")
                    except Exception as e:
                        log_msg(f"sweep_apply FAIL: {type(e).__name__}: {e}")

                async def sweep_off():
                    c = get_controller()
                    if c is None: log_msg("sweep: not connected"); return
                    try:
                        await _in_thread(c.disable_sweep, int(sw_ch.value))
                        log_msg(f"sweep disabled ch{sw_ch.value}")
                    except Exception as e:
                        log_msg(f"sweep_off FAIL: {type(e).__name__}: {e}")

                with ui.row().classes("gap-2 mt-1"):
                    ui.button("enable sweep",  on_click=sweep_apply).props("color=primary")
                    ui.button("disable sweep", on_click=sweep_off).props("color=negative")

        # ----- Arbitrary -----
        with ui.tab_panel(t_arb):
            with ui.card().classes("ks-card w-full"):
                ui.html("<h2>generate &amp; download an arbitrary waveform</h2>")
                arb_test = ui.select(
                    ["Sine cubed", "Triangle", "Gaussian pulse",
                     "Exp rise", "Exp fall"],
                    value="Sine cubed", label="generator",
                ).classes("w-40")
                arb_npts = ui.number(label="samples", value=1024,
                                      step=64, min=8, max=65536).classes("w-32 num")
                arb_name = ui.input(label="name (on instrument)",
                                     value="ETSARB").classes("w-32")
                arb_ch   = ui.select(["1", "2"], value="1",
                                      label="channel").classes("w-24")

                arb_plot = ui.matplotlib(figsize=(8, 2.4)).classes("w-full")
                arb_ax   = arb_plot.figure.add_subplot(111)
                _style_dark(arb_plot.figure, arb_ax)
                arb_plot.figure.tight_layout()

                def _gen():
                    n = max(8, int(arb_npts.value))
                    t = np.linspace(0.0, 1.0, n)
                    name = str(arb_test.value)
                    if name == "Sine cubed":
                        return np.sin(2 * np.pi * t) ** 3
                    elif name == "Triangle":
                        return 2 * np.abs(2 * (t - np.floor(t + 0.5))) - 1
                    elif name == "Gaussian pulse":
                        return np.exp(-((t - 0.5) / 0.07) ** 2)
                    elif name == "Exp rise":
                        return 1 - np.exp(-5 * t)
                    else:  # "Exp fall"
                        return np.exp(-5 * t)

                def generate_and_preview():
                    samples = _gen()
                    _own["arb_samples"] = samples
                    arb_ax.clear()
                    _style_dark(arb_plot.figure, arb_ax)
                    arb_ax.plot(samples, lw=1.2, color="#58a6ff")
                    arb_ax.set_xlabel("sample", color="#8a93a6", fontsize=9)
                    arb_ax.set_ylabel("amplitude (normalised)",
                                       color="#8a93a6", fontsize=9)
                    arb_plot.figure.tight_layout()
                    arb_plot.update()
                    log_msg(f"generated {len(samples)} samples ({arb_test.value})")

                async def download():
                    samples = _own["arb_samples"]
                    if samples is None:
                        log_msg("no samples — click 'generate + preview' first"); return
                    c = get_controller()
                    if c is None: log_msg("arb: not connected"); return
                    try:
                        await _in_thread(
                            c.load_arbitrary,
                            str(arb_name.value).strip() or "ETSARB",
                            samples, int(arb_ch.value),
                        )
                        log_msg(f"downloaded {len(samples)} samples '{arb_name.value}' "
                                 f"to ch{arb_ch.value}")
                    except Exception as e:
                        log_msg(f"load_arbitrary FAIL: {type(e).__name__}: {e}")

                with ui.row().classes("gap-2 mt-1"):
                    ui.button("generate + preview",   on_click=generate_and_preview).props("color=primary")
                    ui.button("download to instrument", on_click=download).props("color=primary")


# ---------------------------------------------------------------------------
# Standalone entry — `python -m ks33500b.gui`
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Keysight 33500B web GUI")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8772)
    args = p.parse_args()

    @ui.page("/")
    def index():
        ui.add_head_html(f"<style>{_CSS}</style>")
        ui.dark_mode().enable()
        with ui.element("header").style(
            "display:flex;align-items:center;gap:.8rem;"
            "padding:.55rem 1rem;background:var(--panel);"
            "border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5"
        ):
            ui.html("<h1 style='font-size:1.05rem;font-weight:600;margin:0'>"
                    "33500B · waveform generator</h1>")
        build_page()

    ui.run(host=args.host, port=args.port, reload=False,
           title="33500B WFG", show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
