"""Interactive Tk GUI plotter for ASD/NEP noise curves using the Sensor model."""

from __future__ import annotations

from dataclasses import asdict
from math import pi
from pathlib import Path
import pickle
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from sensor import Sensor, Version1SensorInputs


SETTINGS_FILE = Path(__file__).resolve().parent / "last_plot_settings.pkl"

INPUT_SECTIONS = [
    (
        "Operating Point",
        [
            "T0_K",
            "Tb_K",
            "pg_drive_dBm",
            "f0_Hz",
            "detuning_widths",
        ],
    ),
    (
        "Absorber",
        [
            "heat_capacity_eV_per_mK",
            "ho_in_au_atomic_fraction",
        ],
    ),
    (
        "TLS",
        [
            "tls_phi_asd_100hz_per_rtHz",
            "tls_beta",
        ],
    ),
    (
        "Resonator",
        [
            "Qi",
            "Qc",
            "tau_qp_s",
            "kinetic_inductance_fraction",
            "alpha_phi",
        ],
    ),
]
INPUT_KEYS = [k for _, keys in INPUT_SECTIONS for k in keys]

LABELS = {
    "T0_K": "T0 [K]",
    "Tb_K": "Tb [K]",
    "pg_drive_dBm": "Drive [dBm]",
    "f0_Hz": "f0 [Hz]",
    "detuning_widths": "Detuning [widths]",
    "heat_capacity_eV_per_mK": "C [eV/mK]",
    "ho_in_au_atomic_fraction": "Ho/Au frac",
    "tls_phi_asd_100hz_per_rtHz": "TLS ASD @100Hz",
    "tls_beta": "TLS beta",
    "Qi": "Qi",
    "Qc": "Qc",
    "tau_qp_s": "tau_qp [s]",
    "kinetic_inductance_fraction": "kinetic frac",
    "alpha_phi": "alpha_phi",
}


def _positive_limits(arrays: list[np.ndarray], pad: float = 1.3) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return (1e-30, 1.0)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    return (vmin / pad, vmax * pad)


class NoiseGui:
    def __init__(self) -> None:
        self.defaults = asdict(Version1SensorInputs())
        self.current = self._load_saved_or_defaults()
        self.last_loaded = dict(self.current)
        self.undo_stack: list[dict[str, float]] = []

        self.root = tk.Tk()
        self.root.title("Noise ASD / NEP Plotter")
        self.root.geometry("1500x900")

        self.mode_var = tk.StringVar(value="Noise ASD")
        self.status_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self.entry_vars: dict[str, tk.StringVar] = {}

        self._build_layout()
        self._recompute_and_draw()

    @staticmethod
    def _settings_filetypes() -> list[tuple[str, str]]:
        return [("Pickle files", "*.pkl"), ("All files", "*.*")]

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=4)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(self.root, padding=8)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        # Keep Matplotlib canvas + toolbar in a pack-managed subframe to avoid
        # mixing Tk geometry managers in the same parent.
        plot_inner = ttk.Frame(plot_frame)
        plot_inner.grid(row=0, column=0, sticky="nsew")

        self.fig = Figure(figsize=(11, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_inner)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_inner)
        toolbar.update()

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=1, sticky="ns")

        ttk.Label(controls, text="Noise/NEP Controls", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        mode_frame = ttk.LabelFrame(controls, text="Mode", padding=6)
        mode_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(mode_frame, text="Noise ASD", variable=self.mode_var, value="Noise ASD", command=self._on_mode).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Radiobutton(mode_frame, text="NEP", variable=self.mode_var, value="NEP", command=self._on_mode).grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )

        row = 2
        for section_name, keys in INPUT_SECTIONS:
            section = ttk.LabelFrame(controls, text=section_name, padding=6)
            section.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 8))
            row += 1
            for i, key in enumerate(keys):
                ttk.Label(section, text=LABELS[key]).grid(row=i, column=0, sticky="w", padx=(0, 6), pady=2)
                var = tk.StringVar(value=f"{self.current[key]:.8g}")
                ent = ttk.Entry(section, textvariable=var, width=16)
                ent.grid(row=i, column=1, sticky="ew", pady=2)
                ent.bind("<Return>", self._on_field_commit)
                ent.bind("<FocusOut>", self._on_field_commit)
                self.entry_vars[key] = var

        button_frame = ttk.Frame(controls)
        button_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(2, 6))
        row += 1

        ttk.Button(button_frame, text="Defaults", command=self._load_defaults).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Undo", command=self._undo).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Save", command=self._save_current).grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Load", command=self._load_saved).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Restore", command=self._restore_last_loaded).grid(row=1, column=1, padx=2, pady=2, sticky="ew")

        for i in range(3):
            button_frame.columnconfigure(i, weight=1)

        ttk.Label(controls, textvariable=self.status_var, wraplength=300, foreground="#444").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        row += 1

        ttk.Label(
            controls,
            textvariable=self.summary_var,
            justify="right",
            anchor="e",
            wraplength=320,
            foreground="#222",
        ).grid(row=row, column=0, columnspan=2, sticky="e", pady=(14, 0))

    def _load_saved_or_defaults(self) -> dict[str, float]:
        if SETTINGS_FILE.exists():
            try:
                with SETTINGS_FILE.open("rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    merged = dict(self.defaults)
                    for k in INPUT_KEYS:
                        if k in loaded:
                            merged[k] = float(loaded[k])
                    return merged
            except Exception:
                pass
        return dict(self.defaults)

    def _read_fields(self) -> dict[str, float]:
        vals = dict(self.current)
        for k, var in self.entry_vars.items():
            vals[k] = float(var.get().strip())
        return vals

    def _write_fields(self, vals: dict[str, float]) -> None:
        for k, var in self.entry_vars.items():
            var.set(f"{vals[k]:.8g}")

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _set_summary(self, s: Sensor) -> None:
        delta_t_mk = 1.0e3 * float(s.deltaT_event_full_absorption_K)
        rate_hz = float(s.count_rate_Hz)
        shorten = float(s.mt_pulse_shortening_ratio)
        self.summary_var.set(
            "Event dT (island): "
            f"{delta_t_mk:.3g} mK\n"
            "Average event rate: "
            f"{rate_hz:.3g} Hz\n"
            "Pulse shortening factor: "
            f"{shorten:.3g}"
        )

    def _push_undo(self, prev: dict[str, float]) -> None:
        self.undo_stack.append(dict(prev))
        if len(self.undo_stack) > 20:
            self.undo_stack = self.undo_stack[-20:]

    def _build_sensor(self, settings: dict[str, float]) -> Sensor:
        kwargs = dict(self.defaults)
        kwargs.update(settings)
        return Sensor(Version1SensorInputs(**kwargs))

    def _compute_data(self, s: Sensor) -> dict[str, object]:
        eigs = np.array(s.mt_eigenvalues, dtype=complex)
        max_mode_rate_per_s = float(np.max(np.abs(eigs)))
        f_min_hz = 0.1
        f_max_hz = max(f_min_hz * 10.0, 10.0 * max_mode_rate_per_s / (2.0 * pi))
        freqs_hz = np.logspace(np.log10(f_min_hz), np.log10(f_max_hz), 1000)

        asd_johnson = np.zeros_like(freqs_hz)
        asd_phonon = np.zeros_like(freqs_hz)
        asd_tls = np.zeros_like(freqs_hz)
        asd_electronic = np.zeros_like(freqs_hz)
        phase_resp = np.zeros_like(freqs_hz)

        for i, f_hz in enumerate(freqs_hz):
            y_j_a = s._propagate_noise_vector(s.n_johnson_A(), f_hz)
            y_j_phi = s._propagate_noise_vector(s.n_johnson_phi(), f_hz)
            y_ph = s._propagate_noise_vector(s.n_phonon(), f_hz)
            m_tls_f = s.m_tls_from_ratio(s.sf_over_f0sq_tls_at_hz(float(f_hz)), s.sf_over_f0sq_johnson_full)
            y_tls = s._propagate_noise_vector(s.n_tls_phi(m_tls_f), f_hz)
            y_e_a = s._propagate_noise_vector(s.n_electronic_A(), f_hz)
            y_e_phi = s._propagate_noise_vector(s.n_electronic_phi(), f_hz)

            asd_johnson[i] = np.sqrt(abs(y_j_a[1]) ** 2 + abs(y_j_phi[1]) ** 2)
            asd_phonon[i] = abs(y_ph[1])
            asd_tls[i] = abs(y_tls[1])
            asd_electronic[i] = np.sqrt(abs(y_e_a[1]) ** 2 + abs(y_e_phi[1]) ** 2)
            phase_resp[i] = s.phase_responsivity_mag_rad_per_W_at_hz(float(f_hz))

        asd_total = np.sqrt(asd_johnson**2 + asd_phonon**2 + asd_tls**2 + asd_electronic**2)
        asd_tls_direct = s.tls_phi_asd_100hz_per_rtHz * ((freqs_hz / 100.0) ** (-s.tls_beta / 2.0))
        asd_johnson_simple = abs(s.f0_Hz * s.dphi_df_detuning_per_hz) * np.sqrt(s.sf_over_f0sq_johnson_simple)
        nep_johnson = np.where(phase_resp > 0.0, asd_johnson / phase_resp, np.nan)
        nep_phonon = np.where(phase_resp > 0.0, asd_phonon / phase_resp, np.nan)
        nep_tls = np.where(phase_resp > 0.0, asd_tls / phase_resp, np.nan)
        nep_electronic = np.where(phase_resp > 0.0, asd_electronic / phase_resp, np.nan)
        nep_total = np.where(phase_resp > 0.0, asd_total / phase_resp, np.nan)

        sigma_e_mev = 1.0e3 * s.sigma_energy_from_nep_spectrum_eV(freqs_hz, nep_total)

        marker_specs = [
            (1.0 / (2.0 * pi * s.tau_th_s), "f_therm"),
            (1.0 / (2.0 * pi * s.tau_res_s), "f_res"),
        ]
        marker_specs.extend((abs(complex(lam)) / (2.0 * pi), f"f_eig{i + 1}") for i, lam in enumerate(eigs))
        valid_markers = [(float(fm), name) for fm, name in marker_specs if np.isfinite(fm) and fm > 0.0]
        valid_markers.sort(key=lambda x: x[0])

        asd_ylim = _positive_limits(
            [asd_johnson, asd_phonon, asd_tls, asd_electronic, asd_total, asd_tls_direct, np.array([asd_johnson_simple])]
        )
        nep_ylim = _positive_limits([nep_johnson, nep_phonon, nep_tls, nep_electronic, nep_total])

        return {
            "sensor": s,
            "freqs": freqs_hz,
            "asd": (asd_johnson, asd_phonon, asd_tls, asd_electronic, asd_total),
            "asd_tls_direct": asd_tls_direct,
            "nep": (nep_johnson, nep_phonon, nep_tls, nep_electronic, nep_total),
            "asd_johnson_simple": asd_johnson_simple,
            "sigma_mev": sigma_e_mev,
            "markers": valid_markers,
            "asd_ylim": asd_ylim,
            "nep_ylim": nep_ylim,
        }

    def _draw(self) -> None:
        d = self.data
        s: Sensor = d["sensor"]  # type: ignore[assignment]
        freqs = d["freqs"]
        asd_johnson, asd_phonon, asd_tls, asd_electronic, asd_total = d["asd"]
        nep_johnson, nep_phonon, nep_tls, nep_electronic, nep_total = d["nep"]

        self.ax.clear()
        mode = self.mode_var.get()
        if mode == "NEP":
            ysets = (nep_johnson, nep_phonon, nep_tls, nep_electronic, nep_total)
            ylab = "NEP [W/rtHz]"
            title = "Noise-Equivalent Power vs Frequency"
            self.ax.set_ylim(*d["nep_ylim"])
        else:
            ysets = (asd_johnson, asd_phonon, asd_tls, asd_electronic, asd_total)
            ylab = "Phase ASD [rad/rtHz]"
            title = "Noise ASD vs Frequency"
            self.ax.set_ylim(*d["asd_ylim"])

        self.ax.loglog(freqs, ysets[0], label="Johnson", color="tab:blue")
        self.ax.loglog(freqs, ysets[1], label="Phonon", color="tab:orange")
        self.ax.loglog(freqs, ysets[2], label="TLS", color="tab:green")
        self.ax.loglog(freqs, ysets[3], label="Electronic", color="tab:red")
        self.ax.loglog(freqs, ysets[4], "k", linewidth=2.0, label="Total (quadrature)")

        if mode == "Noise ASD":
            self.ax.axhline(
                d["asd_johnson_simple"],
                linestyle=":",
                linewidth=1.8,
                color="tab:blue",
                alpha=0.9,
                label="Johnson (simple estimate)",
            )
            self.ax.loglog(
                freqs,
                d["asd_tls_direct"],
                linestyle=":",
                linewidth=2.0,
                color="tab:green",
                alpha=0.9,
                label="TLS (direct beta law)",
            )

        y_levels = [0.96, 0.82, 0.68, 0.54, 0.40, 0.26]
        cluster_step_log10 = 0.035
        prev_log10_f = None
        cluster_idx = -1
        for f_mark_hz, name in d["markers"]:
            self.ax.axvline(f_mark_hz, linestyle="--", linewidth=1.0, alpha=0.7, color="#666")
            log10_f = float(np.log10(f_mark_hz))
            if prev_log10_f is None or (log10_f - prev_log10_f) > cluster_step_log10:
                cluster_idx = 0
            else:
                cluster_idx += 1
            y_text = y_levels[cluster_idx % len(y_levels)]
            self.ax.text(f_mark_hz, y_text, name, rotation=90, va="top", ha="right", transform=self.ax.get_xaxis_transform())
            prev_log10_f = log10_f

        if s.mt_stable:
            label = f"Estimated energy resolution: sigma_E = {d['sigma_mev']:.3f} meV"
            color = "black"
        else:
            label = "UNSTABLE: Mt eigenvalues indicate instability"
            color = "red"
        self.ax.text(
            0.98,
            0.02,
            label,
            color=color,
            transform=self.ax.transAxes,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )

        self.ax.set_title(title)
        self.ax.set_xlabel("Frequency [Hz]")
        self.ax.set_ylabel(ylab)
        self.ax.grid(True, which="both", alpha=0.25)
        legend_loc = "lower left" if mode == "Noise ASD" else "upper left"
        self.ax.legend(loc=legend_loc, framealpha=0.9)
        self.canvas.draw_idle()

    def _recompute_and_draw(self) -> None:
        sensor = self._build_sensor(self.current)
        self.data = self._compute_data(sensor)
        self._set_summary(sensor)
        self._draw()

    def _on_mode(self) -> None:
        self._draw()

    def _apply_from_fields(self) -> None:
        try:
            new_vals = self._read_fields()
            prev = dict(self.current)
            self.current = new_vals
            self._recompute_and_draw()
            self._push_undo(prev)
            self._set_status("Applied settings")
        except Exception as exc:
            self._set_status(f"Apply failed: {exc}")

    def _on_field_commit(self, _event=None) -> None:
        self._apply_from_fields()

    def _load_defaults(self) -> None:
        prev = dict(self.current)
        self.current = dict(self.defaults)
        self._write_fields(self.current)
        try:
            self._recompute_and_draw()
            self._push_undo(prev)
            self._set_status("Loaded defaults")
        except Exception as exc:
            self.current = prev
            self._write_fields(self.current)
            self._set_status(f"Defaults failed: {exc}")

    def _save_current(self) -> None:
        try:
            self.current = self._read_fields()
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title="Save Plot Settings",
                initialdir=str(SETTINGS_FILE.parent),
                initialfile=SETTINGS_FILE.name,
                defaultextension=".pkl",
                filetypes=self._settings_filetypes(),
            )
            if not path:
                self._set_status("Save cancelled")
                return
            save_path = Path(path)
            with save_path.open("wb") as f:
                pickle.dump({k: self.current[k] for k in INPUT_KEYS}, f)
            self.last_loaded = dict(self.current)
            self._set_status(f"Saved settings to {save_path.name}")
        except Exception as exc:
            self._set_status(f"Save failed: {exc}")

    def _load_saved(self) -> None:
        try:
            path = filedialog.askopenfilename(
                parent=self.root,
                title="Load Plot Settings",
                initialdir=str(SETTINGS_FILE.parent),
                filetypes=self._settings_filetypes(),
            )
            if not path:
                self._set_status("Load cancelled")
                return
            load_path = Path(path)
            with load_path.open("rb") as f:
                loaded = pickle.load(f)
            vals = dict(self.defaults)
            for k in INPUT_KEYS:
                vals[k] = float(loaded[k])
            prev = dict(self.current)
            self.current = vals
            self.last_loaded = dict(vals)
            self._write_fields(self.current)
            self._recompute_and_draw()
            self._push_undo(prev)
            self._set_status(f"Loaded settings from {load_path.name}")
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")

    def _restore_last_loaded(self) -> None:
        prev = dict(self.current)
        self.current = dict(self.last_loaded)
        self._write_fields(self.current)
        try:
            self._recompute_and_draw()
            self._push_undo(prev)
            self._set_status("Restored last loaded settings")
        except Exception as exc:
            self.current = prev
            self._write_fields(self.current)
            self._set_status(f"Restore failed: {exc}")

    def _undo(self) -> None:
        if not self.undo_stack:
            self._set_status("Undo stack empty")
            return
        prev = self.undo_stack.pop()
        self.current = prev
        self._write_fields(self.current)
        try:
            self._recompute_and_draw()
            self._set_status("Undo applied")
        except Exception as exc:
            self._set_status(f"Undo failed: {exc}")

    def run(self) -> None:
        startup = "saved" if SETTINGS_FILE.exists() else "defaults"
        self._set_status(f"Loaded startup settings ({startup})")
        self.root.mainloop()


def main() -> None:
    app = NoiseGui()
    app.run()


if __name__ == "__main__":
    main()
