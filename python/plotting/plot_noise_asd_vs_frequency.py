"""Plot phase ASD noise sources vs frequency using the Sensor model."""

from __future__ import annotations

from math import pi

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons

from sensor import Sensor, Version1SensorInputs


def main() -> None:
    s = Sensor(Version1SensorInputs())

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
    nep_johnson = np.where(phase_resp > 0.0, asd_johnson / phase_resp, np.nan)
    nep_phonon = np.where(phase_resp > 0.0, asd_phonon / phase_resp, np.nan)
    nep_tls = np.where(phase_resp > 0.0, asd_tls / phase_resp, np.nan)
    nep_electronic = np.where(phase_resp > 0.0, asd_electronic / phase_resp, np.nan)
    nep_total = np.where(phase_resp > 0.0, asd_total / phase_resp, np.nan)
    sigma_e_mev = 1.0e3 * s.sigma_energy_from_nep_spectrum_eV(freqs_hz, nep_total)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=0.10, right=0.78, top=0.92, bottom=0.11)
    (line_johnson,) = ax.loglog(freqs_hz, asd_johnson, label="Johnson")
    (line_phonon,) = ax.loglog(freqs_hz, asd_phonon, label="Phonon")
    (line_tls,) = ax.loglog(freqs_hz, asd_tls, label="TLS")
    (line_electronic,) = ax.loglog(freqs_hz, asd_electronic, label="Electronic")
    (line_total,) = ax.loglog(freqs_hz, asd_total, "k", linewidth=2.0, label="Total (quadrature)")

    marker_specs = [
        (1.0 / (2.0 * pi * s.tau_th_s), "f_therm"),
        (1.0 / (2.0 * pi * s.tau_res_s), "f_res"),
    ]
    marker_specs.extend((abs(complex(lam)) / (2.0 * pi), f"f_eig{i + 1}") for i, lam in enumerate(eigs))

    valid_markers = [(float(f_mark_hz), name) for f_mark_hz, name in marker_specs if np.isfinite(f_mark_hz) and f_mark_hz > 0.0]
    valid_markers.sort(key=lambda x: x[0])

    y_levels = [0.96, 0.82, 0.68, 0.54, 0.40, 0.26]
    cluster_step_log10 = 0.035
    prev_log10_f = None
    cluster_idx = -1

    for f_mark_hz, name in valid_markers:
        ax.axvline(f_mark_hz, linestyle="--", linewidth=1.0, alpha=0.7)
        log10_f = float(np.log10(f_mark_hz))
        if prev_log10_f is None or (log10_f - prev_log10_f) > cluster_step_log10:
            cluster_idx = 0
        else:
            cluster_idx += 1
        y_text = y_levels[cluster_idx % len(y_levels)]
        ax.text(f_mark_hz, y_text, name, rotation=90, va="top", ha="right", transform=ax.get_xaxis_transform())
        prev_log10_f = log10_f

    ax.set_title("Noise ASD vs Frequency")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Phase ASD [rad/rtHz]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left")
    if s.mt_stable:
        sigma_label = f"Estimated energy resolution: sigma_E = {sigma_e_mev:.3f} meV"
        sigma_color = "black"
    else:
        sigma_label = "UNSTABLE: Mt eigenvalues indicate instability"
        sigma_color = "red"
    sigma_text = ax.text(
        0.98,
        0.02,
        sigma_label,
        color=sigma_color,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )

    radio_ax = fig.add_axes([0.81, 0.62, 0.17, 0.22])
    radio_ax.set_title("Display")
    mode_radio = RadioButtons(radio_ax, ("Noise ASD", "NEP"), active=0)

    def _set_mode(mode_label: str) -> None:
        if mode_label == "NEP":
            line_johnson.set_ydata(nep_johnson)
            line_phonon.set_ydata(nep_phonon)
            line_tls.set_ydata(nep_tls)
            line_electronic.set_ydata(nep_electronic)
            line_total.set_ydata(nep_total)
            ax.set_title("Noise-Equivalent Power vs Frequency")
            ax.set_ylabel("NEP [W/rtHz]")
        else:
            line_johnson.set_ydata(asd_johnson)
            line_phonon.set_ydata(asd_phonon)
            line_tls.set_ydata(asd_tls)
            line_electronic.set_ydata(asd_electronic)
            line_total.set_ydata(asd_total)
            ax.set_title("Noise ASD vs Frequency")
            ax.set_ylabel("Phase ASD [rad/rtHz]")
        ax.relim()
        ax.autoscale_view()
        if s.mt_stable:
            sigma_text.set_text(f"Estimated energy resolution: sigma_E = {sigma_e_mev:.3f} meV")
            sigma_text.set_color("black")
        else:
            sigma_text.set_text("UNSTABLE: Mt eigenvalues indicate instability")
            sigma_text.set_color("red")
        fig.canvas.draw_idle()

    mode_radio.on_clicked(_set_mode)
    plt.show()


if __name__ == "__main__":
    main()
