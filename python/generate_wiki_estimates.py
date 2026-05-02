"""Generate code-derived estimates for the wiki.

Outputs:
- python/outputs/wiki_estimates.json
- wiki/python-estimates.html
"""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

from sensor import nominal_sensor


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "python" / "outputs" / "wiki_estimates.json"
OUT_HTML = ROOT / "wiki" / "python-estimates.html"


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e4:
        return f"{x:.6g}"
    return f"{x:.6e}"


def main() -> None:
    s = nominal_sensor()
    est = s.estimates()
    input_keys = [f.name for f in fields(s)]

    units = {
        "T0_K": "K",
        "Tb_K": "K",
        "count_rate_Hz": "Hz",
        "pileup_probability_max": "1",
        "ho_activity_per_m3_Hz": "Hz/m^3",
        "absorber_length_m": "m",
        "absorber_width_m": "m",
        "absorber_thickness_m": "m",
        "absorber_edge_m": "m",
        "kid_length_m": "m",
        "kid_width_m": "m",
        "membrane_margin_m": "m",
        "leg_count": "count",
        "leg_width_m": "m",
        "leg_thickness_m": "m",
        "cap_thickness_m": "m",
        "membrane_thickness_m": "m",
        "cv_absorber_J_per_m3K": "J/(m^3 K)",
        "kappa_leg_W_per_mK": "W/(m K)",
        "tls_F_participation": "1",
        "tls_tan_delta": "1",
        "tls_beta": "1",
        "tls_A_scale": "1",
        "tls_power_exponent_m": "1",
        "tls_Pint_W": "W",
        "tls_Pc_W": "W",
        "tls_nu_Hz": "Hz",
        "sphi_j_ref_per_hz": "rad^2/Hz",
        "f0_Hz": "Hz",
        "Qr": "1",
        "Qi": "1",
        "kinetic_inductance_fraction": "1",
        "kid_trace_length_m": "m",
        "kid_trace_width_m": "m",
        "alpha_A": "1",
        "alpha_phi": "1",
        "beta_A": "1",
        "beta_phi": "1",
        "Tc_K": "K",
        "P0_W": "W",
        "delta_J": "J",
        "eqp_J": "J",
        "detuning_Hz": "Hz",
        "f_demod_Hz": "Hz",
        "absorber_volume_m3": "m^3",
        "membrane_length_m": "m",
        "membrane_width_m": "m",
        "membrane_span_m": "m",
        "leg_length_m": "m",
        "C_J_per_K": "J/K",
        "G_W_per_K": "W/K",
        "tau_th_s": "s",
        "tau_target_from_rate_s": "s",
        "tau_error_fraction": "1",
        "tau_res_s": "s",
        "tau_ratio_res_over_th": "1",
        "L_geo_H": "H",
        "L_total_H": "H",
        "C_res_F": "F",
        "Z0_res_Ohm": "Ohm",
        "R0_Ohm": "Ohm",
        "sf_over_f0sq_johnson_ref": "1/Hz",
        "sf_over_f0sq_tls_model": "1/Hz",
        "phonon_power_rms_W": "W",
        "johnson_voltage_rms_V": "V",
        "johnson_sv_V2_per_Hz": "V^2/Hz",
        "M_e": "1",
        "N_J_scale": "norm",
        "N_J_thermal_scale": "norm",
    }

    # TLS multiplier from modeled TLS PSD and reference Johnson PSD.
    sf_f2_j = s.sf_over_f0sq_johnson_ref()
    sf_f2_tls = s.sf_over_f0sq_tls()
    m_tls = s.m_tls_from_ratio(sf_f2_tls, sf_f2_j)

    vectors = {
        "N_ph": [complex(v).real if complex(v).imag == 0 else [complex(v).real, complex(v).imag] for v in s.n_phonon()],
        "N_J_A": [complex(v).real if complex(v).imag == 0 else [complex(v).real, complex(v).imag] for v in s.n_johnson_A()],
        "N_J_phi": [complex(v).real if complex(v).imag == 0 else [complex(v).real, complex(v).imag] for v in s.n_johnson_phi()],
        "N_g_A": [complex(v).real if complex(v).imag == 0 else [complex(v).real, complex(v).imag] for v in s.n_electronic_A()],
        "N_g_phi": [complex(v).real if complex(v).imag == 0 else [complex(v).real, complex(v).imag] for v in s.n_electronic_phi()],
        "N_TLS_phi_example": [complex(v).real if complex(v).imag == 0 else [complex(v).real, complex(v).imag] for v in s.n_tls_phi(m_tls)],
    }
    m_1hz = s.m_matrix(1.0)
    m_1hz_serialized = [[[z.real, z.imag] for z in row] for row in m_1hz]

    model_inputs = {k: getattr(s, k) for k in input_keys}
    model_outputs = {k: v for k, v in est.items() if k not in model_inputs}

    payload = {
        "notes": {
            "f0_Hz": "Nominal detector carrier frequency",
            "f_demod_Hz": "Demodulated analysis frequency",
            "detuning_Hz": "Project parameter (currently 0)",
            "complex_value_format": "[real, imag]",
        },
        "units": units,
        "model_inputs": model_inputs,
        "model_outputs": model_outputs,
        "tls_ratio_example": {
            "Sf_over_f0sq_J": sf_f2_j,
            "Sf_over_f0sq_TLS": sf_f2_tls,
            "M_TLS": m_tls,
        },
        "vectors": vectors,
        "M_matrix_1Hz": {
            "frequency_Hz": 1.0,
            "format": "[real, imag]",
            "rows": m_1hz_serialized,
        },
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    input_groups = {
        "Source Geometry": [
            "count_rate_Hz",
            "pileup_probability_max",
            "kid_length_m",
            "kid_width_m",
            "membrane_margin_m",
            "leg_count",
            "leg_width_m",
            "cap_thickness_m",
            "membrane_thickness_m",
        ],
        "Setpoints": [
            "T0_K",
            "Tb_K",
            "P0_W",
            "f0_Hz",
            "f_demod_Hz",
            "detuning_Hz",
        ],
        "KID Properties": [
            "Qr",
            "Qi",
            "kinetic_inductance_fraction",
            "kid_trace_length_m",
            "kid_trace_width_m",
            "alpha_A",
            "alpha_phi",
            "beta_A",
            "beta_phi",
            "Tc_K",
        ],
        "Material and Activity": [
            "ho_activity_per_m3_Hz",
            "cv_absorber_J_per_m3K",
            "kappa_leg_W_per_mK",
        ],
        "Capacitor TLS": [
            "tls_F_participation",
            "tls_tan_delta",
            "tls_beta",
            "tls_A_scale",
            "tls_power_exponent_m",
            "tls_Pint_W",
            "tls_Pc_W",
            "tls_nu_Hz",
            "sphi_j_ref_per_hz",
        ],
    }

    def _rows_for_keys(keys: list[str]) -> str:
        return "\n".join(
            f"<tr><td>{k}</td><td><code>{_fmt(float(model_inputs[k]))}</code></td><td>{units.get(k,'')}</td></tr>"
            for k in keys
            if k in model_inputs
        )

    input_sections = "\n".join(
        f"""
    <section class=\"card\">
      <h3>{group_name}</h3>
      <table>
        <tr><th>Quantity</th><th>Value</th><th>Units</th></tr>
        {_rows_for_keys(keys)}
      </table>
    </section>
    """
        for group_name, keys in input_groups.items()
    )
    output_groups = {
        "Derived Geometry": [
            "absorber_volume_m3",
            "absorber_edge_m",
            "absorber_length_m",
            "absorber_width_m",
            "absorber_thickness_m",
            "membrane_length_m",
            "membrane_width_m",
            "membrane_span_m",
            "leg_length_m",
            "leg_thickness_m",
        ],
        "Derived Thermal": [
            "C_J_per_K",
            "G_W_per_K",
            "tau_th_s",
            "tau_target_from_rate_s",
            "tau_error_fraction",
            "tau_res_s",
            "tau_ratio_res_over_th",
            "phonon_power_rms_W",
        ],
        "Derived KID Electrical": [
            "f0_Hz",
            "detuning_Hz",
            "f_demod_Hz",
            "L_geo_H",
            "L_total_H",
            "C_res_F",
            "Z0_res_Ohm",
            "R0_Ohm",
            "Qr",
            "Qi",
            "alpha_A",
            "alpha_phi",
            "beta_A",
            "beta_phi",
            "Tc_K",
            "delta_J",
            "eqp_J",
            "johnson_voltage_rms_V",
            "johnson_sv_V2_per_Hz",
            "M_e",
            "N_J_scale",
            "N_J_thermal_scale",
        ],
        "Derived TLS": [
            "sf_over_f0sq_johnson_ref",
            "sf_over_f0sq_tls_model",
        ],
        "Rate and Source": [
            "count_rate_Hz",
            "pileup_probability_max",
            "ho_activity_per_m3_Hz",
        ],
    }

    def _output_rows_for_keys(keys: list[str]) -> str:
        return "\n".join(
            f"<tr><td>{k}</td><td><code>{_fmt(model_outputs[k])}</code></td><td>{units.get(k,'')}</td></tr>"
            for k in keys
            if k in model_outputs
        )

    output_sections = "\n".join(
        f"""
    <section class=\"card\">
      <h3>{group_name}</h3>
      <table>
        <tr><th>Quantity</th><th>Value</th><th>Units</th></tr>
        {_output_rows_for_keys(keys)}
      </table>
    </section>
    """
        for group_name, keys in output_groups.items()
    )

    html = f"""<!doctype html>
<html lang=\"en\"> 
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Python-Derived Estimates | TKID Neutrino Detector Project Wiki</title>
  <link rel=\"stylesheet\" href=\"styles.css\" />
</head>
<body>
  <header class=\"hero\">
    <h1>Python-Derived Estimates</h1>
    <p>Generated from <code>python/sensor.py</code> by <code>python/generate_wiki_estimates.py</code>.</p>
  </header>

  <nav class=\"nav\">
    <a href=\"index.html\">Home</a>
    <a href=\"project.html\">Project</a>
    <a href=\"theory.html\">Theory</a>
    <a href=\"noise-sources.html\">Noise Sources</a>
    <a href=\"python-estimates.html\" class=\"active\">Python Estimates</a>
  </nav>

  <main class=\"container\">
    <section class=\"card\">
      <h2>Nominal Settings</h2>
      <p>All values use nominal <strong>f0 = 1.0 GHz</strong>, demodulated band frequency <strong>0 Hz</strong>, and detuning <strong>0 Hz</strong>.</p>
      <p>JSON source: <code>../python/outputs/wiki_estimates.json</code></p>
    </section>

    <section class=\"card\">
      <h2>Model Inputs</h2>
    </section>
    {input_sections}

    <section class=\"card\">
      <h2>Model Outputs</h2>
    </section>
    {output_sections}
  </main>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
