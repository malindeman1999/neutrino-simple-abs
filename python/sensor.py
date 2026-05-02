"""Sensor model for TKID wiki estimates.

Wiki references:
- wiki/project.html
- wiki/theory.html
- wiki/noise-sources.html
- wiki/noise-johnson.html
- wiki/noise-tls.html
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt
from typing import Dict, Tuple

K_B = 1.380649e-23
MU0 = 4.0e-7 * pi


@dataclass(frozen=True)
class Sensor:
    """Primary project parameters only (no derived fields).

    Basic inputs are geometry + count-rate requirements, plus independent
    electrical/material constants. Leg width is a basic geometry parameter.
    Tau derives from count rate, G derives from C/tau, and leg length derives
    from G and leg geometry/material parameters.

    Notes:
    - Noise estimates are evaluated in phase direction at demodulated frequency 0 Hz.
    - Modulated-band frequency at this point is the carrier f0.
    - Zero detuning is currently assumed and stored as a project parameter.
    """

    # Operating temperatures
    T0_K: float
    Tb_K: float

    # Geometry + count-rate requirements (primary project knobs)
    count_rate_Hz: float
    pileup_probability_max: float
    ho_activity_per_m3_Hz: float
    kid_length_m: float
    kid_width_m: float
    membrane_margin_m: float
    leg_count: int
    leg_width_m: float
    cap_thickness_m: float
    membrane_thickness_m: float

    # Material properties used with geometry (independent inputs)
    cv_absorber_J_per_m3K: float
    kappa_leg_W_per_mK: float

    # Capacitor TLS model inputs (IDC-related)
    tls_F_participation: float
    tls_tan_delta: float
    tls_beta: float
    tls_A_scale: float
    tls_power_exponent_m: float
    tls_Pint_W: float
    tls_Pc_W: float
    tls_nu_Hz: float
    sphi_j_ref_per_hz: float

    # Resonator/electrical operating point
    f0_Hz: float
    Qr: float
    Qi: float
    kinetic_inductance_fraction: float
    kid_trace_length_m: float
    kid_trace_width_m: float
    alpha_A: float
    alpha_phi: float
    beta_A: float
    beta_phi: float
    Tc_K: float
    P0_W: float

    # Readout condition
    detuning_Hz: float
    f_demod_Hz: float = 0.0

    @property
    def absorber_volume_m3(self) -> float:
        """Derived Ho absorber volume from count rate and volumetric activity."""
        return self.count_rate_Hz / self.ho_activity_per_m3_Hz

    @property
    def absorber_edge_m(self) -> float:
        """Cubic absorber edge from derived volume."""
        return self.absorber_volume_m3 ** (1.0 / 3.0)

    @property
    def absorber_length_m(self) -> float:
        return self.absorber_edge_m

    @property
    def absorber_width_m(self) -> float:
        return self.absorber_edge_m

    @property
    def absorber_thickness_m(self) -> float:
        return self.absorber_edge_m

    @property
    def membrane_length_m(self) -> float:
        """Derived membrane length from absorber + KID + margin."""
        return self.absorber_length_m + self.kid_length_m + self.membrane_margin_m

    @property
    def membrane_width_m(self) -> float:
        """Derived membrane width from absorber + KID + margin."""
        return self.absorber_width_m + self.kid_width_m + self.membrane_margin_m

    @property
    def membrane_span_m(self) -> float:
        """Characteristic span for leg scaling."""
        return max(self.membrane_length_m, self.membrane_width_m)

    @property
    def leg_thickness_m(self) -> float:
        """Leg thickness derived from membrane thickness."""
        return self.membrane_thickness_m

    @property
    def C_J_per_K(self) -> float:
        """Derived heat capacity from absorber geometry."""
        return self.cv_absorber_J_per_m3K * self.absorber_volume_m3

    @property
    def tau_resolve_s(self) -> float:
        """Resolve window from pileup requirement: P ~ R * tau_resolve."""
        return self.pileup_probability_max / self.count_rate_Hz

    @property
    def tau_th_s(self) -> float:
        """Thermal decay tau from pileup approximation: P ~ R * tau."""
        return self.pileup_probability_max / self.count_rate_Hz

    @property
    def G_W_per_K(self) -> float:
        """Derived thermal conductance from C / tau."""
        return self.C_J_per_K / self.tau_th_s

    @property
    def leg_length_m(self) -> float:
        """Derived leg length from G and leg geometry/material.

        Rearranged from G = N * kappa * (w * t / L):
        L = N * kappa * w * t / G
        """
        return (
            self.leg_count
            * self.kappa_leg_W_per_mK
            * self.leg_width_m
            * self.leg_thickness_m
            / self.G_W_per_K
        )

    @property
    def tau_res_s(self) -> float:
        return self.Qr / (pi * self.f0_Hz)

    @property
    def L_geo_H(self) -> float:
        """Approximate geometric inductance from trace length/width.

        Thin straight-trace approximation:
        L ~ mu0 * l * (ln(2l/w) + 0.5)
        """
        l = self.kid_trace_length_m
        w = self.kid_trace_width_m
        ratio = max((2.0 * l) / w, 1.000001)
        return MU0 * l * (log(ratio) + 0.5)

    @property
    def L_total_H(self) -> float:
        """Total inductance from geometric inductance and kinetic fraction."""
        kfrac = self.kinetic_inductance_fraction
        if not (0.0 <= kfrac < 1.0):
            raise ValueError("kinetic_inductance_fraction must be in [0,1)")
        return self.L_geo_H / (1.0 - kfrac)

    @property
    def C_res_F(self) -> float:
        """Resonator capacitance from f0 and L_total."""
        w0 = 2.0 * pi * self.f0_Hz
        return 1.0 / (w0 * w0 * self.L_total_H)

    @property
    def Z0_res_Ohm(self) -> float:
        """Nominal resonator characteristic impedance at f0."""
        w0 = 2.0 * pi * self.f0_Hz
        return w0 * self.L_total_H

    @property
    def R0_Ohm(self) -> float:
        """Series-equivalent resonator resistance from Z0 and Q."""
        return self.Z0_res_Ohm / self.Qr

    @property
    def phonon_power_rms_W(self) -> float:
        return sqrt(4.0 * K_B * (self.Tb_K**2) * self.G_W_per_K)

    @property
    def johnson_voltage_rms_V(self) -> float:
        return sqrt(4.0 * K_B * self.T0_K * self.R0_Ohm)

    @property
    def johnson_sv_V2_per_Hz(self) -> float:
        return 4.0 * K_B * self.T0_K * self.R0_Ohm

    @property
    def me_electronic(self) -> float:
        return sqrt(self.eqp_J / (K_B * self.T0_K))

    @property
    def delta_J(self) -> float:
        """BCS gap: Delta = 1.764 k_B Tc."""
        return 1.764 * K_B * self.Tc_K

    @property
    def eqp_J(self) -> float:
        """Paper convention: Eqp ~ Delta."""
        return self.delta_J

    @property
    def nj_scale(self) -> float:
        return sqrt(16.0 * K_B * self.T0_K / self.P0_W)

    @property
    def nj_thermal_scale(self) -> float:
        return sqrt(4.0 * K_B * self.T0_K * self.P0_W)

    def n_phonon(self) -> Tuple[complex, complex, complex]:
        return (0.0 + 0.0j, 0.0 + 0.0j, self.phonon_power_rms_W + 0.0j)

    def n_johnson_A(self) -> Tuple[complex, complex, complex]:
        return (self.nj_scale + 0.0j, 0.0 + 0.0j, -self.nj_thermal_scale + 0.0j)

    def n_johnson_phi(self) -> Tuple[complex, complex, complex]:
        return (0.0 + 0.0j, 1j * self.nj_scale, 0.0 + 0.0j)

    def fractional_frequency_psd_from_phase(self, s_phi_per_hz: float) -> float:
        return s_phi_per_hz / (4.0 * self.Qr) ** 2

    def m_tls_from_ratio(self, sf_over_f2_tls: float, sf_over_f2_j: float) -> float:
        if sf_over_f2_j <= 0.0:
            raise ValueError("sf_over_f2_j must be > 0")
        return sqrt(sf_over_f2_tls / sf_over_f2_j)

    def sf_over_f0sq_tls(self) -> float:
        """Semi-empirical TLS model used in wiki TLS estimate page."""
        power_term = (1.0 + (self.tls_Pint_W / self.tls_Pc_W)) ** (-self.tls_power_exponent_m)
        return (
            self.tls_A_scale
            * self.tls_F_participation
            * self.tls_tan_delta
            * (self.tls_nu_Hz ** (-self.tls_beta))
            * power_term
        )

    def sf_over_f0sq_johnson_ref(self) -> float:
        """Johnson fractional-frequency PSD from reference phase PSD."""
        return self.fractional_frequency_psd_from_phase(self.sphi_j_ref_per_hz)

    def n_tls_phi(self, m_tls: float) -> Tuple[complex, complex, complex]:
        _, n2, _ = self.n_johnson_phi()
        return (0.0 + 0.0j, m_tls * n2, 0.0 + 0.0j)

    def n_electronic_A(self) -> Tuple[complex, complex, complex]:
        a1, a2, a3 = self.n_johnson_A()
        me = self.me_electronic
        return (me * a1, me * a2, me * a3)

    def n_electronic_phi(self) -> Tuple[complex, complex, complex]:
        p1, p2, p3 = self.n_johnson_phi()
        me = self.me_electronic
        return (me * p1, me * p2, me * p3)

    def m_matrix(self, f_hz: float = 1.0) -> Tuple[Tuple[complex, complex, complex], Tuple[complex, complex, complex], Tuple[complex, complex, complex]]:
        """Compute full 3x3 complex M(omega) at demodulated frequency f_hz.

        Assumption used here:
        - Linearized Eq. (15)-style dynamics in (r, phi, T) define state matrix A.
        - Frequency-domain matrix is M(omega) = i*omega*I - A.
        """
        w0 = 2.0 * pi * self.f0_Hz
        omega = 2.0 * pi * f_hz
        c = self.C_J_per_K
        g = self.G_W_per_K

        a11 = -((w0 / (2.0 * self.Qr)) + (w0 * self.beta_A / (2.0 * self.Qi)))
        a12 = 0.0
        a13 = -(w0 * self.alpha_A / (self.T0_K * self.Qi))

        a21 = +(w0 * self.beta_phi / (2.0 * self.Qi))
        a22 = -(w0 / (2.0 * self.Qr))
        a23 = +(w0 * self.alpha_phi / (self.T0_K * self.Qi))

        a31 = +(self.P0_W / c) * (1.0 + self.beta_A / 2.0)
        a32 = 0.0
        a33 = +(self.P0_W * self.alpha_A / (c * self.T0_K)) - (g / c)

        iw = 1j * omega
        return (
            (iw - a11, -a12, -a13),
            (-a21, iw - a22, -a23),
            (-a31, -a32, iw - a33),
        )

    def estimates(self) -> Dict[str, float]:
        return {
            "f0_Hz": self.f0_Hz,
            "detuning_Hz": self.detuning_Hz,
            "f_demod_Hz": self.f_demod_Hz,
            "count_rate_Hz": self.count_rate_Hz,
            "pileup_probability_max": self.pileup_probability_max,
            "ho_activity_per_m3_Hz": self.ho_activity_per_m3_Hz,
            "absorber_volume_m3": self.absorber_volume_m3,
            "absorber_edge_m": self.absorber_edge_m,
            "absorber_length_m": self.absorber_length_m,
            "absorber_width_m": self.absorber_width_m,
            "absorber_thickness_m": self.absorber_thickness_m,
            "membrane_length_m": self.membrane_length_m,
            "membrane_width_m": self.membrane_width_m,
            "membrane_span_m": self.membrane_span_m,
            "cap_thickness_m": self.cap_thickness_m,
            "membrane_thickness_m": self.membrane_thickness_m,
            "leg_length_m": self.leg_length_m,
            "leg_thickness_m": self.leg_thickness_m,
            "leg_width_m": self.leg_width_m,
            "C_J_per_K": self.C_J_per_K,
            "G_W_per_K": self.G_W_per_K,
            "tau_th_s": self.tau_th_s,
            "tau_target_from_rate_s": self.pileup_probability_max / self.count_rate_Hz,
            "tau_error_fraction": 0.0,
            "tau_res_s": self.tau_res_s,
            "tau_ratio_res_over_th": self.tau_res_s / self.tau_th_s,
            "L_geo_H": self.L_geo_H,
            "L_total_H": self.L_total_H,
            "C_res_F": self.C_res_F,
            "Z0_res_Ohm": self.Z0_res_Ohm,
            "R0_Ohm": self.R0_Ohm,
            "tls_F_participation": self.tls_F_participation,
            "tls_tan_delta": self.tls_tan_delta,
            "tls_beta": self.tls_beta,
            "tls_A_scale": self.tls_A_scale,
            "tls_power_exponent_m": self.tls_power_exponent_m,
            "tls_Pint_W": self.tls_Pint_W,
            "tls_Pc_W": self.tls_Pc_W,
            "tls_nu_Hz": self.tls_nu_Hz,
            "sphi_j_ref_per_hz": self.sphi_j_ref_per_hz,
            "sf_over_f0sq_johnson_ref": self.sf_over_f0sq_johnson_ref(),
            "sf_over_f0sq_tls_model": self.sf_over_f0sq_tls(),
            "alpha_A": self.alpha_A,
            "alpha_phi": self.alpha_phi,
            "beta_A": self.beta_A,
            "beta_phi": self.beta_phi,
            "Tc_K": self.Tc_K,
            "delta_J": self.delta_J,
            "eqp_J": self.eqp_J,
            "phonon_power_rms_W": self.phonon_power_rms_W,
            "johnson_voltage_rms_V": self.johnson_voltage_rms_V,
            "johnson_sv_V2_per_Hz": self.johnson_sv_V2_per_Hz,
            "M_e": self.me_electronic,
            "N_J_scale": self.nj_scale,
            "N_J_thermal_scale": self.nj_thermal_scale,
        }


def nominal_sensor() -> Sensor:
    """Project nominal parameter set for wiki calculations.

    Wiki references:
    - wiki/project.html (f0=1.0 GHz spec)
    - wiki/physics.html (timing scales)
    - wiki/noise-*.html (noise vectors)
    """
    return Sensor(
        T0_K=0.100,
        Tb_K=0.100,
        count_rate_Hz=50.0,
        pileup_probability_max=5.0e-4,
        ho_activity_per_m3_Hz=6.25e17,
        kid_length_m=220e-6,
        kid_width_m=220e-6,
        membrane_margin_m=20e-6,
        leg_count=4,
        leg_width_m=0.50e-6,
        cap_thickness_m=1.0e-6,
        membrane_thickness_m=1.0e-6,
        cv_absorber_J_per_m3K=0.075,
        kappa_leg_W_per_mK=1.5e-3,
        tls_F_participation=2.0e-3,
        tls_tan_delta=1.0e-3,
        tls_beta=0.5,
        tls_A_scale=1.0,
        tls_power_exponent_m=0.5,
        tls_Pint_W=1.0e-12,
        tls_Pc_W=1.0e-12,
        tls_nu_Hz=1.0e5,
        sphi_j_ref_per_hz=1.0e-18,
        f0_Hz=1.0e9,
        Qr=3000.0,
        Qi=6000.0,
        kinetic_inductance_fraction=0.5,
        kid_trace_length_m=10.0e-3,
        kid_trace_width_m=2.0e-6,
        alpha_A=0.1,
        alpha_phi=60.0,
        beta_A=0.0,
        beta_phi=0.0,
        Tc_K=2.0,
        P0_W=5.0e-13,
        detuning_Hz=0.0,
        f_demod_Hz=0.0,
    )
