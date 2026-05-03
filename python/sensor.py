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
from functools import cached_property
from math import log, pi, sqrt
from typing import Dict, Tuple
import numpy as np

K_B = 1.380649e-23
MU0 = 4.0e-7 * pi
J_PER_EV = 1.602176634e-19
N_A = 6.02214076e23
RHO_AU_KG_PER_M3 = 19300.0
M_AU_KG_PER_MOL = 0.19696657
HO163_HALF_LIFE_Y = 4570.0
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


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
    ho_in_au_atomic_fraction: float
    ho_decay_energy_J: float
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
    tau_qp_s: float
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

    @cached_property
    def au_number_density_per_m3(self) -> float:
        """Gold atomic number density (atoms/m^3)."""
        return (RHO_AU_KG_PER_M3 / M_AU_KG_PER_MOL) * N_A

    @cached_property
    def ho_number_density_per_m3(self) -> float:
        """Ho number density from chosen Ho/Au atomic fraction."""
        return self.ho_in_au_atomic_fraction * self.au_number_density_per_m3

    @cached_property
    def ho_decay_constant_per_s(self) -> float:
        """163Ho decay constant lambda in 1/s."""
        t_half_s = HO163_HALF_LIFE_Y * SECONDS_PER_YEAR
        return log(2.0) / t_half_s

    @cached_property
    def ho_activity_per_m3_Hz(self) -> float:
        """Derived volumetric activity from Ho number density."""
        return self.ho_decay_constant_per_s * self.ho_number_density_per_m3

    @cached_property
    def absorber_volume_m3(self) -> float:
        """Absorber volume required to meet count rate at chosen Ho density."""
        return self.count_rate_Hz / self.ho_activity_per_m3_Hz

    @cached_property
    def absorber_edge_m(self) -> float:
        """Cubic absorber edge from derived volume."""
        return self.absorber_volume_m3 ** (1.0 / 3.0)

    @cached_property
    def absorber_length_m(self) -> float:
        return self.absorber_edge_m

    @cached_property
    def absorber_width_m(self) -> float:
        return self.absorber_edge_m

    @cached_property
    def absorber_thickness_m(self) -> float:
        return self.absorber_edge_m

    @cached_property
    def membrane_length_m(self) -> float:
        """Derived membrane length from absorber + KID + margin."""
        return self.absorber_length_m + self.kid_length_m + self.membrane_margin_m

    @cached_property
    def membrane_width_m(self) -> float:
        """Derived membrane width from absorber + KID + margin."""
        return self.absorber_width_m + self.kid_width_m + self.membrane_margin_m

    @cached_property
    def membrane_span_m(self) -> float:
        """Characteristic span for leg scaling."""
        return max(self.membrane_length_m, self.membrane_width_m)

    @cached_property
    def leg_thickness_m(self) -> float:
        """Leg thickness derived from membrane thickness."""
        return self.membrane_thickness_m

    @cached_property
    def C_J_per_K(self) -> float:
        """Derived heat capacity from absorber geometry."""
        return self.cv_absorber_J_per_m3K * self.absorber_volume_m3

    @cached_property
    def tau_resolve_s(self) -> float:
        """Resolve window from pileup requirement: P ~ R * tau_resolve."""
        return self.pileup_probability_max / self.count_rate_Hz

    @cached_property
    def tau_th_s(self) -> float:
        """Thermal decay tau from pileup approximation: P ~ R * tau."""
        return self.pileup_probability_max / self.count_rate_Hz

    @cached_property
    def G_W_per_K(self) -> float:
        """Derived thermal conductance from C / tau."""
        return self.C_J_per_K / self.tau_th_s

    @cached_property
    def deltaT_abs_over_bath_K(self) -> float:
        """Absorber/KID operating-point temperature elevation over bath."""
        return self.P0_W / self.G_W_per_K

    @cached_property
    def deltaT_event_full_absorption_K(self) -> float:
        """Island temperature step from a fully absorbed Ho decay event."""
        return self.ho_decay_energy_J / self.C_J_per_K

    @cached_property
    def C_eV_per_mK(self) -> float:
        """Island heat capacity in eV/mK."""
        return (self.C_J_per_K / J_PER_EV) / 1.0e3

    @cached_property
    def C_ho_eV_per_mK(self) -> float:
        """Ho absorber heat capacity in eV/mK."""
        return (self.C_J_per_K / J_PER_EV) / 1.0e3

    @cached_property
    def ho_decay_energy_eV(self) -> float:
        """Ho decay event energy in eV."""
        return self.ho_decay_energy_J / J_PER_EV

    @cached_property
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

    @cached_property
    def tau_res_s(self) -> float:
        return self.Qr / (pi * self.f0_Hz)

    @cached_property
    def L_geo_H(self) -> float:
        """Approximate geometric inductance from trace length/width.

        Thin straight-trace approximation:
        L ~ mu0 * l * (ln(2l/w) + 0.5)
        """
        l = self.kid_trace_length_m
        w = self.kid_trace_width_m
        ratio = max((2.0 * l) / w, 1.000001)
        return MU0 * l * (log(ratio) + 0.5)

    @cached_property
    def L_total_H(self) -> float:
        """Total inductance from geometric inductance and kinetic fraction."""
        kfrac = self.kinetic_inductance_fraction
        if not (0.0 <= kfrac < 1.0):
            raise ValueError("kinetic_inductance_fraction must be in [0,1)")
        return self.L_geo_H / (1.0 - kfrac)

    @cached_property
    def C_res_F(self) -> float:
        """Resonator capacitance from f0 and L_total."""
        w0 = 2.0 * pi * self.f0_Hz
        return 1.0 / (w0 * w0 * self.L_total_H)

    @cached_property
    def Z0_res_Ohm(self) -> float:
        """Nominal resonator characteristic impedance at f0."""
        w0 = 2.0 * pi * self.f0_Hz
        return w0 * self.L_total_H

    @cached_property
    def R0_Ohm(self) -> float:
        """Series-equivalent resonator resistance from Z0 and Q."""
        return self.Z0_res_Ohm / self.Qr

    @cached_property
    def phonon_power_rms_W(self) -> float:
        return sqrt(4.0 * K_B * (self.Tb_K**2) * self.G_W_per_K)

    @cached_property
    def phonon_power_asd_device_W_per_rtHz(self) -> float:
        """Simple phonon power ASD at device temperature T0."""
        return sqrt(4.0 * K_B * (self.T0_K**2) * self.G_W_per_K)

    @cached_property
    def johnson_voltage_rms_V(self) -> float:
        return sqrt(4.0 * K_B * self.T0_K * self.R0_Ohm)

    @cached_property
    def johnson_sv_V2_per_Hz(self) -> float:
        return 4.0 * K_B * self.T0_K * self.R0_Ohm

    @cached_property
    def me_electronic(self) -> float:
        return sqrt(self.eqp_J / (K_B * self.T0_K))

    @cached_property
    def delta_J(self) -> float:
        """BCS gap: Delta = 1.764 k_B Tc."""
        return 1.764 * K_B * self.Tc_K

    @cached_property
    def eqp_J(self) -> float:
        """Paper convention: Eqp ~ Delta."""
        return self.delta_J

    @cached_property
    def nj_scale(self) -> float:
        return sqrt(16.0 * K_B * self.T0_K / self.P0_W)

    @cached_property
    def nj_thermal_scale(self) -> float:
        return sqrt(4.0 * K_B * self.T0_K * self.P0_W)

    @cached_property
    def tau_target_from_rate_s(self) -> float:
        """Symbol used in outputs to mirror rate-derived tau target."""
        return self.pileup_probability_max / self.count_rate_Hz

    @cached_property
    def tau_error_fraction(self) -> float:
        """Difference between solved tau and target tau (currently identical)."""
        if self.tau_target_from_rate_s == 0.0:
            return 0.0
        return (self.tau_th_s - self.tau_target_from_rate_s) / self.tau_target_from_rate_s

    @cached_property
    def tau_ratio_res_over_th(self) -> float:
        return self.tau_res_s / self.tau_th_s

    @cached_property
    def core_rule1_left_ratio(self) -> float:
        """Rule 1 left ratio: tau_qp / tau_res (should be << 1)."""
        return self.tau_qp_s / self.tau_res_s

    @cached_property
    def core_rule1_right_ratio(self) -> float:
        """Rule 1 right ratio: tau_res / tau_th (should be << 1)."""
        return self.tau_res_s / self.tau_th_s

    @cached_property
    def core_rule1_ok(self) -> bool:
        """Practical check for '<<' using 0.1 threshold on both ratios."""
        return (self.core_rule1_left_ratio < 0.1) and (self.core_rule1_right_ratio < 0.1)

    @cached_property
    def core_rule2_ratio(self) -> float:
        """Rule 2 ratio: tau_res / (tau_th/3), must be <= 1."""
        return self.tau_res_s / (self.tau_th_s / 3.0)

    @cached_property
    def core_rule2_ok(self) -> bool:
        return self.core_rule2_ratio <= 1.0

    @cached_property
    def sf_over_f0sq_johnson_ref(self) -> float:
        return self.fractional_frequency_psd_from_phase(self.sphi_j_ref_per_hz)

    @cached_property
    def sf_over_f0sq_tls_model(self) -> float:
        return self.sf_over_f0sq_tls()

    @cached_property
    def m_tls(self) -> float:
        """TLS vector scale from TLS/Johnson fractional-frequency-noise ratio."""
        return self.m_tls_from_ratio(self.sf_over_f0sq_tls_model, self.sf_over_f0sq_johnson_full)

    def n_phonon(self) -> Tuple[complex, complex, complex]:
        return (0.0 + 0.0j, 0.0 + 0.0j, self.phonon_power_rms_W + 0.0j)

    def n_johnson_A(self) -> Tuple[complex, complex, complex]:
        return (self.nj_scale + 0.0j, 0.0 + 0.0j, -self.nj_thermal_scale + 0.0j)

    def n_johnson_phi(self) -> Tuple[complex, complex, complex]:
        return (0.0 + 0.0j, 1j * self.nj_scale, 0.0 + 0.0j)

    def _propagate_noise_vector(self, n_vec: Tuple[complex, complex, complex], f_hz: float) -> np.ndarray:
        """Propagate one source vector through Y = M^{-1} N."""
        m = self.m_matrix_array(f_hz)
        n = np.array(n_vec, dtype=complex)
        return np.linalg.solve(m, n)

    @cached_property
    def y_johnson_A(self) -> np.ndarray:
        return self._propagate_noise_vector(self.n_johnson_A(), self.f_demod_Hz)

    @cached_property
    def y_johnson_phi(self) -> np.ndarray:
        return self._propagate_noise_vector(self.n_johnson_phi(), self.f_demod_Hz)

    @cached_property
    def y_phonon(self) -> np.ndarray:
        return self._propagate_noise_vector(self.n_phonon(), self.f_demod_Hz)

    @cached_property
    def y_tls_phi(self) -> np.ndarray:
        """TLS-driven output vector from phase-like TLS source at f_demod."""
        return self._propagate_noise_vector(self.n_tls_phi(self.m_tls), self.f_demod_Hz)

    @cached_property
    def sphi_johnson_full_per_hz(self) -> float:
        """Johnson phase-noise PSD from full matrix model at f_demod."""
        yja_phi = self.y_johnson_A[1]
        yjp_phi = self.y_johnson_phi[1]
        return float(abs(yja_phi) ** 2 + abs(yjp_phi) ** 2)

    @cached_property
    def sphi_tls_per_hz(self) -> float:
        """TLS phase-noise PSD from full matrix model at f_demod."""
        return float(abs(self.y_tls_phi[1]) ** 2)

    @cached_property
    def asd_phi_tls_per_rtHz(self) -> float:
        """TLS phase-noise ASD from full matrix model at f_demod."""
        return sqrt(self.sphi_tls_per_hz)

    @cached_property
    def dphi_df_detuning_per_hz(self) -> float:
        """Approximate local phase slope including detuning dependence."""
        x = self.detuning_Hz / self.f0_Hz
        return (4.0 * self.Qr / self.f0_Hz) / (1.0 + (2.0 * self.Qr * x) ** 2)

    @cached_property
    def dfr_dT_Hz_per_K(self) -> float:
        """Resonance-frequency thermal slope from alpha_phi definition."""
        # From alpha_phi = -2 Qi (T/fr) (dfr/dT), with fr ~ f0 at operating point.
        return -(self.alpha_phi * self.f0_Hz) / (2.0 * self.Qi * self.T0_K)

    @cached_property
    def dT_dE_K_per_J(self) -> float:
        """Island temperature rise per absorbed energy."""
        return 1.0 / self.C_J_per_K

    @cached_property
    def dphi_dE_rad_per_J(self) -> float:
        """Simple chain estimate: dphi/dE = (dphi/df)(dfr/dT)(dT/dE)."""
        return self.dphi_df_detuning_per_hz * self.dfr_dT_Hz_per_K * self.dT_dE_K_per_J

    @cached_property
    def deltafr_event_Hz(self) -> float:
        """Resonance-frequency shift for one absorbed Ho event."""
        return self.dfr_dT_Hz_per_K * self.deltaT_event_full_absorption_K

    @cached_property
    def deltaphi_event_rad(self) -> float:
        """Phase shift estimate for one absorbed Ho event."""
        return self.dphi_dE_rad_per_J * self.ho_decay_energy_J

    @cached_property
    def phonon_temp_asd_device_K_per_rtHz(self) -> float:
        """Low-frequency island temperature ASD from phonon power ASD and G."""
        return self.phonon_power_asd_device_W_per_rtHz / self.G_W_per_K

    @cached_property
    def phonon_energy_asd_device_J_per_rtHz(self) -> float:
        """Equivalent low-frequency energy ASD from temperature ASD and C."""
        return self.C_J_per_K * self.phonon_temp_asd_device_K_per_rtHz

    @cached_property
    def asd_phi_phonon_simple_per_rtHz(self) -> float:
        """Simple low-frequency phase ASD from phonon energy ASD and dphi/dE."""
        return abs(self.dphi_dE_rad_per_J) * self.phonon_energy_asd_device_J_per_rtHz

    @cached_property
    def thermal_energy_fluct_rms_J(self) -> float:
        """Simple island thermal-energy fluctuation scale: sqrt(k_B T0^2 C)."""
        return sqrt(K_B * (self.T0_K**2) * self.C_J_per_K)

    @cached_property
    def thermal_energy_fluct_rms_eV(self) -> float:
        """Island thermal-energy fluctuation scale in eV."""
        return self.thermal_energy_fluct_rms_J / J_PER_EV

    @cached_property
    def sf_over_f0sq_johnson_full(self) -> float:
        """General conversion using detuning-dependent local phase slope."""
        slope = self.dphi_df_detuning_per_hz
        if slope == 0.0:
            return float("nan")
        return self.sphi_johnson_full_per_hz / ((self.f0_Hz**2) * (slope**2))

    @cached_property
    def sf_over_f0sq_johnson_simple(self) -> float:
        """Simplified on-resonance-style conversion with 4Qr slope."""
        return self.sphi_johnson_full_per_hz / ((4.0 * self.Qr) ** 2)

    @cached_property
    def m_phonon_over_johnson_phi(self) -> float:
        """Phase-quadrature ASD ratio at f_demod: phonon / Johnson-phi."""
        denom = abs(self.y_johnson_phi[1])
        if denom == 0.0:
            return float("nan")
        return float(abs(self.y_phonon[1]) / denom)

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

    def sf_over_f0sq_tls_at_hz(self, nu_hz: float) -> float:
        """Gao-style semi-empirical TLS fractional frequency-noise PSD at nu_hz."""
        power_term = (1.0 + (self.tls_Pint_W / self.tls_Pc_W)) ** (-self.tls_power_exponent_m)
        return (
            self.tls_A_scale
            * self.tls_F_participation
            * self.tls_tan_delta
            * (nu_hz ** (-self.tls_beta))
            * power_term
        )

    @cached_property
    def I0_rms_A(self) -> float:
        """RMS current from resonator dissipation relation P0 = I0^2 R0."""
        return sqrt(self.P0_W / self.R0_Ohm)

    @cached_property
    def sf_over_f0sq_tls_1hz(self) -> float:
        """TLS fractional frequency-noise PSD at 1 Hz offset from f0."""
        return self.sf_over_f0sq_tls_at_hz(1.0)

    @cached_property
    def s_deltaC_tls_1hz_F2_per_Hz(self) -> float:
        """Equivalent capacitance-fluctuation PSD at 1 Hz: S_dC = 4 C0^2 S_y."""
        c0 = self.C_res_F
        return 4.0 * (c0**2) * self.sf_over_f0sq_tls_1hz

    @cached_property
    def asd_deltaC_tls_1hz_F_per_rtHz(self) -> float:
        """Equivalent capacitance-fluctuation ASD at 1 Hz."""
        return sqrt(self.s_deltaC_tls_1hz_F2_per_Hz)

    @cached_property
    def sv_usb_tls_1hz_V2_per_Hz(self) -> float:
        """Equivalent USB capacitor-voltage PSD at f0 + 1 Hz from TLS C-noise."""
        c0 = self.C_res_F
        w0 = 2.0 * pi * self.f0_Hz
        return (self.I0_rms_A**2 / (w0 * w0 * c0 * c0)) * self.sf_over_f0sq_tls_1hz

    @cached_property
    def asd_v_usb_tls_1hz_V_per_rtHz(self) -> float:
        """Equivalent USB capacitor-voltage ASD at f0 + 1 Hz."""
        return sqrt(self.sv_usb_tls_1hz_V2_per_Hz)

    @cached_property
    def sv_usb_johnson_V2_per_Hz(self) -> float:
        """Johnson USB voltage PSD from Eq. (17) branch split."""
        return 0.5 * self.johnson_sv_V2_per_Hz

    @cached_property
    def asd_v_usb_johnson_V_per_rtHz(self) -> float:
        """Johnson USB voltage ASD."""
        return sqrt(self.sv_usb_johnson_V2_per_Hz)

    @cached_property
    def m_usb_tls_over_johnson_1hz(self) -> float:
        """USB voltage ASD ratio at 1 Hz offset: TLS / Johnson."""
        denom = self.asd_v_usb_johnson_V_per_rtHz
        if denom == 0.0:
            return float("nan")
        return self.asd_v_usb_tls_1hz_V_per_rtHz / denom

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
        """Compute 3x3 complex M(omega) using Eq. (13)-style normalized form.

        Notes:
        - Q is identified with loaded resonator Qr.
        - x_det is dimensionless detuning: detuning_Hz / f0_Hz.
        - omega is demodulated angular frequency (rad/s).
        """
        w0 = 2.0 * pi * self.f0_Hz
        omega = 2.0 * pi * f_hz
        q = self.Qr
        qi = self.Qi
        x_det = self.detuning_Hz / self.f0_Hz
        c = self.C_J_per_K
        g = self.G_W_per_K

        m11 = (2.0j * omega * qi / w0) + (qi / q) + self.beta_A + (4.0 * qi * q * x_det * x_det)
        m12 = -(4.0j * omega * qi * q * x_det / w0)
        m13 = -(2.0 * q * x_det * self.beta_A) + (2.0 * self.alpha_A / self.T0_K)

        m21 = +(4.0j * omega * qi * q * x_det / w0) - self.beta_phi
        m22 = (2.0j * omega * qi / w0) + (qi / q) + (4.0 * qi * q * x_det * x_det) + (2.0 * q * x_det * self.beta_phi)
        m23 = -(2.0 * self.alpha_phi / self.T0_K)

        # Eq. (13): third row is in power-balance form (no /C factor here).
        m31 = -((1.0 + self.beta_A / 2.0) * self.P0_W)
        m32 = +(q * x_det * (self.beta_A + 2.0) * self.P0_W)
        m33 = (1.0j * omega * c) + g - (self.P0_W * self.alpha_A / self.T0_K)

        return (
            (m11, m12, m13),
            (m21, m22, m23),
            (m31, m32, m33),
        )

    def m_matrix_array(self, f_hz: float = 1.0) -> np.ndarray:
        """M matrix as a 3x3 complex ndarray."""
        return np.array(self.m_matrix(f_hz), dtype=complex)

    @cached_property
    def d0_matrix(self) -> np.ndarray:
        """D0 = M(0)."""
        return self.m_matrix_array(0.0)

    @cached_property
    def d1_matrix(self) -> np.ndarray:
        """D1 = -i dM/domega, using central finite difference around omega=0.

        The derivative is with respect to angular frequency omega (rad/s).
        """
        df_hz = 1.0
        dw = 2.0 * pi * df_hz
        mp = self.m_matrix_array(+df_hz)
        mm = self.m_matrix_array(-df_hz)
        dmdw = (mp - mm) / (2.0 * dw)
        return -1j * dmdw

    @cached_property
    def mt_matrix(self) -> np.ndarray:
        """Time-domain dynamics matrix Mt = -D1^{-1} D0."""
        return -np.linalg.inv(self.d1_matrix) @ self.d0_matrix

    @cached_property
    def mt_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of Mt (1/s)."""
        return np.linalg.eigvals(self.mt_matrix)

    @cached_property
    def mt_eigenvalues_sorted(self) -> np.ndarray:
        """Eigenvalues sorted by real part descending (worst stability first)."""
        vals = np.array(self.mt_eigenvalues, dtype=complex)
        idx = np.argsort(np.real(vals))[::-1]
        return vals[idx]

    @cached_property
    def mt_max_real_part(self) -> float:
        return float(np.max(np.real(self.mt_eigenvalues)))

    @cached_property
    def mt_stable(self) -> bool:
        """Stability criterion: all eigenvalue real parts must be negative."""
        return bool(np.all(np.real(self.mt_eigenvalues) < 0.0))

    @cached_property
    def mt_pulse_shortening_ratio(self) -> float:
        """Pulse shortening ratio: (G/C) normalized by slowest |lambda|.

        Defined only for stable eigenvalues. Returns NaN otherwise.
        """
        if not self.mt_stable:
            return float("nan")
        slowest = float(np.min(np.abs(self.mt_eigenvalues)))
        gc_freq = self.G_W_per_K / self.C_J_per_K
        if slowest == 0.0:
            return float("nan")
        return gc_freq / slowest

    def estimates(self) -> Dict[str, float]:
        return {
            "f0_Hz": self.f0_Hz,
            "detuning_Hz": self.detuning_Hz,
            "f_demod_Hz": self.f_demod_Hz,
            "count_rate_Hz": self.count_rate_Hz,
            "pileup_probability_max": self.pileup_probability_max,
            "ho_in_au_atomic_fraction": self.ho_in_au_atomic_fraction,
            "au_number_density_per_m3": self.au_number_density_per_m3,
            "ho_number_density_per_m3": self.ho_number_density_per_m3,
            "ho_decay_constant_per_s": self.ho_decay_constant_per_s,
            "ho_activity_per_m3_Hz": self.ho_activity_per_m3_Hz,
            "ho_decay_energy_J": self.ho_decay_energy_J,
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
            "C_eV_per_mK": self.C_eV_per_mK,
            "C_ho_eV_per_mK": self.C_ho_eV_per_mK,
            "G_W_per_K": self.G_W_per_K,
            "deltaT_abs_over_bath_K": self.deltaT_abs_over_bath_K,
            "deltaT_event_full_absorption_K": self.deltaT_event_full_absorption_K,
            "dfr_dT_Hz_per_K": self.dfr_dT_Hz_per_K,
            "dT_dE_K_per_J": self.dT_dE_K_per_J,
            "dphi_dE_rad_per_J": self.dphi_dE_rad_per_J,
            "deltafr_event_Hz": self.deltafr_event_Hz,
            "deltaphi_event_rad": self.deltaphi_event_rad,
            "phonon_power_asd_device_W_per_rtHz": self.phonon_power_asd_device_W_per_rtHz,
            "phonon_temp_asd_device_K_per_rtHz": self.phonon_temp_asd_device_K_per_rtHz,
            "phonon_energy_asd_device_J_per_rtHz": self.phonon_energy_asd_device_J_per_rtHz,
            "asd_phi_phonon_simple_per_rtHz": self.asd_phi_phonon_simple_per_rtHz,
            "thermal_energy_fluct_rms_J": self.thermal_energy_fluct_rms_J,
            "thermal_energy_fluct_rms_eV": self.thermal_energy_fluct_rms_eV,
            "ho_decay_energy_eV": self.ho_decay_energy_eV,
            "tau_th_s": self.tau_th_s,
            "tau_target_from_rate_s": self.tau_target_from_rate_s,
            "tau_error_fraction": self.tau_error_fraction,
            "tau_res_s": self.tau_res_s,
            "tau_ratio_res_over_th": self.tau_ratio_res_over_th,
            "core_rule1_left_ratio": self.core_rule1_left_ratio,
            "core_rule1_right_ratio": self.core_rule1_right_ratio,
            "core_rule1_ok": float(self.core_rule1_ok),
            "core_rule2_ratio": self.core_rule2_ratio,
            "core_rule2_ok": float(self.core_rule2_ok),
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
            "sf_over_f0sq_johnson_ref": self.sf_over_f0sq_johnson_ref,
            "sphi_johnson_full_per_hz": self.sphi_johnson_full_per_hz,
            "sphi_tls_per_hz": self.sphi_tls_per_hz,
            "asd_phi_tls_per_rtHz": self.asd_phi_tls_per_rtHz,
            "dphi_df_detuning_per_hz": self.dphi_df_detuning_per_hz,
            "sf_over_f0sq_johnson_full": self.sf_over_f0sq_johnson_full,
            "sf_over_f0sq_johnson_simple": self.sf_over_f0sq_johnson_simple,
            "m_phonon_over_johnson_phi": self.m_phonon_over_johnson_phi,
            "sf_over_f0sq_tls_model": self.sf_over_f0sq_tls_model,
            "sf_over_f0sq_tls_1hz": self.sf_over_f0sq_tls_1hz,
            "I0_rms_A": self.I0_rms_A,
            "s_deltaC_tls_1hz_F2_per_Hz": self.s_deltaC_tls_1hz_F2_per_Hz,
            "asd_deltaC_tls_1hz_F_per_rtHz": self.asd_deltaC_tls_1hz_F_per_rtHz,
            "sv_usb_tls_1hz_V2_per_Hz": self.sv_usb_tls_1hz_V2_per_Hz,
            "asd_v_usb_tls_1hz_V_per_rtHz": self.asd_v_usb_tls_1hz_V_per_rtHz,
            "sv_usb_johnson_V2_per_Hz": self.sv_usb_johnson_V2_per_Hz,
            "asd_v_usb_johnson_V_per_rtHz": self.asd_v_usb_johnson_V_per_rtHz,
            "m_usb_tls_over_johnson_1hz": self.m_usb_tls_over_johnson_1hz,
            "m_tls": self.m_tls,
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
            "mt_eig1_real_per_s": float(np.real(self.mt_eigenvalues_sorted[0])),
            "mt_eig1_imag_per_s": float(np.imag(self.mt_eigenvalues_sorted[0])),
            "mt_eig2_real_per_s": float(np.real(self.mt_eigenvalues_sorted[1])),
            "mt_eig2_imag_per_s": float(np.imag(self.mt_eigenvalues_sorted[1])),
            "mt_eig3_real_per_s": float(np.real(self.mt_eigenvalues_sorted[2])),
            "mt_eig3_imag_per_s": float(np.imag(self.mt_eigenvalues_sorted[2])),
            "mt_max_real_part_per_s": self.mt_max_real_part,
            "mt_stable": float(self.mt_stable),
            "mt_pulse_shortening_ratio": self.mt_pulse_shortening_ratio,
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
        ho_in_au_atomic_fraction=0.01,
        ho_decay_energy_J=4.5e-16,
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
        tau_qp_s=5.0e-8,
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
