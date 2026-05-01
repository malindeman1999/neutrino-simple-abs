# TKID-on-Membrane Neutrino Mass Detector Design Specification

## 1. Overview

This document defines the design constraints for a **thermal kinetic inductance detector (TKID)** operating as a calorimeter for neutrino mass measurement using low-Q isotopes (e.g. Ho-163).

The detector operates in the **thermal regime**:
- Energy → absorber → temperature rise
- KID measures temperature via kinetic inductance shift

The design is optimized for:
- **T = 100 mK**
- **Energy resolution ≈ 0.1 eV**
- **Count rate ≈ 50 Hz per pixel**
- **Array size ≈ 10^5 pixels**

---

## 2. System Architecture

Each pixel consists of:

- Absorber (with radioactive isotope)
- Membrane (thermal isolation)
- KID resonator (thermometer)

Thermal topology:

Absorber ↔ KID ↔ Membrane ↔ Bath

---

## 3. Critical Timescale Hierarchy

The detector must satisfy:

τ_qp << τ_res << τ_th

Where:

- τ_qp = quasiparticle lifetime
- τ_res = resonator response time
- τ_th = thermal decay time

### Target values:

| Quantity | Target |
|--------|--------|
| τ_qp | < 1 µs |
| τ_res | 1–2 µs |
| τ_th | 5–10 µs |

---

## 4. Resonator Constraints

### Resonator time constant:

τ_res = Q / (π f₀)

### Requirement:

τ_res ≤ τ_th / 3

### Design targets:

| Parameter | Value |
|----------|------|
| Q | 1×10⁴ – 2×10⁴ |
| f₀ | 2–5 GHz |
| τ_res | 1–2 µs |

### Implications:

- Q cannot be arbitrarily large
- f₀ must remain in GHz range
- Resonator must not distort pulse shape

---

## 5. Absorber Design

| Parameter | Value |
|----------|------|
| Size | 1 µm × 10 µm × 10 µm |
| Volume | 1×10⁻¹⁶ m³ |
| Material | Au / Bi / SC hybrid |

### Requirements:

- Full containment of decay energy
- Fast thermalization
- Short quasiparticle lifetime

---

## 6. Heat Capacity Budget

Total heat capacity:

C_total ≈ 7–9 × 10⁻¹⁶ J/K

### Contributions:

| Component | Contribution |
|----------|-------------|
| Absorber | ~6.5×10⁻¹⁶ J/K |
| Membrane | ~1×10⁻¹⁶ J/K |
| KID | <1×10⁻¹⁷ J/K |

---

## 7. Thermal Conductance

Thermal time constant:

τ_th = C / G

### For τ_th ≈ 5–8 µs:

G ≈ 1×10⁻¹⁰ W/K

---

## 8. Thermal Topology Constraint

Must satisfy:

G_AK >> G_AB

Where:

- G_AK = absorber ↔ KID coupling
- G_AB = island ↔ bath coupling

### Requirement:

G_AK ≥ 10 × G_AB

### Purpose:

- Ensure single thermal node behavior
- Avoid internal thermal fluctuation noise

---

## 9. Membrane Design

| Parameter | Value |
|----------|------|
| Material | SiN |
| Thickness | 0.2–0.5 µm |
| Island size | 50–100 µm |
| Leg length | 50–200 µm |
| Leg width | 0.2–1 µm |

### Target:

G_AB ≈ 1×10⁻¹⁰ W/K

---

## 10. KID Design

### Key parameters:

| Parameter | Target |
|----------|--------|
| α (responsivity) | 100–300 |
| Q_i | 10⁴–10⁵ |
| Frequency | 2–5 GHz |

### Important:

- qp lifetime is NOT a primary design parameter
- must ensure:
  - low loss
  - stable density of states
  - minimal noise

---

## 11. Capacitor / TLS Constraints

TLS noise originates in the capacitor.

### TLS scaling:

S_f / f² ∝ F · tanδ_TLS / f^β

Where:

- F = dielectric participation ratio
- E-field drives TLS noise

---

### Design requirements:

- Minimize dielectric participation (F)
- Minimize electric field magnitude

---

### Capacitor design strategy:

| Requirement | Implementation |
|------------|---------------|
| Low electric field | Large gaps (µm scale) |
| Low participation | No deposited dielectrics |
| Low TLS | IDC or vacuum-gap capacitor |
| Maintain frequency | Limit total capacitance |

---

### Capacitance constraint:

- C ≈ 0.1–1 pF
- Must preserve f₀ = 2–5 GHz

---

## 12. Energy Resolution Model

Baseline (thermal + Johnson noise):

ΔE ≈ sqrt(k_B T² C)

With α:

ΔE ∝ (1 / √α) · sqrt(k_B T² C)

---

### Expected performance:

| Case | ΔE |
|------|----|
| Ideal | 3–5 meV |
| Realistic | 20–100 meV |

---

## 13. TLS Noise Impact

TLS contributes via frequency noise:

ΔE_TLS ∝ √S_f / (df/dE)

### Key point:

TLS depends on:
- capacitor geometry
- dielectric quality

NOT on:
- thermal design

---

### Design condition:

TLS noise at ~100–200 kHz must be below:

~10⁻¹⁸ fractional frequency noise

---

## 14. Count Rate Constraint

Pileup condition:

P ≈ R · τ_resolve

Require:

P << 1

---

### For 50 Hz operation:

- τ_resolve ≈ 5–10 µs
- consistent with τ_th design

---

## 15. Array Scaling

| Parameter | Value |
|----------|------|
| Pixels | 10⁵ |
| Rate per pixel | 50 Hz |
| Total rate | 5×10⁶ events/s |

---

### Runtime:

| Target events | Time |
|--------------|------|
| 10¹⁴ | ~0.5–3 years |
| 10¹⁵ | ~6 years |

---

## 16. Key Design Rules (Summary)

1. τ_qp << τ_res << τ_th
2. τ_res ≤ τ_th / 3
3. G_AK >> G_AB
4. Maintain GHz resonator frequency
5. Minimize TLS via capacitor design
6. Keep heat capacity minimal
7. Ensure phonon containment

---

## 17. Critical Risk Areas

- TLS noise (capacitor design)
- Phonon escape (membrane + geometry)
- Thermal coupling hierarchy
- Resonator bandwidth vs pulse time
- Fabrication uniformity across large arrays

---

## 18. Final Statement

This design is viable if and only if:

- Resonator response is faster than thermal pulse
- TLS noise is suppressed below thermal noise
- Thermal system behaves as a single node

Under these conditions, the detector can achieve:

- ~100 meV resolution
- ~50 Hz per pixel
- scalable neutrino mass sensitivity