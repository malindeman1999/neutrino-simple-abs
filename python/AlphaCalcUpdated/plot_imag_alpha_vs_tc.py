import numpy as np
import matplotlib.pyplot as plt

from alpha_calc import compute_alpha


def main():
    # Fixed model settings for the sweep.
    tdb_k = 275.0
    w0_rad_s = 1.0e9 * 2.0 * np.pi

    # Tc sweep on a log scale from 0.5 K to 20 K.
    tc_values_k = np.logspace(np.log10(0.5), np.log10(20.0), 100)

    imag_thick = []
    imag_extreme = []
    imag_thin = []

    for tc_k in tc_values_k:
        constants = {
            "Tc": float(tc_k),
            "Tdb": tdb_k,
            "w0": w0_rad_s,
            "type": "Tc_sweep",
        }
        t_eval_k = 0.1 * tc_k

        alpha_set, _, _ = compute_alpha(
            constants,
            [t_eval_k],
            approximate_sigma=False,
            approximate_Delta=False,
        )

        # alpha_set columns are: [Thick Local, Extreme Anomalous, Thin]
        imag_thick.append(np.imag(alpha_set[0, 0]))
        imag_extreme.append(np.imag(alpha_set[0, 1]))
        imag_thin.append(np.imag(alpha_set[0, 2]))

    imag_thick = np.array(imag_thick)
    imag_extreme = np.array(imag_extreme)
    imag_thin = np.array(imag_thin)

    plt.figure(figsize=(10, 6))
    plt.semilogx(tc_values_k, imag_thick, label="Thick Local", linewidth=2)
    plt.semilogx(tc_values_k, imag_extreme, label="Extreme Anomalous", linewidth=2)
    plt.semilogx(tc_values_k, imag_thin, label="Thin", linewidth=2)
    plt.xlabel("Tc (K)")
    plt.ylabel("Imaginary Alpha at T = 0.1 Tc")
    plt.title("Imaginary Alpha vs Tc (Exact, T = 0.1 Tc)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
