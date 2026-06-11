#!/usr/bin/env python3

import numpy as np

from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg, tebd


def build_tfi_model(L, J, g, bc_MPS="finite"):
    """
    TeNPy TFIChain convention is approximately

        H = - J sum_i Sigmax_i Sigmax_{i+1}
            - g sum_i Sigmaz_i

    This is just a basis convention.

    For the symmetry-breaking order parameter in this convention,
    use Sigmax rather than Sigmaz.

    Ordered/broken phase:     |J| > |g|
    Disordered/symmetric:     |g| > |J|
    Critical point:           |g/J| = 1
    """
    model_params = {
        "L": L,
        "J": J,
        "g": g,
        "bc_MPS": bc_MPS,
        "bc_x": "open",
        "conserve": None,
    }
    return TFIChain(model_params)


def ground_state_tfi(L, J, g, chi_max=100, svd_min=1e-12):
    """
    Compute ground state of TFIChain with DMRG.
    """
    M = build_tfi_model(L, J, g)

    # Product-state initial guess.
    # For g > J, this is close to the paramagnetic/disordered phase.
    product_state = ["up"] * L
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    dmrg_params = {
        "mixer": True,
        "max_sweeps": 12,
        "trunc_params": {
            "chi_max": chi_max,
            "svd_min": svd_min,
        },
    }

    info = dmrg.run(psi, M, dmrg_params)
    return psi, M, info


def measure_observables(psi):
    """
    In TeNPy's TFIChain convention, the Ising order parameter is Sigmax.
    The transverse-field direction is Sigmaz.

    We measure:
      mx  = average <Sigmax>
      mx2 = structure-factor-like squared magnetization
            1/L^2 sum_ij <Sigmax_i Sigmax_j>
      C(r) from correlations if desired.
    """
    L = psi.L

    sx = np.array(psi.expectation_value("Sigmax"), dtype=float)
    sz = np.array(psi.expectation_value("Sigmaz"), dtype=float)

    mx = np.mean(sx)
    mz = np.mean(sz)

    # Correlation matrix of order parameter.
    # This can be a bit expensive for large L, but fine for first tests.
    Cxx = psi.correlation_function("Sigmax", "Sigmax")
    mx2 = np.sum(Cxx).real / (L * L)

    # A simple connected correlation from the center.
    j0 = L // 2
    center_corr = Cxx[j0, :].real

    return {
        "mx": mx,
        "mz": mz,
        "mx2": mx2,
        "center_corr": center_corr,
        "sx_profile": sx,
        "sz_profile": sz,
    }


def run_quench(
    L=80,
    J=1.0,
    gi=1.10,
    gf=0.90,
    dt=0.02,
    tmax=10.0,
    chi_max_dmrg=100,
    chi_max_tebd=200,
    svd_min=1e-12,
    output_file="ising_quench_tenpy.dat",
):
    """
    Quench from gi > 1 to gf < 1.

    In this TeNPy convention:
        H = -J XX - g Z

    So:
        gi > 1: paramagnetic/symmetric
        gf < 1: ferromagnetic/broken
    """

    print("Preparing initial ground state...")
    psi, M_i, dmrg_info = ground_state_tfi(
        L=L,
        J=J,
        g=gi,
        chi_max=chi_max_dmrg,
        svd_min=svd_min,
    )

    print("Initial ground-state energy:", dmrg_info["E"])

    print("Building final Hamiltonian...")
    M_f = build_tfi_model(L, J, gf)

    tebd_params = {
        "dt": dt,
        "N_steps": 1,
        "order": 2,
        "trunc_params": {
            "chi_max": chi_max_tebd,
            "svd_min": svd_min,
        },
    }

    engine = tebd.TEBDEngine(psi, M_f, tebd_params)

    n_steps = int(round(tmax / dt))

    with open(output_file, "w") as f:
        f.write("# t mx mz mx2 max_chi\n")

        obs = measure_observables(psi)
        f.write(
            f"{0.0:.10f} "
            f"{obs['mx']:.16e} "
            f"{obs['mz']:.16e} "
            f"{obs['mx2']:.16e} "
            f"{max(psi.chi)}\n"
        )

        print("# t mx mz mx2 max_chi")
        print(
            f"{0.0:.4f} "
            f"{obs['mx']:+.8e} "
            f"{obs['mz']:+.8e} "
            f"{obs['mx2']:+.8e} "
            f"{max(psi.chi)}"
        )

        for n in range(1, n_steps + 1):
            engine.run()
            t = n * dt

            if n % 10 == 0:
                obs = measure_observables(psi)

                f.write(
                    f"{t:.10f} "
                    f"{obs['mx']:.16e} "
                    f"{obs['mz']:.16e} "
                    f"{obs['mx2']:.16e} "
                    f"{max(psi.chi)}\n"
                )

                print(
                    f"{t:.4f} "
                    f"{obs['mx']:+.8e} "
                    f"{obs['mz']:+.8e} "
                    f"{obs['mx2']:+.8e} "
                    f"{max(psi.chi)}"
                )

    print(f"Saved data to {output_file}")


if __name__ == "__main__":
    run_quench(
        L=80,
        J=1.0,
        gi=1.10,
        gf=0.90,
        dt=0.02,
        tmax=10.0,
        chi_max_dmrg=100,
        chi_max_tebd=200,
        output_file="ising_symmetry_breaking_quench.dat",
    )
