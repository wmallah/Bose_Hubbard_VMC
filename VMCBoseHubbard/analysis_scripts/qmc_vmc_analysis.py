from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_QMC_BASE_DIR = REPO_ROOT / "data" / "QMC"
DEFAULT_VMC_BASE_DIR = REPO_ROOT / "data" / "C" / "1D"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "analysis_output"

VMC_ANSATZ_NAME = "Jastrow_realspace"
VMC_PBC = True


@dataclass
class EnergyPoint:
    method: str
    ansatz: str
    L: int
    N: int
    U: float
    beta: Optional[float]
    pbc: bool
    mean_energy: float
    sem_energy: float
    mean_kinetic: Optional[float]
    sem_kinetic: Optional[float]
    mean_potential: Optional[float]
    sem_potential: Optional[float]
    source: str


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(x.size)


def ensure_output_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_qmc_filename(filepath: str | Path) -> Dict:
    """
    Example:
    K_square_1D_64L_63N_PBC_3.000U_32.00beta_200bins1000_seed2001.dat
    """
    filepath = Path(filepath)
    stem = filepath.stem
    parts = stem.split("_")

    observable = parts[0]

    L = None
    N = None
    U = None
    beta = None
    pbc = None

    for token in parts:
        m = re.fullmatch(r"(\d+)L", token)
        if m:
            L = int(m.group(1))
            continue

        m = re.fullmatch(r"(\d+)N", token)
        if m:
            N = int(m.group(1))
            continue

        m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)U", token)
        if m:
            U = float(m.group(1))
            continue

        m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)beta", token)
        if m:
            beta = float(m.group(1))
            continue

        if token == "PBC":
            pbc = True
        elif token == "OBC":
            pbc = False

    if observable not in {"K", "V"}:
        raise ValueError(f"Expected filename to start with K_ or V_: {filepath.name}")

    if L is None or N is None or U is None:
        raise ValueError(f"Could not parse required metadata from filename: {filepath.name}")

    return {
        "observable": observable,
        "L": L,
        "N": N,
        "U": U,
        "beta": beta,
        "pbc": pbc,
        "source": str(filepath),
    }


def load_1col_data(filepath: str | Path) -> np.ndarray:
    arr = np.loadtxt(filepath, dtype=float)
    return np.atleast_1d(arr)


def build_matching_qmc_pair(k_file: str | Path, v_file: str | Path) -> EnergyPoint:
    k_meta = parse_qmc_filename(k_file)
    v_meta = parse_qmc_filename(v_file)

    for key in ["L", "N", "U", "beta", "pbc"]:
        if k_meta[key] != v_meta[key]:
            raise ValueError(
                f"QMC file mismatch for key '{key}': "
                f"{k_meta[key]} (K) vs {v_meta[key]} (V)"
            )

    k_bins = load_1col_data(k_file)
    v_bins = load_1col_data(v_file)

    if k_bins.size != v_bins.size:
        raise ValueError(
            f"Bin count mismatch: {Path(k_file).name} has {k_bins.size}, "
            f"{Path(v_file).name} has {v_bins.size}"
        )

    e_bins = k_bins + v_bins

    return EnergyPoint(
        method="QMC",
        ansatz="QMC",
        L=k_meta["L"],
        N=k_meta["N"],
        U=k_meta["U"],
        beta=k_meta["beta"],
        pbc=k_meta["pbc"],
        mean_energy=float(np.mean(e_bins)),
        sem_energy=float(sem(e_bins)),
        mean_kinetic=float(np.mean(k_bins)),
        sem_kinetic=float(sem(k_bins)),
        mean_potential=float(np.mean(v_bins)),
        sem_potential=float(sem(v_bins)),
        source=f"{k_file} | {v_file}",
    )


def get_qmc_folder(base_dir: str | Path, L: int, N: int) -> Path:
    """
    Expected QMC location:
        base_dir / 1D_L{L}_N{N}
    """
    base_dir = Path(base_dir)
    return base_dir / f"1D_L{L}_N{N}"


def find_qmc_pairs(folder: str | Path) -> List[Tuple[Path, Path]]:
    folder = Path(folder)

    k_files: Dict[str, Path] = {}
    v_files: Dict[str, Path] = {}

    for f in folder.glob("*.dat"):
        name = f.name
        if name.startswith("K_"):
            k_files[name[2:]] = f
        elif name.startswith("V_"):
            v_files[name[2:]] = f

    shared_keys = sorted(set(k_files) & set(v_files))
    return [(k_files[key], v_files[key]) for key in shared_keys]


def load_all_qmc_points(folder: str | Path, *, L: int, N: int) -> List[EnergyPoint]:
    pairs = find_qmc_pairs(folder)
    points = []

    for kf, vf in pairs:
        pt = build_matching_qmc_pair(kf, vf)
        if pt.L == L and pt.N == N:
            points.append(pt)

    return points


def get_vmc_file(base_dir: str | Path, L: int, N: int) -> Path:
    """
    Expected VMC location:
        base_dir / L{L}_N{N} / jastrow_realspace / VMC_results.dat
    """
    base_dir = Path(base_dir)
    return base_dir / f"L{L}_N{N}" / "jastrow_realspace" / "VMC_results.dat"


def load_vmc_points(
    filepath: str | Path,
    *,
    L: int,
    N: int,
    pbc: bool = True,
    ansatz: str = "Jastrow_realspace",
) -> List[EnergyPoint]:
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found.")

    data = np.loadtxt(filepath, comments="#", dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(
            f"VMC file must have at least 3 columns (U, E_mean, E_sem): {filepath}"
        )

    U_vals = data[:, 0]
    E_vals = data[:, 1]
    dE_vals = data[:, 2]

    points: List[EnergyPoint] = []
    for U, E, dE in zip(U_vals, E_vals, dE_vals):
        points.append(
            EnergyPoint(
                method="VMC",
                ansatz=ansatz,
                L=L,
                N=N,
                U=float(U),
                beta=None,
                pbc=pbc,
                mean_energy=float(E),
                sem_energy=float(dE),
                mean_kinetic=None,
                sem_kinetic=None,
                mean_potential=None,
                sem_potential=None,
                source=str(filepath),
            )
        )

    return points


def sort_points_by_U(points: List[EnergyPoint]) -> List[EnergyPoint]:
    return sorted(points, key=lambda p: (p.U, p.method, p.ansatz))


def write_summary_csv(points: List[EnergyPoint], outpath: str | Path) -> None:
    outpath = Path(outpath)
    rows = [asdict(p) for p in sort_points_by_U(points)]

    if not rows:
        print(f"No rows to write for {outpath}")
        return

    with outpath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _group_points(points: List[EnergyPoint]) -> Dict[Tuple[str, str], List[EnergyPoint]]:
    grouped: Dict[Tuple[str, str], List[EnergyPoint]] = {}
    for pt in points:
        key = (pt.method, pt.ansatz)
        grouped.setdefault(key, []).append(pt)
    return grouped


def plot_energy_vs_U(points: List[EnergyPoint], outpath: str | Path, title: str) -> None:
    if not points:
        print(f"Skipping plot {outpath}: no points.")
        return

    grouped = _group_points(points)

    plt.figure(figsize=(8, 6))
    for (method, ansatz), group in grouped.items():
        group = sorted(group, key=lambda p: p.U)
        U = np.array([p.U for p in group], dtype=float)
        E = np.array([p.mean_energy for p in group], dtype=float)
        dE = np.array([p.sem_energy for p in group], dtype=float)

        plt.errorbar(U, E, yerr=dE, marker="o", linestyle="-", capsize=3,
                     label=f"{method} ({ansatz})")

    plt.xlabel("U")
    plt.ylabel("Energy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_energy_per_site_vs_U(points: List[EnergyPoint], outpath: str | Path, title: str) -> None:
    if not points:
        print(f"Skipping plot {outpath}: no points.")
        return

    grouped = _group_points(points)

    plt.figure(figsize=(8, 6))
    for (method, ansatz), group in grouped.items():
        group = sorted(group, key=lambda p: p.U)
        U = np.array([p.U for p in group], dtype=float)
        E = np.array([p.mean_energy / p.L for p in group], dtype=float)
        dE = np.array([p.sem_energy / p.L for p in group], dtype=float)

        plt.errorbar(U, E, yerr=dE, marker="o", linestyle="-", capsize=3,
                     label=f"{method} ({ansatz})")

    plt.xlabel("U")
    plt.ylabel("Energy per site")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_delta_E_vmc_minus_qmc(points: List[EnergyPoint], outpath: str | Path, title: str) -> None:
    qmc = {p.U: p for p in points if p.method == "QMC"}
    vmc = {p.U: p for p in points if p.method == "VMC"}

    common_U = sorted(set(qmc) & set(vmc))
    if not common_U:
        print(f"Skipping plot {outpath}: no matching VMC/QMC U values.")
        return

    U = np.array(common_U, dtype=float)
    dE = np.array([vmc[u].mean_energy - qmc[u].mean_energy for u in common_U], dtype=float)
    ddE = np.array(
        [np.sqrt(vmc[u].sem_energy**2 + qmc[u].sem_energy**2) for u in common_U],
        dtype=float,
    )

    plt.figure(figsize=(8, 6))
    plt.errorbar(U, dE, yerr=ddE, marker="o", linestyle="-", capsize=3)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("U")
    plt.ylabel(r"$E_{\mathrm{VMC}} - E_{\mathrm{QMC}}$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare QMC and VMC energies for one system size.")
    parser.add_argument("L", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("--qmc-base-dir", type=Path, default=DEFAULT_QMC_BASE_DIR)
    parser.add_argument("--vmc-base-dir", type=Path, default=DEFAULT_VMC_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    L = args.L
    N = args.N
    qmc_base_dir = args.qmc_base_dir.resolve()
    vmc_base_dir = args.vmc_base_dir.resolve()
    output_dir = ensure_output_dir(args.output_dir.resolve() / f"L{L}_N{N}")

    qmc_folder = get_qmc_folder(qmc_base_dir, L, N)
    vmc_file = get_vmc_file(vmc_base_dir, L, N)

    print(f"SCRIPT_DIR   = {SCRIPT_DIR}")
    print(f"REPO_ROOT    = {REPO_ROOT}")
    print(f"QMC_BASE_DIR = {qmc_base_dir}")
    print(f"QMC_FOLDER   = {qmc_folder}")
    print(f"VMC_BASE_DIR = {vmc_base_dir}")
    print(f"VMC_FILE     = {vmc_file}")
    print(f"OUTPUT_DIR   = {output_dir}")
    print(f"SYSTEM       = L={L}, N={N}")
    print()

    if qmc_folder.exists():
        print(f"Found QMC folder: {qmc_folder}")
        qmc_points = load_all_qmc_points(qmc_folder, L=L, N=N)
    else:
        print(f"QMC folder not found: {qmc_folder}")
        qmc_points = []

    print(f"Loaded {len(qmc_points)} QMC points")

    if vmc_file.exists():
        print(f"Found VMC file: {vmc_file}")
        vmc_points = load_vmc_points(
            vmc_file,
            L=L,
            N=N,
            pbc=VMC_PBC,
            ansatz=VMC_ANSATZ_NAME,
        )
    else:
        print(f"VMC file not found: {vmc_file}")
        vmc_points = []

    print(f"Loaded {len(vmc_points)} VMC points")

    all_points = qmc_points + vmc_points
    print(f"Loaded {len(all_points)} total points")

    write_summary_csv(all_points, output_dir / "energy_summary.csv")

    plot_energy_vs_U(
        all_points,
        output_dir / "energy_vs_U.png",
        title=f"QMC and VMC energies (L={L}, N={N})",
    )
    plot_energy_per_site_vs_U(
        all_points,
        output_dir / "energy_per_site_vs_U.png",
        title=f"QMC and VMC energies per site (L={L}, N={N})",
    )
    plot_delta_E_vmc_minus_qmc(
        all_points,
        output_dir / "delta_E_vmc_minus_qmc.png",
        title=f"VMC minus QMC (L={L}, N={N})",
    )


if __name__ == "__main__":
    main()