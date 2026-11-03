#!/usr/bin/env python3
"""Create a GIF of walker 1 from a VMC walk_history.csv file."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import numpy as np


def read_parameters(path):
    parameters = {}
    with path.open(encoding="utf-8") as parameter_file:
        for line in parameter_file:
            fields = line.split()
            if len(fields) == 2 and not fields[0].startswith("#"):
                parameters[fields[0]] = fields[1]
    return parameters


def read_history(path):
    with path.open(newline="", encoding="utf-8") as history_file:
        metadata = {}
        data_lines = []
        for line in history_file:
            if line.startswith("# "):
                key, separator, value = line[2:].strip().partition("=")
                if separator:
                    metadata[key] = value
            elif line.strip():
                data_lines.append(line)

        reader = csv.DictReader(data_lines)
        if not reader.fieldnames or reader.fieldnames[0] != "step":
            raise ValueError(f"{path} must have a 'step' column")
        site_columns = reader.fieldnames[1:]
        if not site_columns or site_columns != [
            f"site_{i}" for i in range(1, len(site_columns) + 1)
        ]:
            raise ValueError("History columns must be step, site_1, site_2, ...")

        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} contains no configuration snapshots")

    steps = np.array([int(row["step"]) for row in rows])
    occupations = np.array(
        [[int(row[column]) for column in site_columns] for row in rows],
        dtype=int,
    )
    if np.any(occupations < 0):
        raise ValueError("Site occupations must be non-negative")
    return metadata, steps, occupations


def system_description(parameters):
    if parameters["dimension"] == "1":
        lattice = f"1D, L={parameters['L']}"
    else:
        lattice = f"2D, Lx={parameters['Lx']}, Ly={parameters['Ly']}"
    return (
        f"{lattice} | N={parameters['N']} | "
        f"U/t={parameters['U_over_t']} | "
        f"{parameters['trial_state']} | seed={parameters['seed']}"
    )


def make_boson_animation(ax, occupations, steps, dimension, parameters):
    site_count = occupations.shape[1]
    capacity = int(occupations.max())
    patches = []

    if dimension == 1:
        sites = np.arange(site_count)
        ax.plot(sites, np.zeros(site_count), color="0.65", linewidth=2, zorder=1)
        ax.scatter(sites, np.zeros(site_count), color="0.25", s=18, zorder=2)
        radius = 0.11
        for site in range(site_count):
            site_patches = []
            for boson in range(capacity):
                patch = Circle((site, radius + boson * 2.2 * radius), radius)
                patch.set_facecolor("tab:blue")
                patch.set_edgecolor("white")
                patch.set_linewidth(0.7)
                patch.set_visible(False)
                ax.add_patch(patch)
                site_patches.append(patch)
            patches.append(site_patches)
        ax.set_xlim(-0.5, site_count - 0.5)
        ax.set_ylim(-0.2, max(0.8, capacity * 2.2 * radius + radius + 0.2))
        ax.set_xticks(sites)
        ax.set_xlabel("Site")
        ax.set_yticks([])

        def update(frame):
            for site, site_patches in enumerate(patches):
                count = occupations[frame, site]
                for boson, patch in enumerate(site_patches):
                    patch.set_visible(boson < count)
                    patch.center = (site, radius + boson * 2.2 * radius)
            ax.set_title(f"Bosons at MC step {steps[frame]}")
            return [patch for site_patches in patches for patch in site_patches]

    else:
        lx = int(parameters["Lx"])
        ly = int(parameters["Ly"])
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(lx))
        ax.set_yticks(np.arange(ly))
        ax.set_xticks(np.arange(-0.5, lx, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, ly, 1), minor=True)
        ax.grid(which="minor", color="0.75", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.scatter(
            *np.meshgrid(np.arange(lx), np.arange(ly)),
            color="0.25",
            s=18,
            zorder=2,
        )

        columns = max(1, int(np.ceil(np.sqrt(capacity))))
        rows = max(1, int(np.ceil(capacity / columns)))
        pitch = 0.72 / max(columns, rows)
        radius = pitch * 0.38
        for site in range(site_count):
            x = site % lx
            y = site // lx
            site_patches = []
            for boson in range(capacity):
                column = boson % columns
                row = boson // columns
                offset_x = (column - (columns - 1) / 2) * pitch
                offset_y = (row - (rows - 1) / 2) * pitch
                patch = Circle((x + offset_x, y + offset_y), radius)
                patch.set_facecolor("tab:blue")
                patch.set_edgecolor("white")
                patch.set_linewidth(0.7)
                patch.set_visible(False)
                ax.add_patch(patch)
                site_patches.append((patch, offset_x, offset_y))
            patches.append(site_patches)

        ax.set_xlim(-0.5, lx - 0.5)
        ax.set_ylim(-0.5, ly - 0.5)
        ax.invert_yaxis()

        def update(frame):
            for site, site_patches in enumerate(patches):
                x = site % lx
                y = site // lx
                count = occupations[frame, site]
                for boson, (patch, offset_x, offset_y) in enumerate(site_patches):
                    patch.set_visible(boson < count)
                    patch.center = (x + offset_x, y + offset_y)
            ax.set_title(f"Bosons at MC step {steps[frame]}")
            return [
                patch
                for site_patches in patches
                for patch, _, _ in site_patches
            ]

    return update


def make_animation(history_path, output_path, interval, style="plot"):
    if style not in ("plot", "bosons"):
        raise ValueError(f"Unsupported visualization style {style!r}")
    metadata, steps, occupations = read_history(history_path)
    if metadata:
        parameters = metadata
    else:
        parameters_path = history_path.with_name("parameters.dat")
        if not parameters_path.is_file():
            raise FileNotFoundError(
                f"Could not find parameter metadata in {history_path} or "
                f"{parameters_path}"
            )
        parameters = read_parameters(parameters_path)
    dimension = int(parameters["dimension"])
    n_max = int(parameters["n_max"])

    fig, ax = plt.subplots()
    fig.suptitle(system_description(parameters))
    if style == "bosons":
        if dimension == 1:
            if occupations.shape[1] != int(parameters["L"]):
                raise ValueError("History site count does not match parameters.dat")
        elif dimension == 2:
            if occupations.shape[1] != int(parameters["Lx"]) * int(parameters["Ly"]):
                raise ValueError("History site count does not match parameters.dat")
        else:
            raise ValueError(f"Unsupported lattice dimension {dimension}")
        update = make_boson_animation(ax, occupations, steps, dimension, parameters)
        animation = FuncAnimation(
            fig,
            update,
            frames=len(steps),
            interval=interval,
            blit=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        animation.save(output_path, writer=PillowWriter(fps=1000 / interval))
        plt.close(fig)
        return

    if dimension == 1:
        if occupations.shape[1] != int(parameters["L"]):
            raise ValueError("History site count does not match parameters.dat")
        sites = np.arange(1, occupations.shape[1] + 1)
        bars = ax.bar(sites, occupations[0], color="tab:blue")
        ax.set(xlim=(0.5, len(sites) + 0.5), ylim=(0, n_max + 0.5))
        ax.set_xticks(sites)
        ax.set_xlabel("Site")
        ax.set_ylabel("Occupation")

        def update(frame):
            for bar, value in zip(bars, occupations[frame]):
                bar.set_height(value)
            ax.set_title(f"MC step {steps[frame]}")
            return bars

    elif dimension == 2:
        lx = int(parameters["Lx"])
        ly = int(parameters["Ly"])
        if occupations.shape[1] != lx * ly:
            raise ValueError("History site count does not match parameters.dat")
        image = ax.imshow(
            occupations[0].reshape(ly, lx),
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=max(n_max, 1),
            cmap="viridis",
        )
        fig.colorbar(image, ax=ax, label="Occupation")
        ax.set_xticks(np.arange(lx))
        ax.set(xlabel="x", ylabel="y")

        def update(frame):
            image.set_data(occupations[frame].reshape(ly, lx))
            ax.set_title(f"MC step {steps[frame]}")
            return (image,)

    else:
        raise ValueError(f"Unsupported lattice dimension {dimension}")

    animation = FuncAnimation(
        fig,
        update,
        frames=len(steps),
        interval=interval,
        blit=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    animation.save(output_path, writer=PillowWriter(fps=1000 / interval))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="Path to walk_history.csv")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output GIF (default: next to the history CSV)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="Milliseconds per frame (default: 100)",
    )
    parser.add_argument(
        "--style",
        choices=("plot", "bosons"),
        default="plot",
        help="Visualization style: occupation plot or circles on lattice (default: plot)",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    default_name = "walk_history_bosons.gif" if args.style == "bosons" else "walk_history.gif"
    output_path = args.output or args.history.with_name(default_name)
    make_animation(args.history, output_path, args.interval, args.style)
    print(f"Saved animation to {output_path}")


if __name__ == "__main__":
    main()
