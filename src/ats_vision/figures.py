"""Static diagnostics from the fixed first test clip and complete test cohort."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ats_vision.data import VideoData


def plot_results(
    output: Path,
    summaries: list[dict],
    test: VideoData,
    arrays: dict,
    decoded: np.ndarray,
) -> None:
    """Save comparative error and a prespecified reconstruction/trajectory example."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in summaries if row["condition"] == "all"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    labels = [row["policy"] for row in selected]
    colors = [
        "#9ca3af" if row["policy"] == "all_frames" else "#2563eb" for row in selected
    ]
    axes[0].bar(
        np.arange(len(selected)),
        [row["mean_position_mse"] for row in selected],
        color=colors,
    )
    axes[0].set_xticks(np.arange(len(selected)), labels, rotation=40, ha="right")
    axes[0].set_ylabel("Mean clip position MSE (lower is better)")
    axes[0].set_title("Same read budget; all_frames has extra observations")
    truth = test.states[0, :, :2]
    axes[1].plot(truth[:, 0], truth[:, 1], color="black", label="True position")
    for policy, color in (("uniform", "#16a34a"), ("ats", "#dc2626")):
        means = arrays[f"{policy}__mean"][0]
        observed = arrays[f"{policy}__acquired"][0]
        axes[1].plot(means[:, 0], means[:, 1], color=color, label=policy)
        axes[1].scatter(means[observed, 0], means[observed, 1], color=color, s=18)
    axes[1].set(
        xlabel="Normalized x",
        ylabel="Normalized y",
        title="First test clip; dots mark acquisitions",
    )
    axes[1].legend()
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)

    clocks = [0, test.frames.shape[1] // 2, test.frames.shape[1] - 1]
    figure, axes = plt.subplots(3, 3, figsize=(7, 7), layout="constrained")
    for column, clock in enumerate(clocks):
        for row, (name, values) in enumerate(
            (
                ("Stored frame", test.frames[0]),
                ("Clean target", test.clean_frames[0]),
                ("ATS decoder", decoded),
            )
        ):
            image = values[clock]
            if image.shape[0] == 1:
                axes[row, column].imshow(
                    image[0], cmap="gray", vmin=0, vmax=1, origin="lower"
                )
            elif image.shape[0] == 3:
                axes[row, column].imshow(
                    image.transpose(1, 2, 0), vmin=0, vmax=1, origin="lower"
                )
            else:
                axes[row, column].imshow(
                    image[0], cmap="gray", vmin=0, vmax=1, origin="lower"
                )
            suffix = "read" if arrays["ats__acquired"][0, clock] else "unread"
            axes[row, column].set_title(f"{name}: t={clock} ({suffix})", fontsize=9)
            axes[row, column].axis("off")
    figure.savefig(output / "first_clip_reconstruction.png", dpi=160)
    plt.close(figure)
