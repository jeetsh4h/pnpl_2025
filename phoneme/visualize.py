import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from h5py import Dataset
from typing import Literal
from pandas import DataFrame
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mne._fiff.meas_info import Info

from config import DATA_DIR, EXAMPLE_SENSORS_MASK, SENSORS_SPEECH_MASK


def meg_label_visualization(
    tsv_data: DataFrame,
    meg_raw: Dataset,
    info: Info,
    start_time: int,
    end_time: int,
    title: str | None = None,
    show_phonemes: bool = False,
    apply_sensor_mask: bool = False,
    mask: list[int] | Literal["example", "notebook"] = "notebook",
    show_plot: bool = False,
):
    """
    Combine MEG data visualization and speech/silence labels.

    Parameters:
      - tsv_data: DataFrame with timing and labeling information.
      - meg_raw: MEG data array (channels x samples).
      - info: MNE Info object containing metadata (including sampling frequency).
      - start_time, end_time: Time window (in seconds) to visualize.
      - title: Optional title for the plots.
      - show_phonemes: Whether to show phoneme annotations.
      - apply_sensor_mask: force applying the sensor mask;
      - mask: List of sensor IDs to apply as a mask, or "example" for EXAMPLE_SENSORS_MASK, or "notebook" for SENSORS_SPEECH_MASK.
      - show_plot: Whether to display the plot interactively.
    """
    # --- Optionally apply sensor mask if using filtered data ---
    # refer to the notebook,,, it is just a list of sensor ids
    if apply_sensor_mask:
        if isinstance(mask, str):
            mask = EXAMPLE_SENSORS_MASK if mask == "example" else SENSORS_SPEECH_MASK
        try:
            meg_raw = meg_raw[mask, :]
        except Exception as e:
            print(f"Error applying sensor mask: {e}")

    # --- Process TSV data to build ground-truth labels ---
    tsv_data = tsv_data.copy()
    tsv_data["timemeg"] = tsv_data["timemeg"].astype(float)
    last_before = tsv_data[tsv_data["timemeg"] < start_time].iloc[-1:]
    window_data = tsv_data[
        (tsv_data["timemeg"] >= start_time) & (tsv_data["timemeg"] <= end_time)
    ]
    filtered_data = pd.concat([last_before, window_data]).sort_values("timemeg")
    if filtered_data.empty:
        filtered_data = pd.DataFrame(
            {"timemeg": [start_time, end_time], "speech_label": [0, 0]}
        )
    filtered_data["speech_label"] = 0
    filtered_data.loc[
        filtered_data["kind"].isin(["word", "phoneme"]), "speech_label"
    ] = 1

    # TODO: add type annotations later
    first_value = filtered_data.iloc[0]["speech_label"]
    plot_times = [start_time]
    plot_values = [first_value]
    plot_times.extend(filtered_data["timemeg"].tolist())
    plot_values.extend(filtered_data["speech_label"].tolist())
    plot_times.append(end_time)
    plot_values.append(plot_values[-1])
    plot_times = np.array(plot_times)
    plot_values = np.array(plot_values)

    # --- Extract the MEG segment ---
    sfreq = info["sfreq"]
    start_sample = int(start_time * sfreq)
    end_sample = int(end_time * sfreq)
    meg_segment = meg_raw[:, start_sample:end_sample]  # shape: (channels, samples)
    time_points = np.linspace(start_time, end_time, meg_segment.shape[1])

    # --- Plotting ---
    plot: tuple[Figure, list[Axes]] = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )
    fig, axs = plot
    meg_title = (
        f"{title}: MEG Data ({start_time}s to {end_time}s)"
        if title
        else f"MEG Data ({start_time}s to {end_time}s)"
    )
    # Define a list of line styles to cycle through
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 10))]
    n_channels = meg_segment.shape[0]
    for i in range(n_channels):
        style = line_styles[i % len(line_styles)]
        axs[0].plot(
            time_points,
            meg_segment[i, :],
            alpha=0.7,
            linewidth=3,
            linestyle=style,
            label=f"Ch {i + 1}" if n_channels <= 10 else None,  # avoid clutter
        )
    axs[0].set_title(meg_title)
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)
    if n_channels <= 10:
        axs[0].legend(loc="upper right", fontsize=8)

    # keeping it constant for all plots
    axs[0].set_ylim(-5e-11, 5e-11)

    axs[1].plot(
        plot_times,
        plot_values,
        drawstyle="steps-post",
        label="Speech (1) / Silence (0)",
        linewidth=2,
        color="black",
    )
    for _, row in filtered_data.iterrows():
        if pd.isna(row["kind"]) or row["kind"] == "silence":
            continue
        if pd.isna(row["timemeg"]) or pd.isna(row["segment"]):
            continue
        try:
            if (
                row["kind"] == "phoneme"
                and show_phonemes
                and start_time <= row["timemeg"] <= end_time
            ):
                axs[1].text(
                    row["timemeg"],
                    1.1,
                    str(row["segment"]),
                    fontsize=9,
                    rotation=45,
                    ha="center",
                )
            elif row["kind"] == "word" and start_time <= row["timemeg"] <= end_time:
                axs[1].text(
                    row["timemeg"],
                    1.3,
                    str(row["segment"]),
                    fontsize=10,
                    rotation=0,
                    ha="center",
                    color="blue",
                )
        except Exception as e:
            print(f"Warning: Could not plot annotation at time {row['timemeg']}: {e}")

    axs[1].set_ylabel("Speech / Silence")
    labels_title = (
        f"{title}: Labels ({start_time}s to {end_time}s)"
        if title
        else f"Labels ({start_time}s to {end_time}s)"
    )
    axs[1].set_title(labels_title)
    axs[1].set_ylim(-0.2, 1.5)
    axs[1].set_xlim(start_time, end_time)
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlabel("Time (s)")

    plt.tight_layout()

    filename_prefix = (
        title.replace(" ", "_") if title is not None else "meg_label_visualization"
    )
    filename = (
        f"{DATA_DIR.parent}/figures/{filename_prefix}_{start_time}_{end_time}.png"
    )
    plt.savefig(filename)
    print(f"Saved plot to {filename}")

    if show_plot:
        plt.show()

    plt.close(fig)
