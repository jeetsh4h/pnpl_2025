import numpy as np
import pandas as pd
from h5py import File
from mne import create_info
import matplotlib.pyplot as plt
from pnpl.datasets import LibriBrainSpeech

# types
from h5py import Dataset
from pandas import DataFrame
from mne._fiff.meas_info import Info
# some numpy types are not imported explicitly
# because types like float32 float64 can be ambiguous
# about their module origin

from phoneme.config import DATA_DIR, EXAMPLE_SENSORS_MASK


def meg_label_visualization(
    tsv_data: DataFrame,
    meg_raw: Dataset,
    info: Info,
    start_time: int,
    end_time: int,
    title: str | None = None,
    show_phonemes: bool = False,
    apply_sensor_mask: bool = False,
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
      - show_plot: Whether to display the plot interactively.
    """
    # --- Optionally apply sensor mask if using filtered data ---
    # refer to the notebook,,, it is just a list of sensor ids
    if apply_sensor_mask:
        try:
            meg_raw = meg_raw[EXAMPLE_SENSORS_MASK, :]
        except NameError:
            print("SMALL_SENSORS_MASK is not defined. Proceeding without sensor mask.")

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
    fig, axs = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )
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
    plt.savefig(
        f"{DATA_DIR.parent}/figures/{filename_prefix}_{start_time}_{end_time}.png"
    )

    if show_plot:
        plt.show()

    plt.close(fig)


def load_meg_data(hdf5_file_path) -> tuple[Dataset, Info]:
    with File(hdf5_file_path, "r") as f:
        raw_data: Dataset = f["data"][:]  # type: ignore
        times: Dataset = f["times"][:]  # type: ignore

    if len(times) >= 2:
        dt: np.float32 = times[1] - times[0]
        sfreq: np.float32 = 1.0 / dt  # type: ignore
    else:
        raise ValueError(
            "Not enough time points in 'times' to determine sampling frequency."
        )
    n_channels: int = raw_data.shape[0]

    # Create MNE Info object with default channel names.
    channel_names = [f"MEG {i + 1:03d}" for i in range(n_channels)]
    info = create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types=["mag"] * n_channels,  # type: ignore
    )
    return raw_data, info


def main():
    example_data = LibriBrainSpeech(
        data_path=str(DATA_DIR / "example_data"),
        include_run_keys=[("0", "1", "Sherlock1", "1")],  # type: ignore
        tmin=0.0,
        tmax=0.8,
        preload_files=True,
    )
    print(f"Loaded {len(example_data)} examples from LibriBrainSpeech dataset.")

    tsv_file_path = f"{DATA_DIR}/example_data/Sherlock1/derivatives/events/sub-0_ses-1_task-Sherlock1_run-1_events.tsv"

    hdf5_file_path = f"{DATA_DIR}/example_data/Sherlock1/derivatives/serialised/sub-0_ses-1_task-Sherlock1_run-1_proc-bads+headpos+sss+notch+bp+ds_meg.h5"

    meg_raw, info = load_meg_data(hdf5_file_path)
    tsv_data = pd.read_csv(tsv_file_path, sep="\t")

    meg_label_visualization(
        tsv_data,
        meg_raw,
        info,
        start_time=54,
        end_time=55,
        title="Pure silence",
        apply_sensor_mask=True,
    )
    meg_label_visualization(
        tsv_data,
        meg_raw,
        info,
        start_time=746,
        end_time=747,
        title="Pure silence",
        apply_sensor_mask=True,
    )

    meg_label_visualization(
        tsv_data,
        meg_raw,
        info,
        start_time=143,
        end_time=144,
        title="Pure Speech",
        apply_sensor_mask=True,
    )

    meg_label_visualization(
        tsv_data,
        meg_raw,
        info,
        start_time=240,
        end_time=241,
        title="Pure Speech",
        apply_sensor_mask=True,
    )


if __name__ == "__main__":
    main()
