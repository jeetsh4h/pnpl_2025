import pandas as pd
from pnpl.datasets import LibriBrainSpeech

from phoneme.config import DATA_DIR
from .utils import load_meg_data, meg_label_visualization


def visualize_example_data():
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
        mask="example",
    )
    meg_label_visualization(
        tsv_data,
        meg_raw,
        info,
        start_time=746,
        end_time=747,
        title="Pure silence",
        apply_sensor_mask=True,
        mask="example",
    )

    meg_label_visualization(
        tsv_data,
        meg_raw,
        info,
        start_time=143,
        end_time=144,
        title="Pure Speech",
        apply_sensor_mask=True,
        mask="example",
    )

    meg_label_visualization(
        tsv_data,
        meg_raw,
        info,
        start_time=240,
        end_time=241,
        title="Pure Speech",
        apply_sensor_mask=True,
        mask="example",
    )
