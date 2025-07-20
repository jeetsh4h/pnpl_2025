import pandas as pd
from torch.utils.data import DataLoader
from pnpl.datasets import LibriBrainSpeech

from train import train
from visualize import meg_label_visualization
from utils import FilteredDataset, load_meg_data
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS


def main():
    train_notebook_model()


def visualize_example_data():
    # this is just to ensure that the example data is downloaded
    example_data = LibriBrainSpeech(
        data_path=str(DATA_DIR / "example_data"),
        include_run_keys=[("0", "1", "Sherlock1", "1")],  # type: ignore
        tmin=0.0,
        tmax=0.8,
        preload_files=True,
    )
    print(f"Loaded {len(example_data)} examples from LibriBrainSpeech dataset.")

    EXAMPLE_DATA_DIR = DATA_DIR / "example_data" / "Sherlock1" / "derivatives"
    tsv_file_path = (
        f"{EXAMPLE_DATA_DIR}/events/sub-0_ses-1_task-Sherlock1_run-1_events.tsv"
    )
    hdf5_file_path = f"{EXAMPLE_DATA_DIR}/serialised/sub-0_ses-1_task-Sherlock1_run-1_proc-bads+headpos+sss+notch+bp+ds_meg.h5"

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


def train_notebook_model():
    # this is the filtered dataset
    # with the sensor mask applied
    train_data = LibriBrainSpeech(
        data_path=str(DATA_DIR),
        partition="train",
        tmin=0.0,
        tmax=0.8,
        preload_files=True,
    )
    train_loader: DataLoader[FilteredDataset] = DataLoader(
        FilteredDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    print(f"Training data: {len(train_data)} samples")

    val_data = LibriBrainSpeech(
        data_path=str(DATA_DIR),
        partition="validation",
        tmin=0.0,
        tmax=0.8,
        preload_files=True,
    )
    val_loader: DataLoader[FilteredDataset] = DataLoader(
        FilteredDataset(val_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    print(f"Validation data: {len(val_data)} samples")

    test_data = LibriBrainSpeech(
        data_path=str(DATA_DIR),
        partition="test",
        tmin=0.0,
        tmax=0.8,
        preload_files=True,
    )
    test_loader: DataLoader[FilteredDataset] = DataLoader(
        FilteredDataset(test_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    print(f"Test data: {len(test_data)} samples")

    train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    main()
