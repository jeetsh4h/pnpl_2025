import random
import numpy as np
from mne import create_info
from pnpl.datasets import LibriBrainSpeech

from h5py import File
from typing import Literal
from mne._fiff.meas_info import Info
from h5py import Dataset as h_Dataset
from torch.utils.data import Dataset as t_Dataset

from config import SENSORS_SPEECH_MASK


def load_meg_data(hdf5_file_path) -> tuple[h_Dataset, Info]:
    with File(hdf5_file_path, "r") as f:
        raw_data: h_Dataset = f["data"][:]  # type: ignore
        times: h_Dataset = f["times"][:]  # type: ignore

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


class FilteredDataset(t_Dataset):
    """
    Parameters:
        dataset: LibriBrain dataset.
        limit_samples (int, optional): If provided, limits the length of the dataset to this
                          number of samples.
        speech_silence_only (bool, optional): If True, only includes segments that are either
                          purely speech or purely silence (with additional balancing).
        apply_sensors_speech_mask (bool, optional): If True, applies a fixed sensor mask to the sensor
                          data in each sample.
    """

    def __init__(
        self,
        dataset: LibriBrainSpeech,
        limit_samples: int | None = None,
        apply_sensors_speech_mask: bool = True,
        sensors_speech_mask: Literal["notebook"] | list[int] = "notebook",
    ):
        self.dataset = dataset
        self.limit_samples = limit_samples
        self.apply_sensors_speech_mask = apply_sensors_speech_mask

        # These are the sensors we identified:
        if sensors_speech_mask == "notebook":
            self.sensors_speech_mask = SENSORS_SPEECH_MASK
        else:
            self.sensors_speech_mask: list[int] = sensors_speech_mask

        self.balanced_indices = list(range(len(dataset.samples)))
        # Shuffle the indices
        self.balanced_indices = random.sample(
            self.balanced_indices, len(self.balanced_indices)
        )

    def __len__(self):
        """Returns the number of samples in the filtered dataset."""
        if self.limit_samples is not None:
            return self.limit_samples
        return len(self.balanced_indices)

    def __getitem__(self, index):
        # Map index to the original dataset using balanced indices
        original_idx = self.balanced_indices[index]
        if self.apply_sensors_speech_mask:
            sensors = self.dataset[original_idx][0][self.sensors_speech_mask]
        else:
            sensors = self.dataset[original_idx][0][:]
        label_from_the_middle_idx = self.dataset[original_idx][1].shape[0] // 2
        return [sensors, self.dataset[original_idx][1][label_from_the_middle_idx]]
