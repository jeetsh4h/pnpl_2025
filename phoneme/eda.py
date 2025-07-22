from torch.utils.data import DataLoader
from pnpl.datasets import LibriBrainSpeech

from torch import Tensor

from config import DATA_DIR, NUM_WORKERS, SENSORS_SPEECH_MASK


# no filter, because we want all the channels
val_data: DataLoader[LibriBrainSpeech] = DataLoader(
    # TODO: FilteredDataset is perforoming shuffling
    # maybe just perfrom the label picking as a pre-processing step
    # within the for loop itself
    LibriBrainSpeech(
        data_path=str(DATA_DIR),
        partition="validation",
        tmin=0.0,
        tmax=0.8,
        preload_files=True,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

# TODO: ask Kaushik sir about the data thing
#       I am thoroughly confused.
for data, labels in val_data:
    # data.shape = (1, 306, 200)
    # labels.shape = (1, 200)

    # only take the middle label (like in notebook)
    label: Tensor = labels[:, labels.shape[1] // 2]  # shape: (1,)

    # mask the data
    masked_data: Tensor = data[:, SENSORS_SPEECH_MASK, :]  # shape: (1, 23, 200)
