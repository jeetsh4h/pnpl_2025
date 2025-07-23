import torch
import lightning as L
import datetime as dt
from torch.utils.data import DataLoader
from pnpl.datasets import LibriBrainSpeech
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


from utils import FilteredDataset
from models import SpeechClassifier
from config import (
    DATA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SENSORS_SPEECH_MASK,
    TRAINING_ARTIFACTS_DIR,
)


# TODO: create submission file right after training
def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
):
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(42)

    CURR_DATE = dt.datetime.now().strftime("%d-%m_%H-%M")
    LOG_DIR = TRAINING_ARTIFACTS_DIR / "logs"
    CHECKPOINT_PATH = TRAINING_ARTIFACTS_DIR / "checkpoints" / CURR_DATE

    logger = CSVLogger(
        save_dir=LOG_DIR,
        name="notebook_model",
        version=CURR_DATE,
    )

    model = SpeechClassifier(
        input_dim=len(SENSORS_SPEECH_MASK),
        model_dim=1000,
        learning_rate=1e-4,
        dropout_rate=0.5,
        lstm_layers=4,
        weight_decay=0.01,
        batch_norm=True,
        bi_directional=True,
    )

    logger.log_hyperparams((model.hparams))  # type: ignore

    # tune this later
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10, verbose=True, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        filename="{epoch}-{val_loss:.2f}_notebook_model",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=25,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[early_stopping_callback, checkpoint_callback],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


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
