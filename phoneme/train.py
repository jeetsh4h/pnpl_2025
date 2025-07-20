import torch
import lightning as L
import datetime as dt
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping

from torch.utils.data import DataLoader

from models import SpeechClassifier
from config import TRAINING_ARTIFACTS_DIR, SENSORS_SPEECH_MASK


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
):
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(42)

    LOG_DIR = TRAINING_ARTIFACTS_DIR / "logs"
    CHECKPOINT_PATH = TRAINING_ARTIFACTS_DIR / "checkpoints" / "notebook_model.ckpt"

    logger = CSVLogger(
        save_dir=LOG_DIR,
        name="notebook_model",
        version=f"{dt.datetime.now().strftime('%d-%m_%H-%M')}",
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

    trainer = L.Trainer(
        max_epochs=25,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[early_stopping_callback],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(CHECKPOINT_PATH)
    trainer.test(model, test_loader)
