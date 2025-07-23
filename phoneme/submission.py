import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pnpl.datasets import LibriBrainCompetitionHoldout

from pathlib import Path
from typing import Any, Literal
from lightning.pytorch import LightningModule

from models import SpeechClassifier
from utils import SimpleHoldoutWrapper, collate_fn_for_holdout
from config import DATA_DIR, TRAINING_ARTIFACTS_DIR, SENSORS_SPEECH_MASK, NUM_WORKERS


def create_submission_for_first_model():
    model_path = TRAINING_ARTIFACTS_DIR / "checkpoints" / "notebook_model.ckpt"
    submission_file = (
        TRAINING_ARTIFACTS_DIR
        / "logs"
        / "notebook_model"
        / "20-07_23-13"
        / "submission_v2.csv"
    )
    model_kwargs = {
        "input_dim": len(SENSORS_SPEECH_MASK),
        "model_dim": 1000,
        "learning_rate": 1e-4,
        "dropout_rate": 0.5,
        "lstm_layers": 4,
        "weight_decay": 0.01,
        "batch_norm": True,
        "bi_directional": True,
    }
    create_submission_file(
        model_path,
        submission_file,
        SpeechClassifier,
        model_kwargs,
    )


def create_submission_file(
    best_model_path: Path,
    submission_file: Path,
    model_aarch: type[LightningModule],
    model_kwargs: dict[str, Any] = {},
):
    print(f"Loading {model_aarch.__name__} model from {best_model_path}")
    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    trained_model = model_aarch.load_from_checkpoint(
        best_model_path,
        **model_kwargs,
    )
    trained_model.eval()
    trained_model.to(device)

    holdout_data = LibriBrainCompetitionHoldout(
        data_path=str(DATA_DIR),
        tmin=0.0,
        tmax=0.8,
        task="speech",
        download=True,
    )

    predictions = generate_predictions(trained_model, holdout_data, device)

    tensor_predictions = [torch.tensor(pred).unsqueeze(0) for pred in predictions]

    holdout_data.generate_submission_in_csv(tensor_predictions, str(submission_file))
    print(f"Submission file created at {submission_file}")


def generate_predictions(
    model: LightningModule,
    dataset: LibriBrainCompetitionHoldout,
    device: Literal["cuda", "cpu"] = "cuda",
    batch_size: int = 1024,
):
    """
    This is only for the submission prediction
    Sliding window prediction generation:
      - Creates 200-timepoint windows to predict the middle timepoint
      - First 99 timepoints (0-98): default to 1 (speech) - no previous context
      - Last 100 timepoints: default to 1 (speech) - no future context
      - Middle timepoints 99-560537: use model predictions
    """
    total_timepoints = len(dataset)
    print(f"Total timepoints in dataset: {total_timepoints:,}")

    # Calculate prediction windows
    window_size = 200
    half_window = window_size // 2  # 100

    # Timepoints we can predict (have full 200-point context)
    first_predictable = (
        half_window - 1
    )  # 99 (first model prediction is for timepoint 99)
    last_predictable = total_timepoints - half_window - 1  # 560537
    predictable_count = last_predictable - first_predictable + 1  # 560439

    print(
        f"\nSliding window analysis:\n"
        f"  Window size: {window_size} timepoints\n"
        f"  First {first_predictable + 1} timepoints (0-{first_predictable}): default to speech (no past context)\n"
        f"  Timepoints {first_predictable}-{last_predictable}: {predictable_count:,} model predictions\n"
        f"  Last {half_window} timepoints: default to speech (no future context)\n"
        f"  Total predictions needed: {total_timepoints:,}"
    )

    holdout_loader = DataLoader(
        SimpleHoldoutWrapper(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn_for_holdout,
    )

    # Initialize predictions array
    all_predictions: list[float] = [1.0] * total_timepoints

    # Ensuring this as a redundancy
    model.eval()

    print(f"\nGenerating model predictions for {predictable_count:,} timepoints...")
    with torch.no_grad():
        for i, (meg_batch, sample_types) in enumerate(
            tqdm(holdout_loader, desc="Model Predictions")
        ):
            # Indices for this batch
            start_idx = i * batch_size

            # If collate_fn returned None for the tensor batch, skip model prediction
            if meg_batch is None:
                continue

            meg_masked = meg_batch[:, SENSORS_SPEECH_MASK, :].to(device)

            # Get model predictions
            logits = model(meg_masked)
            probs = torch.sigmoid(logits).squeeze().cpu().tolist()
            if not isinstance(probs, list):
                probs = [probs]

            # Place predictions in the correct global position
            prob_iter = iter(probs)
            for j, sample_type in enumerate(sample_types):
                if sample_type == "full":
                    # The model predicts the center point of the window.
                    # The window starts at index `start_idx + j`.
                    # The center is at `start_idx + j + half_window - 1`.
                    prediction_idx = start_idx + j + first_predictable
                    if prediction_idx <= last_predictable:
                        try:
                            all_predictions[prediction_idx] = next(prob_iter)
                        except StopIteration:
                            # This can happen if the last batch has fewer 'full' items
                            # than the iterator produces.
                            break

    print(f"\nGenerated {len(all_predictions):,} predictions")

    # Summary statistics
    default_start_count = first_predictable
    default_end_count = half_window

    model_preds = [
        all_predictions[i]
        for i in range(first_predictable, last_predictable + 1)
        if abs(all_predictions[i] - 1.0) > 1e-6
    ]

    print("\nSummary:")
    print(f"  Default predictions (start): {default_start_count:,} (all 1.0)")
    if model_preds:
        avg_pred = sum(model_preds) / len(model_preds)
        print(f"  Model predictions: {len(model_preds):,} (avg: {avg_pred:.3f})")
    else:
        print("  Model predictions: 0")
    print(f"  Default predictions (end): {default_end_count:,} (all 1.0)")

    return all_predictions
