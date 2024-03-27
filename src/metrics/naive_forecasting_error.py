import numpy as np
import numpy.typing as npt

from simba_ml.prediction.time_series.metrics import factory
from simba_ml.prediction.time_series.metrics import metrics

def naive_forecasting_error(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> np.float64:
    """Calculates the mean directional error.

    This is equal to 1-mean directional accuracy.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The mean directional error.
    """
    return metrics.mean_absolute_error(y_true[:, 1:], y_true[:, :-1])

def register() -> None:
    """Registers the mean_directional_error in the metrics factory."""
    factory.register(
        "naive_forecasting_error",
        naive_forecasting_error
    )