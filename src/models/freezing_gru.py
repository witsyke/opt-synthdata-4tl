"""Uses a Dense Neural Network to predict the next timestamps."""

import typing
import dataclasses

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from simba_ml.prediction.time_series.models import factory, transfer_learning_factory

from simba_ml.prediction.time_series.config import (
    time_series_config,
)
from simba_ml.prediction.time_series.models import model_to_transfer_learning_model
from models import pytorch_lightning_wandb

@dataclasses.dataclass
class GRUNeuralNetworkConfig(pytorch_lightning_wandb.PytorchLightningWandbModelConfig):
    """Defines the configuration for the GRUNeuralNetwork."""

    name: str = "PyTorch_Lightning_GRU_Neural_Network"
    batch_size: int = 32


class GRUNeuralNetwork(pytorch_lightning_wandb.PytorchLightningWandbModel):
    """Defines a model, which uses a dense neural network for prediction."""

    def get_model(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: pytorch_lightning_wandb.PytorchLightningWandbModelConfig,
    ) -> pl.LightningModule:
        """Returns the model.

        Args:
            time_series_params: parameters of time series that affects
                training and architecture of the model
            model_params: configuration for the model.

        Returns:
            The model.
        """
        return _GRUNeuralNetwork(time_series_params, model_params)


class _GRUNeuralNetwork(pl.LightningModule):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: pytorch_lightning_wandb.PytorchLightningWandbModelConfig,
    ) -> None:
        super().__init__()
        self.time_series_params = time_series_params
        self.model_params = model_params
        self.gru = nn.GRU(input_size=len(time_series_params.input_features),
                           hidden_size=64,
                           batch_first=True)
        self.fc1 = nn.Linear(
            in_features=64,
            out_features=64,
        )
        self.fc2 = nn.Linear(
            in_features=64,
            out_features=time_series_params.output_length
            * len(time_series_params.output_features),
        )

    def forward(  # pylint: disable=arguments-differ
        self, x: torch.Tensor
    ) -> torch.Tensor:
        _, h = self.gru(x)
        out = nn.ReLU()(h)
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out = self.fc2(out)
        return out.view(
            (
                x.shape[0],
                self.time_series_params.output_length,
                len(self.time_series_params.output_features),
            )
        )

    def configure_optimizers(self) -> torch.optim.Adam:
        if self.model_params.finetuning:
            if self.model_params.training_params.finetuning_learning_rate is None:
                raise ValueError("finetuning_learning_rate must be set.")
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.model_params.training_params.finetuning_learning_rate,
            )
            for name, param in self.named_parameters():
                if "gru" in name or "fc1" in name:
                    param.requires_grad = False
            return optimizer
        return torch.optim.Adam(
            self.parameters(),
            lr=self.model_params.training_params.learning_rate,
        )

    def training_step(  # pylint: disable=arguments-differ
        self,
        train_batch: typing.List[typing.Tuple[typing.List[float], typing.List[int]]],
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        x, y = train_batch
        out = self(x)
        loss = F.mse_loss(y, out)
        self.log("train/loss", loss)
        return loss  # type: ignore[arg-type]
    
    def validation_step(  # pylint: disable=arguments-differ
        self,
        val_batch: typing.List[typing.Tuple[typing.List[float], typing.List[int]]],
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        x, y = val_batch
        out = self(x)
        loss = F.mse_loss(y, out)
        self.log("val/loss", loss)
        return loss  # type: ignore[arg-type]

def register():
    factory.register(
        "PytorchLightningGRUNeuralNetwork",
        GRUNeuralNetworkConfig,
        GRUNeuralNetwork,
    )
    transfer_learning_factory.register(
        "PytorchLightningTransferLearningGRUNeuralNetwork",
        GRUNeuralNetworkConfig,
        model_to_transfer_learning_model.model_to_transfer_learning_model_with_pretraining(
            GRUNeuralNetwork
        ),
    )
