import toml
import os

# Define the path to the folder containing the observed data
synthetic_folder = "src/data/algae-rotifers/synthetic"

# Define the path to the output directory
experiment_directory = os.getcwd() + "/src/experiments/julian/algae-rotifers/run-4/runs_transfer_learning"

# Loop through all files in the observed folder
for filename in os.listdir(synthetic_folder):
    if filename.startswith("TS1"):
        seed = int(filename.split('_S')[1])
        # Create a new TOML file with the modified observed path
        toml_data = {
            "plugins": [
                "src.models.deep+wide_freezing_cnn",
                "src.models.deep+wide_freezing_dnn",
                "src.models.deep+wide_freezing_lstm",
                "src.models.deep+wide_freezing_gru",
                "src.metrics.naive_forecasting_error",
            ],
            "metrics": [
                "mean_absolute_scaled_error",
                "prediction_trend_accuracy",
                "mean_directional_accuracy",
                "mean_absolute_error",
                "mean_absolute_scaled_error",
                "mean_squared_error",
                "root_mean_squared_error",
                "normalized_root_mean_squared_error",
                "mean_absolute_percentage_error",
                "naive_forecasting_error",
            ],
            "models": [
                {
                    "id": "PytorchLightningTransferLearningConvolutionalNeuralNetwork",
                    "seed": seed,
                    "training_params": {
                        "epochs": 30,
                        "finetuning_epochs": 5,
                        "learning_rate": 0.001,
                        "finetuning_learning_rate": 0.0001,
                        "batch_size": 64,
                        "accelerator": "auto",
                        "show_progress_bar": False,
                        "patience": 5,
                    }
                },
                {
                    "id": "PytorchLightningTransferLearningCustomDenseNeuralNetwork",
                    "seed": seed,
                    "training_params": {
                        "epochs": 30,
                        "finetuning_epochs": 5,
                        "learning_rate": 0.001,
                        "finetuning_learning_rate": 0.0001,
                        "batch_size": 64,
                        "accelerator": "auto",
                        "show_progress_bar": False,
                        "patience": 5,
                    }
                },
                {
                    "id": "PytorchLightningTransferLearningLSTMNeuralNetwork",
                    "seed": seed,
                    "training_params": {
                        "epochs": 30,
                        "finetuning_epochs": 5,
                        "learning_rate": 0.001,
                        "finetuning_learning_rate": 0.0001,
                        "batch_size": 64,
                        "accelerator": "auto",
                        "show_progress_bar": False,
                        "patience": 5,
                    }
                },
                {
                    "id": "PytorchLightningTransferLearningGRUNeuralNetwork",
                    "seed": seed,
                    "training_params": {
                        "epochs": 30,
                        "finetuning_epochs": 5,
                        "learning_rate": 0.001,
                        "finetuning_learning_rate": 0.0001,
                        "batch_size": 64,
                        "accelerator": "auto",
                        "show_progress_bar": False,
                        "patience": 5,
                    }
                }
            ],
            "data": {
                    "synthetic": os.path.join("/", synthetic_folder, filename),
                    "observed": "/src/data/algae-rotifers/real-world/algae-rotifers-coherent",
                    "test_split": 0.78,
                    "split_axis": "vertical",
                    "export_path": os.path.join("/hpi/fs00/scratch/julian.zabbarov/thesis/experiments/rotifers-algae/run-4", filename),
                    "time_series": {
                        "input_features": ["algae", "rotifers"],
                        "output_features": ["rotifers"],
                        "input_length": 3,
                        "output_length": 3,
                    }
            },
            
        }
        # Write the TOML data to a file with the same name as the observed data file
        toml_filename = os.path.splitext(filename)[0]
        output_folder = experiment_directory + "/" + toml_filename
        output_path = output_folder + "/" + "transfer.toml"
        if not os.path.exists(output_folder):
            # Create a new directory because it does not exist
            os.makedirs(output_folder)
        with open(output_path, "w") as f:
            toml.dump(toml_data, f)
