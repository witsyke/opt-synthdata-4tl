import toml
import os

# Define the path to the output directory
experiment_directory = os.getcwd() + "src/experiments/julian/algae-rotifers/run-4/runs_baseline"

# Loop through all files in the observed folder
for seed in [10, 17, 42, 93, 97]:
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
                "id": "PytorchLightningLSTMNeuralNetwork",
                "seed": seed,
                "training_params": {
                    "epochs": 20,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "accelerator": "auto",
                    "show_progress_bar": False,
                    "patience": 5,
                }
            },
            {
                "id": "PytorchLightningGRUNeuralNetwork",
                "seed": seed,
                "training_params": {
                    "epochs": 20,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "accelerator": "auto",
                    "show_progress_bar": False,
                    "patience": 5,
                }
            },
            {
                "id": "PytorchLightningConvolutionalNeuralNetwork",
                "seed": seed,
                "training_params": {
                    "epochs": 20,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "accelerator": "auto",
                    "show_progress_bar": False,
                    "patience": 5,
                }
            },
            {
                "id": "PytorchLightningCustomDenseNeuralNetwork",
                "seed": seed,
                "training_params": {
                    "epochs": 20,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "accelerator": "auto",
                    "show_progress_bar": False,
                    "patience": 5,
                }
            }
        ],
        "data": {
                "synthetic": "/src/data/algae-rotifers/real-world/algae-rotifers-coherent",
                "test_split": 0.75,
                "split_axis": "vertical",
                "time_series": {
                    "input_features": ["algae", "rotifers"],
                    "output_features": ["rotifers"],
                    "input_length": 5,
                    "output_length": 5,
                }
        }, 
    }
    # Write the TOML data to a file with the same name as the observed data file
    toml_filename = "S" + str(seed)
    output_folder = experiment_directory + "/" + toml_filename
    output_path = output_folder + "/" + "baseline.toml"
    if not os.path.exists(output_folder):
        # Create a new directory because it does not exist
        os.makedirs(output_folder)
    with open(output_path, "w") as f:
        toml.dump(toml_data, f)
