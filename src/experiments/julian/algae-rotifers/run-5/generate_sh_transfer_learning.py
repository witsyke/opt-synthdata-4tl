import os

# define the path to the folder containing the observed data
synthetic_folder = "src/data/algae-rotifers/synthetic"

# define the path to the output directory for the generated bash script
experiment_directory = os.getcwd() + "/src/experiments/julian/algae-rotifers/run-5/"

# name the bash script
bash_script_name = "run_multivariate.sh"


def generate_bash_script(folders):
    script_lines = []

    for folder in folders:
        if str(folder).startswith("TS") and not str(folder).startswith("TS10000"):
            output_path = os.path.join(
                "src/experiments/julian/algae-rotifers/run-5/runs_transfer_learning",
                folder,
                "transfer.csv",
            )
            config_path = os.path.join(
                "src/experiments/julian/algae-rotifers/run-5/runs_transfer_learning",
                folder,
                "transfer.toml",
            )
            script_lines.append(
                f"echo simba_ml start-prediction transfer_learning --output-path {output_path} --config-path {config_path}"
            )
            script_lines.append(
                f"simba_ml start-prediction transfer_learning --output-path {output_path} --config-path {config_path}"
            )

    bash_script = "#!/bin/bash\n\n"
    bash_script += "\n".join(script_lines)

    return bash_script


folders = os.listdir(synthetic_folder)
bash_script = generate_bash_script(folders)

with open(experiment_directory + bash_script_name, "w") as file:
    file.write(bash_script)
