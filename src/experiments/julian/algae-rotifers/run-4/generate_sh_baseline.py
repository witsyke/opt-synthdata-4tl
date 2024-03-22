import os

# define the path to the output directory for the generated bash script
experiment_directory = os.getcwd() + "/src/experiments/julian/algae-rotifers/run-4/"

# name the bash script
bash_script_name = "run_baseline.sh"


def generate_bash_script(folders):
    script_lines = []

    for folder in folders:
        output_path = os.path.join(
            "src/experiments/julian/algae-rotifers/run-4/runs_baseline", folder, "baseline.csv"
        )
        config_path = os.path.join(
            "src/experiments/julian/algae-rotifers/run-4/runs_baseline",
            folder,
            "baseline.toml",
        )
        script_lines.append(
            f"simba_ml start-prediction synthetic_data --output-path {output_path} --config-path {config_path}"
        )

    bash_script = "#!/bin/bash\n\n"
    bash_script += "\n".join(script_lines)

    return bash_script


folders = os.listdir(os.path.join(experiment_directory, "runs_baseline"))
bash_script = generate_bash_script(folders)

with open(experiment_directory + bash_script_name, "w") as file:
    file.write(bash_script)
