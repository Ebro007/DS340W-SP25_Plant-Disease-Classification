import subprocess

# List your config files (ensure these files exist, e.g., config1.json, config2.json, etc.)
config_files = ["config-PV-v3small.json", "config-Merged-v3small.json", "config-PV-v2.json", "config-Merged-v3large.json", "config-Merged-mobilevit.json"]

for config_file in config_files:
    print(f"Running training with config: {config_file}")
    result = subprocess.run(["python", "train.py", config_file])
    if result.returncode != 0:
        print(f"Training failed for {config_file}. Stopping further runs.")
        break