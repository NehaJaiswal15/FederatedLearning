"""
Config Loader Utility
Reads YAML configuration files and returns them as Python dicts.
"""

import yaml
import os


def load_config(config_path="config/default.yaml"):
    """
    Load a YAML configuration file and return it as a dictionary.
    
    Args:
        config_path (str): Path to the YAML config file.
                           Defaults to 'config/default.yaml'.
    
    Returns:
        dict: Configuration parameters as a nested dictionary.
    
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file has syntax errors.
    """
    # Step 1: Make sure the file actually exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Step 2: Open the YAML file and parse it
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Step 3: Return the parsed dictionary
    return config


# ── Quick test: run this file directly to verify it works ──
if __name__ == "__main__":
    config = load_config()

    print("[OK] Config loaded successfully!\n")
    print("-- Full Config --")
    for section, values in config.items():
        print(f"\n[{section}]")
        for key, value in values.items():
            print(f"  {key}: {value}")
