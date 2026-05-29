from pathlib import Path
import yaml

class Config:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    @property
    def paths(self):
        return self.cfg["paths"]

    @property
    def data(self):
        return self.cfg["data"]

    def ensure_dirs(self):
        Path(self.paths["output_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.paths["plots_dir"]).mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)