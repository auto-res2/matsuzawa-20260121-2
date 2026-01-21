import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    python_exec = sys.executable
    module_path = "-m"
    module = "src.train"

    cmd = [python_exec, module_path, module, f"run={cfg.run}", f"results_dir={cfg.results_dir}", f"mode={cfg.mode}"]
    # Allow live streaming of subprocess output
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    if process.returncode != 0:
        sys.exit(process.returncode)

if __name__ == "__main__":
    main()
