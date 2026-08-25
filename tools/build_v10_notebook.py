"""Generate the self-contained Kaggle v10 notebook from train_v10_gpu.py."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "train_v10_gpu.py"
OUTPUT = ROOT / "kaggle" / "v10_exact_gpu_fusion.ipynb"


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    marker = '\nif __name__ == "__main__":\n'
    if marker not in source:
        raise RuntimeError("Could not locate train_v10_gpu.py main guard")
    embedded = source.split(marker, 1)[0]
    embedded += (
        "\n\n# Kaggle entrypoint: auto-discovers the competition and attached pool cache.\n"
        "run(parse_args([\n"
        "    '--output-dir', '/kaggle/working',\n"
        "    '--max-total-iter', '4000',\n"
        "    '--block-iter', '250',\n"
        "    '--chunk-rows', '131072',\n"
        "]))\n"
    )
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Smartphone Addiction v10 - Exact GPU Fusion\n",
                    "\n",
                    "Attach the `playground-series-s6e8` competition and a private dataset "
                    "containing `public_pool_oof.npy`, `public_pool_test.npy`, and "
                    "`public_pool_manifest.json`. Enable a GPU, then run all.\n",
                    "\n",
                    "The notebook writes `submission_v10.csv` only after both float64 "
                    "LBFGS fits report convergence. Checkpoints and diagnostics are saved "
                    "under `/kaggle/working`.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": embedded.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"isGpuEnabled": True, "accelerator": "nvidiaTeslaT4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
