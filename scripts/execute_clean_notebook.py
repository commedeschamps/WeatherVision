"""Execute notebooks/WeatherVision_Complete_A1_A2.ipynb and verify all cells run without error."""

import os
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def execute_notebook():
    nb_path = ROOT / "notebooks/WeatherVision_Complete_A1_A2.ipynb"
    print(f"Reading {nb_path}...")
    nb = nbformat.read(nb_path, as_version=4)

    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.km = km

    print("Executing notebook cells...")
    start_time = time.time()
    client.execute()
    elapsed = time.time() - start_time
    print(f"Execution finished in {elapsed:.2f} seconds.")

    # Validate no cells produced an error output
    error_count = 0
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    print(f"ERROR in cell {idx}: {out.get('ename')}: {out.get('evalue')}")
                    error_count += 1

    if error_count > 0:
        raise RuntimeError(f"Notebook execution encountered {error_count} error(s)!")

    # Save executed notebook with all outputs
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Executed notebook saved with all outputs at {nb_path}")

if __name__ == "__main__":
    execute_notebook()
