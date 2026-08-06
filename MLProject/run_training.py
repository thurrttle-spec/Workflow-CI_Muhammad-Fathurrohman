"""
run_training.py - Script wrapper untuk mlflow run
Dipanggil oleh GitHub Actions CI
"""
import subprocess
import sys

cmd = [
    "mlflow", "run", "MLProject/",
    "--env-manager", "local",
    "-P", "n_estimators=200",
    "-P", "max_depth=None",
    "-P", "min_samples_split=5",
    "-P", "max_features=sqrt",
    "-P", "test_size=0.2",
    "-P", "random_state=42",
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd)
sys.exit(result.returncode)
