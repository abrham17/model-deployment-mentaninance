import json, os, shutil
from pathlib import Path

HERE = Path(__file__).parent
with open(HERE / "config.json") as f:
    cfg = json.load(f)
model_base = HERE / cfg["model_base_path"]
gate = float(cfg["gate"]["min_accuracy_percent"])
model_name = cfg["model_name"]

# Train a candidate
print("Training candidate...")
os.system(f"python {HERE / 'train_tf.py'} > {HERE / 'last_train_log.txt'}")

# Find versions
versions = sorted([int(p.name) for p in model_base.iterdir() if p.is_dir() and p.name.isdigit()])
if not versions:
    raise SystemExit("No versions found after training.")
candidate = versions[-1]

# Determine current (previous) by excluding candidate if possible
current = versions[-2] if len(versions) >= 2 else None

# Read candidate metrics
import json
cand_metrics_path = model_base / str(candidate) / "metrics.json"
with open(cand_metrics_path) as f:
    cand_metrics = json.load(f)
cand_acc = float(cand_metrics["test_accuracy_percent"])

def notify(msg):
    note = HERE / "notifications.log"
    with open(note, "a") as nf:
        nf.write(msg + "\n")
    print(msg)

if cand_acc >= gate:
    # Promote: TF Serving will automatically serve the highest version number.
    notify(f"[PROMOTE] version {candidate} meets gate ({cand_acc:.2f}% >= {gate:.2f}%).")
    # Optionally, we could clean up very old versions here.
else:
    # Rollback: remove candidate so serving stays on previous version
    notify(f"[REJECT] version {candidate} fails gate ({cand_acc:.2f}% < {gate:.2f}%). Keeping version {current}.")
    # Remove candidate directory
    shutil.rmtree(model_base / str(candidate), ignore_errors=True)
