"""
run_one_experiment.py
=====================

Runs ONE training script under controlled, isolated conditions and writes a
JSON file with the captured metrics, stdout tail and resource usage.

For each invocation, this script:
    1. Sets EXPERIMENT_SEED in the environment so the LLM-generated script
       uses the requested random seed (the LLM scripts read this variable;
       see the protocol in prompts/).
    2. Copies the LLM-generated script to an isolated working directory.
    3. Symlinks the data splits into the working directory.
    4. Launches the script as a subprocess with a configurable wall-clock
       timeout.
    5. Monitors CPU and memory of the subprocess in a background thread.
    6. Parses test metrics out of the subprocess stdout.
    7. Emits a JSON record with everything needed for downstream analysis.

Usage:
    python3 run_one_experiment.py \
        --script scripts/task_a_binary/tier1/chatgpt.py \
        --task task_a_binary \
        --seed 0 \
        --out logs/task_a_binary_tier1_chatgpt_seed00.json
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil


WRAPPER_VERSION = "v3_improved_parser_persistent_stdout"
DEFAULT_TIMEOUT_MIN = 240   # 4 h ceiling per run


# ---------------------------------------------------------------------------
# Resource monitor (runs in a background thread)
# ---------------------------------------------------------------------------
class ResourceMonitor(threading.Thread):
    """Samples CPU and memory of the target subprocess at fixed intervals."""

    def __init__(self, pid: int, interval: float = 1.0):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.samples = []  # list of (cpu_percent, mem_percent)
        # Use a renamed event to avoid clashing with Thread._stop (Python >= 3.12)
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        proc.cpu_percent(interval=None)  # prime
        while not self._stop_event.is_set():
            try:
                cpu = proc.cpu_percent(interval=None)
                mem = proc.memory_percent()
                self.samples.append((cpu, mem))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()

    def summary(self) -> dict:
        if not self.samples:
            return {"cpu_mean": None, "cpu_max": None,
                    "mem_mean": None, "mem_max": None, "n_samples": 0}
        cpus = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        return {
            "cpu_mean": sum(cpus) / len(cpus),
            "cpu_max": max(cpus),
            "mem_mean": sum(mems) / len(mems),
            "mem_max": max(mems),
            "n_samples": len(self.samples),
        }


# ---------------------------------------------------------------------------
# stdout parser (best-effort, robust to small variations in LLM output)
# ---------------------------------------------------------------------------
METRIC_PATTERNS = {
    "accuracy":          [r"test\s+accuracy[:=]\s*([0-9.]+)",
                          r"\bacc(?:uracy)?[:=]\s*([0-9.]+)"],
    "f1":                [r"macro[\s_-]*f1[:=]\s*([0-9.]+)",
                          r"\bf1[\s_-]*score[:=]\s*([0-9.]+)",
                          r"\bf1[:=]\s*([0-9.]+)"],
    "balanced_accuracy": [r"balanced[\s_-]*acc(?:uracy)?[:=]\s*([0-9.]+)"],
    "auc":               [r"\bauc[:=]\s*([0-9.]+)",
                          r"roc[\s_-]*auc[:=]\s*([0-9.]+)"],
    "loss":              [r"\btest[\s_-]*loss[:=]\s*([0-9.]+)"],
    "recall":            [r"\brecall[:=]\s*([0-9.]+)"],
    "precision":         [r"\bprecision[:=]\s*([0-9.]+)"],
}

# Per-class recall (Task B): captured separately
PER_CLASS_PATTERNS = {
    "healthy":         r"healthy[^\n]*?recall[:=\s]+([0-9.]+)",
    "red_spider_mite": r"red[_\s-]*spider[_\s-]*mite[^\n]*?recall[:=\s]+([0-9.]+)",
    "coffee_leaf_rust": r"(?:coffee[_\s-]*leaf[_\s-]*)?rust[^\n]*?recall[:=\s]+([0-9.]+)",
    "unhealthy":       r"unhealthy[^\n]*?recall[:=\s]+([0-9.]+)",
}


def parse_metrics(stdout: str) -> dict:
    """Return a dict with the metrics found in stdout (lowercased)."""
    text = stdout.lower()
    metrics: dict = {}
    for name, patterns in METRIC_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                try:
                    metrics[name] = float(m.group(1))
                    break
                except ValueError:
                    continue

    # Per-class recall, precision, f1
    for cls, pat in PER_CLASS_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            try:
                metrics[f"recall_{cls}"] = float(m.group(1))
            except ValueError:
                pass

    return metrics


# ---------------------------------------------------------------------------
# Workdir setup
# ---------------------------------------------------------------------------
def prepare_workdir(script_path: Path, task: str) -> Path:
    """Create a temp workdir with the script and a symlink to data/splits/<task>."""
    workdir = Path(tempfile.mkdtemp(prefix=f"exp_{os.getpid():02d}_"))
    shutil.copy2(script_path, workdir / "script.py")
    data_src = Path("data/splits") / task
    if not data_src.is_dir():
        raise FileNotFoundError(f"Data splits not found: {data_src}. "
                                f"Run data_preparation/prepare_rocole.py first.")
    (workdir / "data").mkdir()
    os.symlink(data_src.resolve(), workdir / "data" / "splits" /
               task if False else workdir / "data_link")
    # Simpler: put the splits at the canonical relative path each script uses
    (workdir / "data" / "splits").mkdir(exist_ok=True)
    os.symlink(data_src.resolve(), workdir / "data" / "splits" / task)
    return workdir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True,
                        help="Path to the LLM-generated training script.")
    parser.add_argument("--task", required=True,
                        choices=["task_a_binary", "task_b_3class"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSON path.")
    parser.add_argument("--timeout-min", type=int, default=DEFAULT_TIMEOUT_MIN)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Reset stdout for this run
    print(f"[RUN] {args.script.parent.name}/{args.script.parent.parent.name}/"
          f"{args.script.name} seed={args.seed} "
          f"(timeout={args.timeout_min}min)")

    # Prepare an isolated workdir
    try:
        workdir = prepare_workdir(args.script, args.task)
    except Exception as e:
        record = {
            "wrapper_version": WRAPPER_VERSION,
            "script": str(args.script), "task": args.task, "seed": args.seed,
            "error": f"workdir setup failed: {e}",
            "syntactic_ok": False,
            "metrics": {},
        }
        args.out.write_text(json.dumps(record, indent=2))
        print(f"  [FAIL] setup: {e}")
        sys.exit(1)

    env = os.environ.copy()
    env["EXPERIMENT_SEED"] = str(args.seed)
    # The LLM-generated script reads splits at the relative path
    # data/splits/{task}; the workdir is already laid out accordingly.
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, "-u", "script.py"]
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=workdir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    monitor = ResourceMonitor(proc.pid)
    monitor.start()

    stdout_chunks = []
    timed_out = False
    try:
        # Read line by line so we can enforce timeout cleanly
        end = t0 + args.timeout_min * 60
        for line in proc.stdout:  # type: ignore[union-attr]
            stdout_chunks.append(line)
            if time.time() > end:
                proc.kill()
                timed_out = True
                break
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        timed_out = True
    finally:
        monitor.stop()
        monitor.join(timeout=2)

    elapsed = time.time() - t0
    stdout = "".join(stdout_chunks)
    metrics = parse_metrics(stdout)
    syntactic_ok = (proc.returncode == 0) and (metrics.get("accuracy") is not None)

    record = {
        "wrapper_version": WRAPPER_VERSION,
        "command": cmd,
        "script": str(args.script),
        "task": args.task,
        "seed": args.seed,
        "workdir": str(workdir),
        "return_code": proc.returncode,
        "wallclock_seconds": elapsed,
        "timed_out": timed_out,
        "syntactic_ok": bool(syntactic_ok),
        "metrics": metrics,
        "resources": monitor.summary(),
        "stdout_tail": stdout[-8000:],   # last ~8 KB
    }
    args.out.write_text(json.dumps(record, indent=2))

    shutil.rmtree(workdir, ignore_errors=True)

    if syntactic_ok:
        m = metrics
        cpu = monitor.summary()["cpu_mean"] or 0
        line = (f"  [OK]  acc={m.get('accuracy', float('nan')):.4f}  "
                f"f1={m.get('f1', float('nan')):.4f}  "
                f"auc={m.get('auc', float('nan')):.3f}  "
                f"time={elapsed/60:.1f}min  cpu={cpu:.0f}%")
        print(line)
        sys.exit(0)
    else:
        print(f"  [FAIL] returncode={proc.returncode}  time={elapsed/60:.1f}min")
        sys.exit(1)


if __name__ == "__main__":
    main()
