#!/usr/bin/env python3
"""Bounded, restart-safe launcher for the audited per-candidate controller."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(__file__).resolve().parent
RUNNER = WORK / "experiment_runner.py"
STATE = WORK / "state.json"
LOCK = WORK / "batch.lock"
LOG = WORK / "logs" / "batch_runner.log"
TERMINAL = {"budget_exhausted", "candidate_pool_exhausted", "success_verified", "finalized"}

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def load_state() -> dict:
    if not STATE.exists():
        return {"status": "uninitialized", "effective_candidates": 0}
    return json.loads(STATE.read_text(encoding="utf-8"))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-effective", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.target_effective <= 600:
        parser.error("--target-effective must be in [1, 600]")
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"{now()} batch lock exists: {LOCK}", flush=True)
        return 2
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "started": now(), "target_effective": args.target_effective}))
            f.flush(); os.fsync(f.fileno())
        while True:
            state = load_state()
            status = state.get("status", "unknown")
            effective = int(state.get("effective_candidates", 0))
            if effective >= args.target_effective or status in TERMINAL or status.startswith("blocked"):
                print(f"{now()} stop: status={status} effective={effective}", flush=True)
                return 0
            if (WORK / "run.lock").exists():
                print(f"{now()} per-round lock exists; refusing concurrent launch", flush=True)
                return 3
            if status == "uninitialized":
                action = "--bootstrap"
            elif status == "target_met_pending_retests":
                action = "--finalize"
            else:
                action = "--run-batch"
            print(f"{now()} launch {action}: effective={effective} target={args.target_effective}", flush=True)
            proc = subprocess.run([sys.executable, str(RUNNER), action], cwd=WORK.parents[1], check=False)
            if proc.returncode != 0:
                print(f"{now()} controller failed rc={proc.returncode}; stopping for recovery", flush=True)
                return proc.returncode
            time.sleep(1)
    finally:
        if LOCK.exists():
            LOCK.unlink()

if __name__ == "__main__":
    raise SystemExit(main())
