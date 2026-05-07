from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Sequence


def find_project_dir() -> Path:
    cwd = Path.cwd().resolve()

    if (cwd / "data_loader.py").exists():
        return cwd

    if cwd.name == "notebooks" and (cwd.parent / "data_loader.py").exists():
        return cwd.parent

    for parent in [cwd, *cwd.parents]:
        candidate = parent / "salient-object-detection-cnn"
        if (candidate / "data_loader.py").exists():
            return candidate

    raise FileNotFoundError("Could not find the salient-object-detection-cnn project directory.")


PROJECT_DIR = find_project_dir()
PYTHON = sys.executable


def run_command(args: Sequence[object], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = [str(arg) for arg in args]
    print(f"Project directory: {PROJECT_DIR}")
    print("Running:", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    if check:
        completed.check_returncode()
    return completed


def run_command_live(args: Sequence[object], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = [str(arg) for arg in args]
    print(f"Project directory: {PROJECT_DIR}")
    print("Running:", " ".join(command))

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
    returncode = process.wait()
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)


def run_script(script_name: str, *args: object, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command([PYTHON, script_name, *args], check=check)


def run_script_live(script_name: str, *args: object, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command_live([PYTHON, script_name, *args], check=check)
