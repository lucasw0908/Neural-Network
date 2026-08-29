# ruff: noqa: T201
import subprocess

if __name__ == "__main__":
    print("Running code linting tools...")

    subprocess.run(["ruff", "check", "."], check=True)
