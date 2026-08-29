# ruff: noqa: T201
import subprocess

if __name__ == "__main__":
    print("Running code formatting tools...")

    print("Running ruff...")
    subprocess.run(["ruff", "check", ".", "--fix"], check=True)

    print("Running yapf...")
    result = subprocess.run(["yapf", "-i", "-r", "-m", "."], check=True)
