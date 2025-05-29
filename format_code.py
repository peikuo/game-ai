#!/usr/bin/env python3
"""
Script to format all Python files in the project.
Uses isort for import sorting, black for code formatting, and autopep8 for additional PEP 8 compliance.
"""
import os
import subprocess
from pathlib import Path


def format_python_files(directory):
    """Format all Python files in the given directory and its subdirectories."""
    print(f"Formatting Python files in {directory}...")

    # Find all Python files
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    if not python_files:
        print("No Python files found.")
        return

    print(f"Found {len(python_files)} Python files.")

    # Format with isort
    print("\nRunning isort...")
    subprocess.run(["isort", "--profile", "black"] + python_files, check=False)

    # Format with black
    print("\nRunning black...")
    subprocess.run(["black", "--line-length", "88"] +
                   python_files, check=False)

    # Format with autopep8 for any remaining issues
    print("\nRunning autopep8...")
    subprocess.run(["autopep8", "--in-place", "--aggressive",
                    "--aggressive"] + python_files, check=False, )

    print("\nFormatting complete!")


if __name__ == "__main__":
    project_dir = Path(__file__).parent
    format_python_files(project_dir)
