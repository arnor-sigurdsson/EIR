import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tomlkit


@dataclass
class VersionInfo:
    version: str
    bump_type: Literal["patch", "minor", "major"]


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def parse_version(version: str) -> tuple[int, int, int]:
    major, minor, patch = map(int, version.split("."))
    return major, minor, patch


def bump_version(
    current_version: str, bump_type: Literal["patch", "minor", "major"]
) -> str:
    major, minor, patch = parse_version(current_version)

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1

    return f"{major}.{minor}.{patch}"


def get_new_version(
    bump_type: Literal["patch", "minor", "major"], pyproject_path: Path
) -> VersionInfo:
    with open(pyproject_path, encoding="utf-8") as f:
        pyproject_data = tomlkit.load(f)

    current_version = str(pyproject_data["project"]["version"])
    new_version = bump_version(current_version, bump_type)

    pyproject_data["project"]["version"] = new_version

    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(pyproject_data))

    return VersionInfo(version=new_version, bump_type=bump_type)


def update_init_file(version: str, init_path: Path) -> None:
    content = init_path.read_text()
    new_content = re.sub(
        r'__version__\s*=\s*"[^"]*"',
        f'__version__ = "{version}"',
        content,
    )
    init_path.write_text(new_content)


def git_commands(version: str, pyproject_path: Path, init_path: Path) -> None:
    commands = [
        ["git", "add", str(pyproject_path), str(init_path)],
        ["git", "commit", "-m", f"Bump version to {version}"],
        ["git", "tag", version],
        ["git", "push"],
        ["git", "push", "origin", "--tags"],
    ]

    for cmd in commands:
        run_command(cmd=cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automate release workflow")
    parser.add_argument(
        "bump_type",
        choices=["patch", "minor", "major"],
        help="Version bump type",
    )
    parser.add_argument(
        "--init-path",
        type=Path,
        default=Path("src/eir/__init__.py"),
        help="Path to __init__.py file",
    )
    parser.add_argument(
        "--pyproject-path",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml file",
    )

    args = parser.parse_args()

    try:
        version_info = get_new_version(
            bump_type=args.bump_type,
            pyproject_path=args.pyproject_path,
        )
        print(f"Bumping version to: {version_info.version}")

        update_init_file(version=version_info.version, init_path=args.init_path)
        print(f"Updated version in {args.init_path}")

        git_commands(
            version=version_info.version,
            pyproject_path=args.pyproject_path,
            init_path=args.init_path,
        )
        print("Successfully completed all git commands")

        print("\nRelease workflow completed successfully!")
        print("Don't forget to update Github release notes!")

    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd}")
        print(f"Output: {e.output}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
