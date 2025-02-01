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
    try:
        return subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Command '{' '.join(cmd)}' failed")
        print(f"Exit code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


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


def git_commit_with_retry(files: list[Path], version: str) -> None:
    add_cmd = ["git", "add"] + [str(f) for f in files]
    commit_cmd = ["git", "commit", "-m", f"Bump version to {version}"]

    print(f"Executing: {' '.join(add_cmd)}")
    run_command(add_cmd)

    try:
        print(f"Executing: {' '.join(commit_cmd)}")
        run_command(commit_cmd)
    except subprocess.CalledProcessError as e:
        if "uv-lock" in e.stderr:
            print("uv.lock was modified, adding it and retrying commit")
            uv_lock = Path("uv.lock")
            if uv_lock.exists():
                print("Executing: git add uv.lock")
                run_command(["git", "add", "uv.lock"])
                print(f"Executing: {' '.join(commit_cmd)}")
                run_command(commit_cmd)
            else:
                raise


def git_commands(version: str, pyproject_path: Path, init_path: Path) -> None:
    git_commit_with_retry(
        files=[pyproject_path, init_path],
        version=version,
    )

    commands = [
        ["git", "tag", version],
        ["git", "push"],
        ["git", "push", "origin", "--tags"],
    ]

    for cmd in commands:
        print(f"Executing: {' '.join(cmd)}")
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
        if not args.pyproject_path.exists():
            raise FileNotFoundError(
                f"pyproject.toml not found at {args.pyproject_path}"
            )
        if not args.init_path.exists():
            raise FileNotFoundError(f"__init__.py not found at {args.init_path}")

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
        print(f"Error output: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
