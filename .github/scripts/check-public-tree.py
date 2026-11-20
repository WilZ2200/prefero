"""Reject private research and local runtime artefacts in the public Git tree."""

import re
import subprocess
from pathlib import PurePosixPath

PRIVATE_DIRS = {
    ".aws", ".claude", ".venv", "credentials", "manuscript", "manuscripts",
    "outputs", "paper", "papers", "thesis", "theses",
}
PRIVATE_FILES = {
    "credentials.json", "credentials.toml", "id_ed25519", "id_rsa",
    "roadmap.md", "secrets.toml", "service-account.json",
}
PRIVATE_DATA_DIRS = {"data/generated/", "data/processed/", "data/raw/"}
LOCAL_FILE = re.compile(r"\.(?:db|sqlite3?)(?:-(?:wal|shm|journal))?$|\.(?:p12|pfx|key)$")
RESEARCH_NAME = re.compile(r"^(?:data[ _-]*analysis[ _-]*plan|manuscript|thesis)(?:[ ._-]|$)")


def is_private(path: str) -> bool:
    relative = path.casefold()
    parts = PurePosixPath(relative).parts
    filename = parts[-1]
    if any(relative.startswith(folder) for folder in PRIVATE_DATA_DIRS):
        return relative not in {folder + ".gitkeep" for folder in PRIVATE_DATA_DIRS}
    if filename in {".env.example", ".env.sample", ".env.template"}:
        return bool(PRIVATE_DIRS.intersection(parts[:-1]))
    return (
        bool(PRIVATE_DIRS.intersection(parts[:-1]))
        or filename in PRIVATE_FILES
        or filename == ".env"
        or filename.startswith(".env.")
        or bool(LOCAL_FILE.search(filename))
        or bool(RESEARCH_NAME.match(filename))
    )


if __name__ == "__main__":
    tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8", "surrogateescape")
    forbidden = sorted(path for path in tracked.split("\0") if path and is_private(path))
    if forbidden:
        print("Remove these private or local-only paths from the public Git tree:")
        for path in forbidden:
            print(f"  {path!r}")
        raise SystemExit(1)
