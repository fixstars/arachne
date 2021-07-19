import os
import shutil
from pathlib import Path

from arachne.env import Env

_dir_count = 1


def make_artifact_dir(name: str, work_dir: str = None) -> Path:
    global _dir_count
    if work_dir is None:
        work_dir = Env.ARTIFACT_DIR
    else:
        work_dir = Path(work_dir)
    dir_path = work_dir / f"{_dir_count}_{name}"
    if dir_path.exists():
        if dir_path.is_dir():
            shutil.rmtree(dir_path)
        else:
            os.remove(dir_path)
    dir_path.mkdir()

    _dir_count += 1

    return dir_path
