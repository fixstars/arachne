import os
import shutil
from os import path
from pathlib import Path


class Env(object):
    PYTHON_DIR = Path(path.abspath(path.join(__file__, "..", "..")))
    ROOT_DIR = PYTHON_DIR.parent
    THIRDPARTY_DIR = ROOT_DIR / "3rdparty"
    ARTIFACT_DIR = ROOT_DIR / ".artifacts"
    DATASET_DIR = Path("/datasets")
    MODEL_DIR = ROOT_DIR / ".models"
    BUILD_CACHE_DIR = ROOT_DIR / ".cache"


assert Env.PYTHON_DIR.exists()
assert Env.ROOT_DIR.exists()
assert Env.THIRDPARTY_DIR.exists()
if Env.ARTIFACT_DIR.exists():
    if Env.ARTIFACT_DIR.is_dir():
        shutil.rmtree(Env.ARTIFACT_DIR)
    else:
        os.remove(Env.ARTIFACT_DIR)
Env.ARTIFACT_DIR.mkdir()
Env.DATASET_DIR.mkdir(exist_ok=True)
Env.MODEL_DIR.mkdir(exist_ok=True)
Env.BUILD_CACHE_DIR.mkdir(exist_ok=True)
