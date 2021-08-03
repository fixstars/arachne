import os
import shutil
from os import path
from pathlib import Path


class Env(object):
    ROOT_DIR = Path(path.abspath(path.join(__file__, "..")))
    ARTIFACT_DIR = ROOT_DIR / ".artifacts"
    MODEL_DIR = ROOT_DIR / ".models"
    BUILD_CACHE_DIR = ROOT_DIR / ".cache"


if Env.ARTIFACT_DIR.exists():
    if Env.ARTIFACT_DIR.is_dir():
        shutil.rmtree(Env.ARTIFACT_DIR)
    else:
        os.remove(Env.ARTIFACT_DIR)
Env.ARTIFACT_DIR.mkdir()
Env.MODEL_DIR.mkdir(exist_ok=True)
Env.BUILD_CACHE_DIR.mkdir(exist_ok=True)
