import os
import shutil
from pathlib import Path

_HOME_DIR = os.environ.get("HOME")
if not _HOME_DIR:
    raise RuntimeError(
        "You have to setup ${HOME} directory because the arachne library follow the XDG Base Directory specification"
    )


class Env(object):
    CACHE_HOME = Path(_HOME_DIR) / ".cache" / "arachne"
    ARTIFACT_DIR = CACHE_HOME / "artifacts"


if Env.ARTIFACT_DIR.exists():
    if Env.ARTIFACT_DIR.is_dir():
        shutil.rmtree(Env.ARTIFACT_DIR)
    else:
        os.remove(Env.ARTIFACT_DIR)
Env.ARTIFACT_DIR.mkdir(parents=True)
