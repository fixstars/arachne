import os
import shutil
from pathlib import Path

if not os.environ.get("HOME"):
    raise RuntimeError(
        "You have to setup ${HOME} directory because the arachne library follow the XDG Base Directory specification"
    )


class Env(object):
    CACHE_HOME = Path(os.environ.get("HOME")) / ".cache" / "arachne"
    ARTIFACT_DIR = CACHE_HOME / "artifacts"


if Env.ARTIFACT_DIR.exists():
    if Env.ARTIFACT_DIR.is_dir():
        shutil.rmtree(Env.ARTIFACT_DIR)
    else:
        os.remove(Env.ARTIFACT_DIR)
Env.ARTIFACT_DIR.mkdir(parents=True)
