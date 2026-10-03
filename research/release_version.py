"""Source identity for archives that are not Git checkouts."""
from pathlib import Path
import json,subprocess
def source_revision():
    root=Path(__file__).resolve().parents[1]
    if (root/".git").exists():
        try:
            return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],
                    text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):
            pass
    return json.loads((root/"SOURCE_VERSION.json").read_text())["release_id"]
