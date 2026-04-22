from datetime import datetime
import json
from pathlib import Path


def log(payload: dict, subdir_name: str, log_dir: Path | None = None):
    if log_dir is None:
        log_dir = Path.cwd() / 'logs'

    now = datetime.now()
    this_dir = log_dir / subdir_name / now.strftime(r'%Y-%m-%d')
    this_dir.mkdir(parents=True, exist_ok=True)

    file_name = this_dir / f"{now.strftime('%H%M%S')}.json"
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=4)
