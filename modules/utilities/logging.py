"""
Convenience layer for making JSON logs.
"""


from datetime import datetime
import json
from pathlib import Path


def log(
        payload: dict,
        subdir_name: str,
        log_dir: Path | None = None
):
    """
    Make sure the directory exists for logging.
    """

    if log_dir is None:
        here = Path.cwd()
        log_dir = here / 'logs'

    print(log_dir)

    current_datetime = datetime.now()
    current_date = current_datetime.date()
    current_time = current_datetime.time()

    this_dir = log_dir / subdir_name / current_date.strftime(r'%Y-%m-%d')
    if not this_dir.exists():
        this_dir.mkdir(parents=True, exist_ok=True)

    file_name = this_dir / f'{current_time.strftime('%H%M%S')}.json'
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=4)
