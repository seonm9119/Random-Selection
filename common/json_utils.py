import json
from pathlib import Path


def write_json(output_path, payload):
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
