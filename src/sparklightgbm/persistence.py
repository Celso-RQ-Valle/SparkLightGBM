import json
from pathlib import Path
def save_native_model(booster, path, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); booster.save_model(str(path)); path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
def load_native_model(path):
    from ._validation import require_runtime
    _, lgb = require_runtime(); return lgb.Booster(model_file=str(path))
