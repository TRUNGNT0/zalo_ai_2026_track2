from pathlib import Path
from typing import Any
import json
import os
import tempfile

class JsonStore:
    """Persist JSON artifacts using an atomic file replacement."""

    @staticmethod
    def write(catalog: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file_handle:
                json.dump(catalog, file_handle, ensure_ascii=False, indent=2)
                file_handle.write("\n")
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
