import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

from .models import REPORT_SCHEMA_VERSION


def report_document(
    results: Sequence[Dict],
    run_config: Optional[Dict] = None,
) -> Dict:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config or {},
        "cases": sorted(results, key=lambda result: result["case_id"]),
    }


def write_verification_report(
    results: Sequence[Dict],
    path: Path,
    run_config: Optional[Dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=".json",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                report_document(results, run_config),
                handle,
                indent=2,
            )
            handle.write("\n")
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

