import json
import sqlite3
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..core.models import RefinementCut


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def create(cls, root: Path) -> "Workspace":
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        for directory in ("index", "generations", "logs"):
            (root / directory).mkdir(exist_ok=True)
        return cls(root)

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def generations_dir(self) -> Path:
        return self.root / "generations"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"


class LocalPartitionStore:
    """Immutable local generations backed by memory-mapped NumPy labels."""

    def __init__(self, root: Path, edge_count: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.edge_count = int(edge_count)
        if not (self.root / "CURRENT").exists():
            self._initialize()

    def _initialize(self) -> None:
        generation_dir = self._generation_dir(0)
        generation_dir.mkdir()
        for side in ("A", "B"):
            labels = np.lib.format.open_memmap(
                generation_dir / f"labels_{side}.npy",
                mode="w+",
                dtype=np.int64,
                shape=(self.edge_count,),
            )
            labels[:] = 0
            labels.flush()
            labels._mmap.close()
        initial_part_count = 0 if self.edge_count == 0 else 1
        self._write_manifest(0, initial_part_count, initial_part_count)
        self._set_current(0)

    def _generation_dir(self, generation: int) -> Path:
        return self.root / f"{generation:08d}"

    def _write_manifest(
        self,
        generation: int,
        part_count_a: int,
        part_count_b: int,
    ) -> None:
        generation_dir = self._generation_dir(generation)
        temporary = generation_dir / "manifest.json.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "generation": generation,
                    "edge_count": self.edge_count,
                    "part_count_A": part_count_a,
                    "part_count_B": part_count_b,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        temporary.replace(generation_dir / "manifest.json")

    def _set_current(self, generation: int) -> None:
        temporary = self.root / "CURRENT.tmp"
        temporary.write_text(f"{generation}\n", encoding="ascii")
        temporary.replace(self.root / "CURRENT")

    @property
    def current_generation(self) -> int:
        return int((self.root / "CURRENT").read_text(encoding="ascii").strip())

    def labels_path(self, generation: int, side: str) -> Path:
        normalized = side.upper()
        if normalized not in {"A", "B"}:
            raise ValueError("side must be 'A' or 'B'")
        return self._generation_dir(generation) / f"labels_{normalized}.npy"

    def labels(self, generation: int, side: str) -> np.ndarray:
        return np.load(self.labels_path(generation, side), mmap_mode="r")

    def part_count(self, generation: int, side: str) -> int:
        normalized = side.upper()
        with open(
            self._generation_dir(generation) / "manifest.json",
            encoding="utf-8",
        ) as handle:
            manifest = json.load(handle)
        return int(manifest[f"part_count_{normalized}"])

    @staticmethod
    def _group_cuts(
        cuts: Sequence[RefinementCut],
    ) -> Tuple[Dict[int, List[np.ndarray]], List[np.memmap]]:
        grouped: Dict[int, List[np.ndarray]] = defaultdict(list)
        opened = []
        for cut in cuts:
            if cut.edge_indices_path is not None:
                indices = np.memmap(
                    cut.edge_indices_path,
                    mode="r",
                    dtype=np.int64,
                )
                opened.append(indices)
            else:
                indices = np.unique(
                    np.asarray(cut.edge_indices, dtype=np.int64)
                )
            if len(indices):
                grouped[int(cut.parent_label)].append(indices)
        return grouped, opened

    def _refine_side(
        self,
        source: np.ndarray,
        target_path: Path,
        cuts: Sequence[RefinementCut],
    ) -> int:
        grouped, opened = self._group_cuts(cuts)
        cursors = {
            parent: [0] * len(parent_cuts)
            for parent, parent_cuts in grouped.items()
        }
        target = np.lib.format.open_memmap(
            target_path,
            mode="w+",
            dtype=np.int64,
            shape=(self.edge_count,),
        )
        mapping_path = target_path.with_suffix(".mapping.sqlite3")
        connection = sqlite3.connect(mapping_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute(
            "CREATE TABLE labels ("
            "parent INTEGER NOT NULL, "
            "signature BLOB NOT NULL, "
            "label INTEGER NOT NULL, "
            "PRIMARY KEY (parent, signature)"
            ") WITHOUT ROWID"
        )
        cache: OrderedDict[Tuple[int, bytes], int] = OrderedDict()
        cache_limit = 4096
        try:
            next_label = 0
            for edge_index in range(self.edge_count):
                parent = int(source[edge_index])
                signature = bytearray(
                    (len(grouped.get(parent, ())) + 7) // 8
                )
                for cut_index, cut in enumerate(grouped.get(parent, ())):
                    cursor = cursors[parent][cut_index]
                    while cursor < len(cut) and cut[cursor] < edge_index:
                        cursor += 1
                    cursors[parent][cut_index] = cursor
                    if cursor < len(cut) and cut[cursor] == edge_index:
                        signature[cut_index // 8] |= 1 << (cut_index % 8)
                key = parent, bytes(signature)
                label = cache.pop(key, None)
                if label is None:
                    row = connection.execute(
                        "SELECT label FROM labels "
                        "WHERE parent = ? AND signature = ?",
                        key,
                    ).fetchone()
                    if row is None:
                        label = next_label
                        connection.execute(
                            "INSERT INTO labels VALUES (?, ?, ?)",
                            (parent, key[1], label),
                        )
                        next_label += 1
                    else:
                        label = int(row[0])
                cache[key] = label
                if len(cache) > cache_limit:
                    cache.popitem(last=False)
                target[edge_index] = label
            target.flush()
            connection.commit()
        finally:
            target._mmap.close()
            connection.close()
            for indices in opened:
                indices._mmap.close()
            if mapping_path.exists():
                mapping_path.unlink()
        return next_label

    def commit_refinement(
        self,
        generation: int,
        cuts_a: Sequence[RefinementCut],
        cuts_b: Sequence[RefinementCut],
    ) -> int:
        if generation != self.current_generation:
            raise ValueError("refinement must start from the current generation")
        next_generation = generation + 1
        generation_dir = self._generation_dir(next_generation)
        generation_dir.mkdir()
        try:
            source_a = self.labels(generation, "A")
            try:
                part_count_a = self._refine_side(
                    source_a,
                    generation_dir / "labels_A.npy",
                    cuts_a,
                )
            finally:
                source_a._mmap.close()
            source_b = self.labels(generation, "B")
            try:
                part_count_b = self._refine_side(
                    source_b,
                    generation_dir / "labels_B.npy",
                    cuts_b,
                )
            finally:
                source_b._mmap.close()
            self._write_manifest(
                next_generation,
                part_count_a,
                part_count_b,
            )
            self._set_current(next_generation)
        except Exception:
            for path in generation_dir.glob("*"):
                path.unlink()
            generation_dir.rmdir()
            raise
        return next_generation
