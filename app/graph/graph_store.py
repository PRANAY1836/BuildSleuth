"""Stage 6: Provenance-backed Multi-Hop GraphRAG store using NetworkX and SQLite persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from pydantic import BaseModel, Field


class GraphEntity(BaseModel):
    id: str
    entity_type: str
    name: str
    properties: dict[str, object] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    relation: str
    provenance: str = "observed"  # observed vs inferred
    properties: dict[str, object] = Field(default_factory=dict)


class GraphStore:
    """Persistent graph store combining SQLite storage with NetworkX traversal."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_entities (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    properties_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    PRIMARY KEY (source_id, target_id, relation)
                )
                """
            )
            conn.commit()

    def add_entity(self, entity: GraphEntity) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO graph_entities (id, entity_type, name, properties_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    entity.id,
                    entity.entity_type,
                    entity.name,
                    json.dumps(entity.properties),
                ),
            )
            conn.commit()

    def add_edge(self, edge: GraphEdge) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO graph_edges (source_id, target_id, relation, provenance, properties_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    edge.source_id,
                    edge.target_id,
                    edge.relation,
                    edge.provenance,
                    json.dumps(edge.properties),
                ),
            )
            conn.commit()

    def get_neighbors(self, entity_id: str, relation: str | None = None) -> list[dict[str, object]]:
        with self._connect() as conn:
            if relation:
                rows = conn.execute(
                    """
                    SELECT e.*, ge.entity_type as target_type, ge.name as target_name
                    FROM graph_edges e
                    JOIN graph_entities ge ON ge.id = e.target_id
                    WHERE e.source_id = ? AND e.relation = ?
                    """,
                    (entity_id, relation),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT e.*, ge.entity_type as target_type, ge.name as target_name
                    FROM graph_edges e
                    JOIN graph_entities ge ON ge.id = e.target_id
                    WHERE e.source_id = ?
                    """,
                    (entity_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    def multi_hop_query(self, start_entity_id: str, max_hops: int = 2) -> list[dict[str, object]]:
        """Perform multi-hop graph traversal starting from an entity."""
        visited: set[str] = set()
        queue = [(start_entity_id, 0, [])]
        results: list[dict[str, object]] = []

        while queue:
            curr_id, hop, path = queue.pop(0)
            if curr_id in visited or hop > max_hops:
                continue
            visited.add(curr_id)

            neighbors = self.get_neighbors(curr_id)
            for n in neighbors:
                target_id = n["target_id"]
                new_path = path + [
                    {
                        "source": curr_id,
                        "target": target_id,
                        "relation": n["relation"],
                        "provenance": n["provenance"],
                    }
                ]
                results.append(
                    {
                        "entity_id": target_id,
                        "entity_name": n["target_name"],
                        "entity_type": n["target_type"],
                        "hops": hop + 1,
                        "path": new_path,
                    }
                )
                if hop + 1 < max_hops:
                    queue.append((target_id, hop + 1, new_path))

        return results