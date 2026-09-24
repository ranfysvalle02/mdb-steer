"""MongoDB persistence: calibration set (vector-indexed), per-query telemetry, and run summaries."""

from __future__ import annotations

import time
from typing import Any

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

VECTOR_INDEX = "calibration_embedding"


class Store:
    def __init__(self, uri: str, db_name: str) -> None:
        self._client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = self._client[db_name]
        self.calibration = db["calibration"]
        self.telemetry = db["telemetry"]
        self.runs = db["benchmark_runs"]
        self.router_models = db["router_models"]

    def ping(self) -> None:
        self._client.admin.command("ping")

    def upsert_calibration(self, doc: dict[str, Any]) -> None:
        self.calibration.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def save_router_model(self, model: dict[str, Any]) -> None:
        self.router_models.insert_one(dict(model))

    def latest_router_model(self) -> dict[str, Any] | None:
        return self.router_models.find_one(sort=[("fitted_at", -1)])

    def ensure_vector_index(self, dimensions: int, timeout_s: float = 180) -> None:
        """Create the Atlas Vector Search index if needed and block until it is queryable."""
        if not list(self.calibration.list_search_indexes(VECTOR_INDEX)):
            self.calibration.create_search_index(
                SearchIndexModel(
                    name=VECTOR_INDEX,
                    type="vectorSearch",
                    definition={
                        "fields": [
                            {"type": "vector", "path": "embedding", "numDimensions": dimensions, "similarity": "cosine"}
                        ]
                    },
                )
            )

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            indexes = list(self.calibration.list_search_indexes(VECTOR_INDEX))
            if indexes and indexes[0].get("queryable"):
                return
            time.sleep(1)
        raise TimeoutError(f"vector index {VECTOR_INDEX!r} not queryable after {timeout_s:.0f}s")

    def nearest(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        """Return the k most similar calibration queries with their per-model scores."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": max(100, k * 20),
                    "limit": k,
                }
            },
            {"$project": {"query": 1, "scores": 1, "similarity": {"$meta": "vectorSearchScore"}}},
        ]
        return list(self.calibration.aggregate(pipeline))
