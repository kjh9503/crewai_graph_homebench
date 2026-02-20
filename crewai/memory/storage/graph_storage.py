import json
import os
import sqlite3
import re
from pathlib import Path
from typing import Any, Optional, Sequence
from openai import OpenAI
import aiosqlite
from copy import deepcopy
from crewai.utilities import Printer
from crewai.utilities.paths import db_storage_path



def clear_triplet(triplet):
    # triplet: [subj, obj, {"label": rel}] 를 기대
    subj, obj, rel = triplet[0], triplet[1], triplet[2].get("label", "")
    subj = str(subj).lower().strip('''"'. `;:''')
    obj  = str(obj).lower().strip('''"'. `;:''')
    rel  = str(rel).lower().strip('''"'. `;:''')
    return [subj, obj, {"label": rel}]


class InMemoryTripletGraph:
    """Minimal in-memory triplet graph used when no external graph is provided."""

    def __init__(self) -> None:
        self.triplets: list[list[Any]] = []
        self.items: list[str] = []
        self.model = os.getenv("GRAPH_MEMORY_MODEL", "gpt-4o-mini")
        self.system_prompt = (
            "You extract high-confidence knowledge graph triplets from text.\n"
            "Always return JSON only with schema: "
            "{\"triplets\":[{\"subject\":\"...\",\"relation\":\"...\",\"object\":\"...\"}]}.\n"
            "If no reliable relation is present, return {\"triplets\": []}."
        )
        self.total_amount = 0
        self.client: OpenAI | None = None
        self.embedding_model = os.getenv("GRAPH_MEMORY_EMBEDDING_MODEL", "text-embedding-3-small")
        self._embedding_cache: dict[str, list[float]] = {}

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", str(text).lower()))

    def _lexical_score(self, query: str, text: str) -> float:
        q = self._tokenize(query)
        t = self._tokenize(text)
        if not q or not t:
            return 0.0
        overlap = len(q.intersection(t)) / max(len(q), 1)
        if str(text).lower() in str(query).lower():
            overlap = max(overlap, 0.9)
        return overlap

    def _get_embedding(self, text: str) -> list[float] | None:
        key = str(text).strip().lower()
        if not key:
            return None
        if key in self._embedding_cache:
            return self._embedding_cache[key]

        if self.client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None
            self.client = OpenAI(api_key=api_key)

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=key,
            )
            vector = response.data[0].embedding
            self._embedding_cache[key] = vector
            return vector
        except Exception:
            return None

    def _cosine(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def generate(self, prompt: str, jsn: bool = False, t: float = 0.7) -> str | None:
        if self.client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set; InMemoryTripletGraph.generate cannot call OpenAI."
                )
            self.client = OpenAI(api_key=api_key)

        if jsn:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=t,
            )
        else:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=t,
            )
        return chat_completion.choices[0].message.content

    def str(self, triplet: list[Any]) -> str:
        return f"{triplet[0]} {triplet[2].get('label', '')} {triplet[1]}"

    def add_triplets(self, triplets: Sequence[Any]) -> None:
        for raw in triplets:
            if not isinstance(raw, (list, tuple)) or len(raw) < 3:
                continue
            if not isinstance(raw[2], dict):
                continue
            triplet = clear_triplet(list(raw))
            if triplet[2].get("label") == "free":
                continue
            if triplet not in self.triplets:
                self.triplets.append(triplet)
            if triplet[0] not in self.items:
                self.items.append(triplet[0])
            if triplet[1] not in self.items:
                self.items.append(triplet[1])

    def get_associated_triplets(self, items: set[str] | list[str], steps: int = 2) -> list[str]:
        lexical_threshold = float(os.getenv("GRAPH_MEMORY_LEXICAL_THRESHOLD", "0.2"))
        semantic_threshold = float(os.getenv("GRAPH_MEMORY_SEMANTIC_THRESHOLD", "0.0"))
        top_k_per_item = int(os.getenv("GRAPH_MEMORY_TOPK_PER_ITEM", "5"))

        current_items = deepcopy([str(x).lower() for x in items])
        associated_triplets: list[str] = []
        for _ in range(steps):
            frontier = set()
            for item in current_items:
                lexical_candidates: list[tuple[float, list[Any], str | None]] = []
                for triplet in self.triplets:
                    subj = str(triplet[0])
                    obj = str(triplet[1])
                    triplet_text = self.str(triplet)
                    subj_score = self._lexical_score(item, subj)
                    obj_score = self._lexical_score(item, obj)
                    full_score = self._lexical_score(item, triplet_text)
                    score = max(subj_score, obj_score, full_score)
                    if score >= lexical_threshold:
                        matched = subj if subj_score >= obj_score else obj
                        lexical_candidates.append((score, triplet, matched))
                

                candidates: list[tuple[float, list[Any], str | None]] = []
                if lexical_candidates:
                    candidates = sorted(
                        lexical_candidates,
                        key=lambda x: x[0],
                        reverse=True,
                    )[:top_k_per_item]
                else:
                    item_vec = self._get_embedding(item)
                    if item_vec is not None:
                        semantic_candidates: list[tuple[float, list[Any], str | None]] = []
                        for triplet in self.triplets:
                            triplet_text = self.str(triplet)
                            triplet_vec = self._get_embedding(triplet_text)
                            if triplet_vec is None:
                                
                                continue
                            score = self._cosine(item_vec, triplet_vec)
                            if score >= semantic_threshold:
                                semantic_candidates.append((score, triplet, None))
                        candidates = sorted(
                            semantic_candidates,
                            key=lambda x: x[0],
                            reverse=True,
                        )[:top_k_per_item]
                        
                    else:
                        # Offline fallback: keep best lexical neighbors even below threshold.
                        best_effort: list[tuple[float, list[Any], str | None]] = []
                        for triplet in self.triplets:
                            subj = str(triplet[0])
                            obj = str(triplet[1])
                            triplet_text = self.str(triplet)
                            subj_score = self._lexical_score(item, subj)
                            obj_score = self._lexical_score(item, obj)
                            full_score = self._lexical_score(item, triplet_text)
                            score = max(subj_score, obj_score, full_score)
                            matched = subj if subj_score >= obj_score else obj
                            if score > 0.0:
                                best_effort.append((score, triplet, matched))
                        candidates = sorted(
                            best_effort,
                            key=lambda x: x[0],
                            reverse=True,
                        )[:top_k_per_item]
                        

                for _, triplet, matched_entity in candidates:

                    triplet_text = self.str(triplet)
                    if triplet_text in associated_triplets:
                        continue
                    associated_triplets.append(triplet_text)
                    subj = str(triplet[0]).lower()
                    obj = str(triplet[1]).lower()
                    if matched_entity is None:
                        frontier.add(subj)
                        frontier.add(obj)
                    else:
                        frontier.add(obj if str(matched_entity).lower() == subj else subj)

            frontier.discard("itself")
            current_items = list(frontier)
        return associated_triplets

class TripletGraphBackendAdapter:
    

    def __init__(self, steps: int = 2):
        self.steps = steps
        self.graph = InMemoryTripletGraph()

        if not hasattr(self.graph, "items"):
            self.graph.items = []

    def add_triplets(self, triplets: Sequence[Any]) -> None:
        self.graph.add_triplets(list(triplets))

    def clear(self) -> None:
        self.graph.triplets = []
        self.graph.items = []

    def search(self, query: str, latest_n: int = 10) -> list[dict[str, Any]]:

        
        associated = self.graph.get_associated_triplets({query}, steps=self.steps)
       

        associated = associated[:latest_n]

        return [
            {
                "task": query,
                "triplets": associated,
                "metadata": {"source": "triplet_graph", "steps": self.steps},
            }
            for _ in range(1)  # 한 묶음으로 반환
        ]





class GraphJSONStorage:
    """JSON file storage for graph memory."""

    def __init__(
        self,
        json_path: str | None = None,
        append_mode: bool = True,
        verbose: bool = True,
    ) -> None:
        if json_path is None:
            json_path = str(Path(db_storage_path()) / "graph_memory_storage.json")
        self.json_path = json_path
        self.append_mode = append_mode
        self._verbose = verbose
        self._printer: Printer = Printer()
        Path(self.json_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_file()

    def _initialize_file(self) -> None:
        """Initialize JSON file if it doesn't exist."""
        if not Path(self.json_path).exists():
            try:
                with open(self.json_path, "w", encoding="utf-8") as f:
                    json.dump([], f)
            except Exception as e:
                if self._verbose:
                    self._printer.print(
                        content=f"MEMORY ERROR: JSON file init failed: {e}",
                        color="red",
                    )

    def save(
        self,
        task_description: str,
        metadata: dict[str, Any],
        datetime: str,
        score: int | float,
        triplets: list[Any] | None = None,
        episodic: dict[str, Any] | None = None,
    ) -> None:
        """Save a graph memory item to JSON file."""
        try:
            # Create new entry
            entry = {
                "task": task_description,
                "agent": metadata.get("agent", ""),
                "expected_output": metadata.get("expected_output", ""),
                "datetime": datetime,
                "score": float(score),
                "metadata": metadata,
                "triplets": triplets or [],
                "episodic": episodic or {},
            }

            # Read existing data
            existing_data = []
            if Path(self.json_path).exists():
                try:
                    with open(self.json_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            existing_data = json.loads(content)
                except json.JSONDecodeError:
                    existing_data = []

            # Append or overwrite
            if self.append_mode:
                existing_data.append(entry)
            else:
                existing_data = [entry]

            # Write back to file
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: JSON save failed: {e}",
                    color="red",
                )

    def load(self, task_description: str, latest_n: int) -> list[dict[str, Any]] | None:
        """Load graph memory items from JSON file matching task description."""
        try:
            if not Path(self.json_path).exists():
                return None

            with open(self.json_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return None
                all_data = json.loads(content)

            # Filter by task description and get latest N
            matching = [
                item for item in all_data
                if item.get("task") == task_description
            ]
            
            # Sort by datetime descending, then score ascending
            matching.sort(
                key=lambda x: (x.get("datetime", ""), -x.get("score", 0)),
                reverse=True
            )

            return matching[:latest_n] if matching else None

        except Exception as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: JSON load failed: {e}",
                    color="red",
                )
            return None

    def load_all_triplets(self) -> list[list[Any]]:
        """Load all triplets from all entries in the JSON file."""
        try:
            if not Path(self.json_path).exists():
                return []

            with open(self.json_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                all_data = json.loads(content)

            # Collect all unique triplets
            all_triplets = []
            seen = set()
            for item in all_data:
                for triplet in item.get("triplets", []):
                    # Create a hashable representation
                    triplet_str = json.dumps(triplet, sort_keys=True)
                    if triplet_str not in seen:
                        seen.add(triplet_str)
                        all_triplets.append(triplet)

            return all_triplets

        except Exception as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: load_all_triplets failed: {e}",
                    color="red",
                )
            return []

    def reset(self) -> None:
        """Clear all data from JSON file."""
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: JSON reset failed: {e}",
                    color="red",
                )

    async def asave(
        self,
        task_description: str,
        metadata: dict[str, Any],
        datetime: str,
        score: int | float,
        triplets: list[Any] | None = None,
        episodic: dict[str, Any] | None = None,
    ) -> None:
        """Async version of save (just calls sync version)."""
        self.save(task_description, metadata, datetime, score, triplets, episodic)

    async def aload(self, task_description: str, latest_n: int) -> list[dict[str, Any]] | None:
        """Async version of load (just calls sync version)."""
        return self.load(task_description, latest_n)

    async def areset(self) -> None:
        """Async version of reset (just calls sync version)."""
        self.reset()


class GraphSQLiteStorage:
    """SQLite storage for graph memory."""

    def __init__(self, db_path: str | None = None, verbose: bool = True) -> None:
        if db_path is None:
            db_path = str(Path(db_storage_path()) / "graph_memory_storage.db")
        self.db_path = db_path
        self._verbose = verbose
        self._printer: Printer = Printer()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def _initialize_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS graph_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_description TEXT,
                        metadata TEXT,
                        datetime TEXT,
                        score REAL,
                        triplets TEXT,
                        episodic TEXT
                    )
                    """
                )
                conn.commit()
        except sqlite3.Error as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: DB init failed: {e}",
                    color="red",
                )

    def save(
        self,
        task_description: str,
        metadata: dict[str, Any],
        datetime: str,
        score: int | float,
        triplets: list[Any] | None = None,
        episodic: dict[str, Any] | None = None,
    ) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO graph_memories (task_description, metadata, datetime, score, triplets, episodic)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_description,
                        json.dumps(metadata, ensure_ascii=False),
                        datetime,
                        float(score),
                        json.dumps(triplets or [], ensure_ascii=False),
                        json.dumps(episodic or {}, ensure_ascii=False),
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: save failed: {e}",
                    color="red",
                )

    def load(self, task_description: str, latest_n: int) -> list[dict[str, Any]] | None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT metadata, datetime, score, triplets, episodic
                    FROM graph_memories
                    WHERE task_description = ?
                    ORDER BY datetime DESC, score ASC
                    LIMIT {int(latest_n)}
                    """,  # nosec # noqa: S608 (same style as CrewAI)
                    (task_description,),
                )
                rows = cursor.fetchall()
                if rows:
                    out = []
                    for md, dt, sc, tr, ep in rows:
                        out.append(
                            {
                                "metadata": json.loads(md) if md else {},
                                "datetime": dt,
                                "score": sc,
                                "triplets": json.loads(tr) if tr else [],
                                "episodic": json.loads(ep) if ep else {},
                            }
                        )
                    return out
        except sqlite3.Error as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: load failed: {e}",
                    color="red",
                )
        return None

    def reset(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM graph_memories")
                conn.commit()
        except sqlite3.Error as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: reset failed: {e}",
                    color="red",
                )

    async def asave(
        self,
        task_description: str,
        metadata: dict[str, Any],
        datetime: str,
        score: int | float,
        triplets: list[Any] | None = None,
        episodic: dict[str, Any] | None = None,
    ) -> None:
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """
                    INSERT INTO graph_memories (task_description, metadata, datetime, score, triplets, episodic)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_description,
                        json.dumps(metadata, ensure_ascii=False),
                        datetime,
                        float(score),
                        json.dumps(triplets or [], ensure_ascii=False),
                        json.dumps(episodic or {}, ensure_ascii=False),
                    ),
                )
                await conn.commit()
        except aiosqlite.Error as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: asave failed: {e}",
                    color="red",
                )

    async def aload(self, task_description: str, latest_n: int) -> list[dict[str, Any]] | None:
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    f"""
                    SELECT metadata, datetime, score, triplets, episodic
                    FROM graph_memories
                    WHERE task_description = ?
                    ORDER BY datetime DESC, score ASC
                    LIMIT {int(latest_n)}
                    """,  # nosec # noqa: S608
                    (task_description,),
                )
                rows = await cursor.fetchall()
                if rows:
                    out = []
                    for md, dt, sc, tr, ep in rows:
                        out.append(
                            {
                                "metadata": json.loads(md) if md else {},
                                "datetime": dt,
                                "score": sc,
                                "triplets": json.loads(tr) if tr else [],
                                "episodic": json.loads(ep) if ep else {},
                            }
                        )
                    return out
        except aiosqlite.Error as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: aload failed: {e}",
                    color="red",
                )
        return None

    async def areset(self) -> None:
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("DELETE FROM graph_memories")
                await conn.commit()
        except aiosqlite.Error as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: areset failed: {e}",
                    color="red",
                )
