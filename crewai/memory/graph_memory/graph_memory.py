import time
from typing import Any, Optional

from pydantic import Field, ConfigDict

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryQueryStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)
from crewai.memory.graph_memory.graph_memory_item import GraphMemoryItem
from crewai.memory.memory import Memory


from typing import Any, Sequence
from crewai.memory.storage.graph_storage import GraphSQLiteStorage, GraphJSONStorage, TripletGraphBackendAdapter


class GraphMemory(Memory):
    """
    Drop-in replacement for LongTermMemory:
    - save/search/asave/asearch/reset
    - emits the same events
    - stores entries in JSON or SQLite
    - optionally uses a graph backend for retrieval
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_backend: Optional[TripletGraphBackendAdapter] = Field(
        default=None,
        description="Optional graph backend adapter for retrieval"
    )
    

    def __init__(
        self,
        storage: Optional[GraphSQLiteStorage | GraphJSONStorage] = None,
        path: Optional[str] = None,
        use_json: bool = True,
        append_mode: bool = True,
        # graph_backend: Optional[TripletGraphBackendAdapter] = None,
    ) -> None:
        if storage is None:
            if use_json:
                storage = GraphJSONStorage(json_path=path, append_mode=append_mode) if path else GraphJSONStorage(append_mode=append_mode)
            else:
                storage = GraphSQLiteStorage(db_path=path) if path else GraphSQLiteStorage()
        super().__init__(storage=storage)
        self.graph_backend = TripletGraphBackendAdapter()


    def save(self, item: GraphMemoryItem) -> None:  # type: ignore[override]
        crewai_event_bus.emit(
            self,
            event=MemorySaveStartedEvent(
                value=item.task,
                metadata=item.metadata,
                agent_role=item.agent,
                source_type="graph_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            # avoid mutating caller's metadata
            metadata = dict(item.metadata)
            metadata.update({"agent": item.agent, "expected_output": item.expected_output})

            # LongTermMemory assumes metadata["quality"] exists.
            # Here: default to 1.0 if missing (lenient). If you want strict, raise instead.
            
            quality = float(metadata.get("quality", 1.0))
            
            gm = getattr(self.crew, "_graph_memory", None)
            

            backend = getattr(gm, "graph_backend", None)
            

            g = getattr(backend, "graph", None) if backend else None
            

            
            # item.triplets.append(['cozy temperature', '30 celcius degree', {'label': 'is'}])
            self.storage.save(
                task_description=item.task,
                score=quality,
                metadata=metadata,
                datetime=item.datetime,
                triplets=item.triplets,
                episodic=item.episodic,
            )

            # keep backend in sync (optional)
            if self.graph_backend is not None and item.triplets:
                self.graph_backend.add_triplets(item.triplets)


            crewai_event_bus.emit(
                self,
                event=MemorySaveCompletedEvent(
                    value=item.task,
                    metadata=metadata,
                    agent_role=item.agent,
                    save_time_ms=(time.time() - start_time) * 1000,
                    source_type="graph_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemorySaveFailedEvent(
                    value=item.task,
                    metadata=item.metadata,
                    agent_role=item.agent,
                    error=str(e),
                    source_type="graph_memory",
                ),
            )
            raise

    def search(self, task: str, latest_n: int = 3) -> list[dict[str, Any]]:  # type: ignore[override]
        crewai_event_bus.emit(
            self,
            event=MemoryQueryStartedEvent(
                query=task,
                limit=latest_n,
                source_type="graph_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            if self.graph_backend is not None:
                results = self.graph_backend.search(task, latest_n)
            else:
                results = self.storage.load(task, latest_n)

            crewai_event_bus.emit(
                self,
                event=MemoryQueryCompletedEvent(
                    query=task,
                    results=results,
                    limit=latest_n,
                    query_time_ms=(time.time() - start_time) * 1000,
                    source_type="graph_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )
            return results or []
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemoryQueryFailedEvent(
                    query=task,
                    limit=latest_n,
                    error=str(e),
                    source_type="graph_memory",
                ),
            )
            raise

    async def asave(self, item: GraphMemoryItem) -> None:  # type: ignore[override]
        self.save(item)

    async def asearch(self, task: str, latest_n: int = 3) -> list[dict[str, Any]]:  # type: ignore[override]
        return self.search(task, latest_n)

    def reset(self) -> None:
        self.storage.reset()
        if self.graph_backend is not None:
            try:
                self.graph_backend.clear()
            except Exception:
                pass