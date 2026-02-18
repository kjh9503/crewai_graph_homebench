from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from crewai.memory import (
    EntityMemory,
    ExternalMemory,
    LongTermMemory,
    ShortTermMemory,
    GraphMemory,
)
from typing import Any



if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.task import Task





class ContextualMemory:
    """Aggregates and retrieves context from multiple memory sources."""

    def __init__(
        self,
        stm: ShortTermMemory,
        ltm: LongTermMemory,
        em: EntityMemory,
        exm: ExternalMemory,
        graph: GraphMemory,
        agent: Agent | None = None,
        task: Task | None = None,
    ) -> None:
        self.stm = stm
        self.ltm = ltm
        self.em = em
        self.exm = exm
        self.graph = graph
        self.agent = agent
        self.task = task

        if self.stm is not None:
            self.stm.agent = self.agent
            self.stm.task = self.task
        if self.ltm is not None:
            self.ltm.agent = self.agent
            self.ltm.task = self.task
        if self.graph is not None:
            self.graph.agent = self.agent
            self.graph.task = self.task
        if self.em is not None:
            self.em.agent = self.agent
            self.em.task = self.task
        if self.exm is not None:
            self.exm.agent = self.agent
            self.exm.task = self.task

    def _format_triplet(self, t: Any) -> str:
        """
        Accepts either:
        - "h, r, o" (str)
        - [h, o, {"label": r}]
        - (h, r, o) / (h, o, r) 형태가 섞여있을 가능성까지 방어적으로 처리
        """
        if isinstance(t, str):
            return t.strip()

        # common: [subj, obj, {"label": rel}]
        if isinstance(t, (list, tuple)) and len(t) == 3:
            a, b, c = t
            if isinstance(c, dict) and "label" in c:
                return f"{str(a)}, {str(c['label'])}, {str(b)}"
            # maybe (h, r, o)
            if isinstance(b, str) and isinstance(c, str):
                return f"{str(a)}, {str(b)}, {str(c)}"
            # fallback
            return ", ".join(map(str, t))

        return str(t)

    def build_context_for_task(self, task: Task, context: str) -> str:
        """Build contextual information for a task synchronously.

        Args:
            task: The task to build context for.
            context: Additional context string.
        Task : used_tools=0 tools_errors=0 delegations=0 i18n=I18N(prompt_file=None) 
        name='research_task' prompt_context='' description='Conduct a thorough 
        research about AI LLMs Make sure you find any interesting and relevant 
        
        information given the current year is 2026.\n' expected_output='A list 
        with 10 bullet points of the most relevant information about AI LLMs\n' 
        config=None callback=None agent=Agent(role=AI LLMs Senior Data Researcher, 
        goal=Uncover cutting-edge developments in AI LLMs, backstory=You're a seasoned 
        researcher with a knack for uncovering the latest developments in AI LLMs. Known 
        for your ability to find the most relevant information and present it in a clear 
        and concise manner.) context=NOT_SPECIFIED async_execution=False output_json=None 
        output_pydantic=None response_model=None output_file=None create_directory=True 
        output=None tools=[] input_files={} security_config=
        SecurityConfig(fingerprint=Fingerprint(metadata={})) 
        id=UUID('cff3c2a6-3205-48b8-a0c0-41673b80fdfc') human_input=False markdown=False 
        converter_cls=None processed_by_agents={'AI LLMs Senior Data Researcher\n'} 
        guardrail=None guardrails=None max_retries=None guardrail_max_retries=3 retry_count=0 
        start_time=datetime.datetime(2026, 2, 11, 18, 30, 14, 739302) end_time=None 
        allow_crewai_trigger_context=None
        
        Context : ''

        Returns:
            Formatted context string from all memory sources.
        """

        query = f"{task.description} {context}".strip()
        # task.description : Conduct a thorough research about AI LLMs Make sure you find any interesting and relevant information given the current year is 2026.
        if query == "":
            return ""

        
        context_parts = [
            # self._fetch_ltm_context(task.description),
            self._fetch_stm_context(query),
            self._fetch_entity_context(query),
            self._fetch_external_context(query),
            self._fetch_graph_context(query),
        ]
        return "\n".join(filter(None, context_parts))

    async def abuild_context_for_task(self, task: Task, context: str) -> str:
        """Build contextual information for a task asynchronously.

        Args:
            task: The task to build context for.
            context: Additional context string.

        Returns:
            Formatted context string from all memory sources.
        """
        query = f"{task.description} {context}".strip()

        if query == "":
            return ""

        # Fetch all contexts concurrently
        results = await asyncio.gather(
            self._afetch_ltm_context(task.description),
            self._afetch_graph_context(task.description),
            self._afetch_stm_context(query),
            self._afetch_entity_context(query),
            self._afetch_external_context(query),
        )

        return "\n".join(filter(None, results))

    def _fetch_stm_context(self, query: str) -> str:
        """
        Fetches recent relevant insights from STM related to the task's description and expected_output,
        formatted as bullet points.
        """

        if self.stm is None:
            return ""

        stm_results = self.stm.search(query)
        formatted_results = "\n".join(
            [f"- {result['content']}" for result in stm_results]
        )
        return f"Recent Insights:\n{formatted_results}" if stm_results else ""

    def _fetch_graph_context(self, task: str) -> str | None:
        """
        Fetch relevant subgraph/episodic context from GraphMemory and format as bullets.
        """

        graph = getattr(self, "graph", None)
        # print('============== self.graph : contextual_memory.py ===================')
        # print(graph)
        # print('============== self.graph : contextual_memory.py ===================')
        if getattr(self, "graph", None) is None:
            return ""

        results = graph.search(task, latest_n=10)
        if not results:
            return None

        triplet_lines: list[str] = []
        episodic_lines: list[str] = []
        hint_lines: list[str] = []

        for r in results:
            # 1) triplets
            triplets = r.get("triplets", [])
            if isinstance(triplets, list):
                for t in triplets:
                    s = self._format_triplet(t)
                    if s:
                        triplet_lines.append(s)

            # 2) episodic (optional)
            episodic = r.get("episodic", [])
            if isinstance(episodic, list):
                for e in episodic:
                    if isinstance(e, str) and e.strip():
                        episodic_lines.append(e.strip())

            # 3) optional hints from metadata (if you store them)
            md = r.get("metadata", {}) or {}
            # examples: md["suggestions"], md["notes"], md["warnings"] 등 무엇이든 넣을 수 있음
            for k in ("suggestions", "notes", "warnings"):
                vals = md.get(k)
                if isinstance(vals, list):
                    for v in vals:
                        if isinstance(v, str) and v.strip():
                            hint_lines.append(v.strip())
        
        
        chunks: list[str] = []

        if triplet_lines:
            triplet_bullets = "\n".join(f"- {x}" for x in triplet_lines[:50])  # 너무 길어지면 컷
            chunks.append(f"Relevant Graph Facts:\n{triplet_bullets}")

        if episodic_lines:
            episodic_bullets = "\n".join(f"- {x}" for x in episodic_lines[:20])
            chunks.append(f"Relevant Past Observations:\n{episodic_bullets}")

        if hint_lines:
            hint_bullets = "\n".join(f"- {x}" for x in hint_lines[:20])
            chunks.append(f"Historical Notes:\n{hint_bullets}")

        return "\n\n".join(chunks) if chunks else ""

    def _fetch_ltm_context(self, task: str) -> str | None:
        """
        Fetches historical data or insights from LTM that are relevant to the task's description and expected_output,
        formatted as bullet points.
        """

        if self.ltm is None:
            return ""

        ltm_results = self.ltm.search(task, latest_n=2)
        if not ltm_results:
            return None

        formatted_results = [
            suggestion
            for result in ltm_results
            for suggestion in result["metadata"]["suggestions"]
        ]
        formatted_results = list(dict.fromkeys(formatted_results))
        formatted_results = "\n".join([f"- {result}" for result in formatted_results])  # type: ignore # Incompatible types in assignment (expression has type "str", variable has type "list[str]")

        return f"Historical Data:\n{formatted_results}" if ltm_results else ""

    def _fetch_entity_context(self, query: str) -> str:
        """
        Fetches relevant entity information from Entity Memory related to the task's description and expected_output,
        formatted as bullet points.
        """
        if self.em is None:
            return ""

        em_results = self.em.search(query)
        formatted_results = "\n".join(
            [f"- {result['content']}" for result in em_results]
        )
        return f"Entities:\n{formatted_results}" if em_results else ""

    def _fetch_external_context(self, query: str) -> str:
        """
        Fetches and formats relevant information from External Memory.
        Args:
            query (str): The search query to find relevant information.
        Returns:
            str: Formatted information as bullet points, or an empty string if none found.
        """
        if self.exm is None:
            return ""

        external_memories = self.exm.search(query)

        if not external_memories:
            return ""

        formatted_memories = "\n".join(
            f"- {result['content']}" for result in external_memories
        )
        return f"External memories:\n{formatted_memories}"

    async def _afetch_stm_context(self, query: str) -> str:
        """Fetch recent relevant insights from STM asynchronously.

        Args:
            query: The search query.

        Returns:
            Formatted insights as bullet points, or empty string if none found.
        """
        if self.stm is None:
            return ""

        stm_results = await self.stm.asearch(query)
        formatted_results = "\n".join(
            [f"- {result['content']}" for result in stm_results]
        )
        return f"Recent Insights:\n{formatted_results}" if stm_results else ""
    from typing import Any

    


    async def _afetch_graph_context(self, task: str) -> str | None:
        """Fetch relevant context from GraphMemory asynchronously.

        Args:
            task: Query string (task description or natural language query).

        Returns:
            Formatted graph context (facts/episodic/notes) as bullet points,
            or None if none found.
        """
        if getattr(self, "graph_memory", None) is None:
            return ""

        results = await self.graph_memory.asearch(task, latest_n=10)
        if not results:
            return None

        triplet_lines: list[str] = []
        episodic_lines: list[str] = []
        note_lines: list[str] = []

        for r in results:
            # Triplets: list[str] or list[raw_triplet]
            triplets = r.get("triplets", [])
            if isinstance(triplets, list):
                for t in triplets:
                    s = self._format_triplet(t)
                    if s:
                        triplet_lines.append(s)

            # Episodic: list[str] (top observations etc.)
            episodic = r.get("episodic", [])
            if isinstance(episodic, list):
                for e in episodic:
                    if isinstance(e, str) and e.strip():
                        episodic_lines.append(e.strip())

            # Optional notes/suggestions if you store them in metadata
            md = r.get("metadata", {}) or {}
            for k in ("suggestions", "notes", "warnings"):
                vals = md.get(k)
                if isinstance(vals, list):
                    for v in vals:
                        if isinstance(v, str) and v.strip():
                            note_lines.append(v.strip())

        # Dedupe preserving order
        def dedupe(xs: list[str]) -> list[str]:
            seen = set()
            out: list[str] = []
            for x in xs:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            return out

        triplet_lines = dedupe(triplet_lines)
        episodic_lines = dedupe(episodic_lines)
        note_lines = dedupe(note_lines)

        chunks: list[str] = []

        if triplet_lines:
            chunks.append("Relevant Graph Facts:\n" + "\n".join(f"- {x}" for x in triplet_lines[:50]))

        if episodic_lines:
            chunks.append("Relevant Past Observations:\n" + "\n".join(f"- {x}" for x in episodic_lines[:20]))

        if note_lines:
            chunks.append("Historical Notes:\n" + "\n".join(f"- {x}" for x in note_lines[:20]))

        return "\n\n".join(chunks) if chunks else ""
    async def _afetch_ltm_context(self, task: str) -> str | None:
        """Fetch historical data from LTM asynchronously.

        Args:
            task: The task description to search for.

        Returns:
            Formatted historical data as bullet points, or None if none found.
        """
        if self.ltm is None:
            return ""

        ltm_results = await self.ltm.asearch(task, latest_n=2)
        if not ltm_results:
            return None

        formatted_results = [
            suggestion
            for result in ltm_results
            for suggestion in result["metadata"]["suggestions"]
        ]
        formatted_results = list(dict.fromkeys(formatted_results))
        formatted_results = "\n".join([f"- {result}" for result in formatted_results])  # type: ignore # Incompatible types in assignment (expression has type "str", variable has type "list[str]")

        return f"Historical Data:\n{formatted_results}" if ltm_results else ""

    async def _afetch_entity_context(self, query: str) -> str:
        """Fetch relevant entity information asynchronously.

        Args:
            query: The search query.

        Returns:
            Formatted entity information as bullet points, or empty string if none found.
        """
        if self.em is None:
            return ""

        em_results = await self.em.asearch(query)
        formatted_results = "\n".join(
            [f"- {result['content']}" for result in em_results]
        )
        return f"Entities:\n{formatted_results}" if em_results else ""

    async def _afetch_external_context(self, query: str) -> str:
        """Fetch relevant information from External Memory asynchronously.

        Args:
            query: The search query.

        Returns:
            Formatted information as bullet points, or empty string if none found.
        """
        if self.exm is None:
            return ""

        external_memories = await self.exm.asearch(query)

        if not external_memories:
            return ""

        formatted_memories = "\n".join(
            f"- {result['content']}" for result in external_memories
        )
        return f"External memories:\n{formatted_memories}"
