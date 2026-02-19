from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from crewai.agents.parser import AgentFinish
from crewai.events.event_listener import event_listener
from crewai.memory.entity.entity_memory_item import EntityMemoryItem
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.memory.graph_memory.graph_memory_item import GraphMemoryItem
from crewai.utilities.converter import ConverterError
from crewai.utilities.evaluators.task_evaluator import TaskEvaluator
from crewai.utilities.printer import Printer
from crewai.utilities.string_utils import sanitize_tool_name


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.crew import Crew
    from crewai.task import Task
    from crewai.utilities.i18n import I18N
    from crewai.utilities.types import LLMMessage


class CrewAgentExecutorMixin:
    crew: Crew | None
    agent: Agent
    task: Task | None
    iterations: int
    max_iter: int
    messages: list[LLMMessage]
    _i18n: I18N
    _printer: Printer = Printer()

    def _create_short_term_memory(self, output: AgentFinish) -> None:
        """Create and save a short-term memory item if conditions are met."""
        if (
            self.crew
            and self.agent
            and self.task
            and f"Action: {sanitize_tool_name('Delegate work to coworker')}"
            not in output.text
        ):
            try:
                if (
                    hasattr(self.crew, "_short_term_memory")
                    and self.crew._short_term_memory
                ):
                    self.crew._short_term_memory.save(
                        value=output.text,
                        metadata={
                            "observation": self.task.description,
                        },
                    )
            except Exception as e:
                self.agent._logger.log(
                    "error", f"Failed to add to short term memory: {e}"
                )

    def _create_external_memory(self, output: AgentFinish) -> None:
        """Create and save a external-term memory item if conditions are met."""
        if (
            self.crew
            and self.agent
            and self.task
            and hasattr(self.crew, "_external_memory")
            and self.crew._external_memory
        ):
            try:
                self.crew._external_memory.save(
                    value=output.text,
                    metadata={
                        "description": self.task.description,
                        "messages": self.messages,
                    },
                )
            except Exception as e:
                self.agent._logger.log(
                    "error", f"Failed to add to external memory: {e}"
                )

    def _create_long_term_memory(self, output: AgentFinish) -> None:
        """Create and save long-term and entity memory items based on evaluation."""
        if (
            self.crew
            and self.crew._long_term_memory
            and self.crew._entity_memory
            and self.task
            and self.agent
        ):
            try:
                ltm_agent = TaskEvaluator(self.agent)
                evaluation = ltm_agent.evaluate(self.task, output.text)

                if isinstance(evaluation, ConverterError):
                    return

                long_term_memory = LongTermMemoryItem(
                    task=self.task.description,
                    agent=self.agent.role,
                    quality=evaluation.quality,
                    datetime=str(time.time()),
                    expected_output=self.task.expected_output,
                    metadata={
                        "suggestions": evaluation.suggestions,
                        "quality": evaluation.quality,
                    },
                )
                self.crew._long_term_memory.save(long_term_memory)

                entity_memories = [
                    EntityMemoryItem(
                        name=entity.name,
                        type=entity.type,
                        description=entity.description,
                        relationships="\n".join(
                            [f"- {r}" for r in entity.relationships]
                        ),
                    )
                    for entity in evaluation.entities
                ]
                if entity_memories:
                    self.crew._entity_memory.save(entity_memories)
            except AttributeError as e:
                self.agent._logger.log(
                    "error", f"Missing attributes for long term memory: {e}"
                )
            except Exception as e:
                self.agent._logger.log(
                    "error", f"Failed to add to long term memory: {e}"
                )
        elif (
            self.crew
            and self.crew._long_term_memory
            and self.crew._entity_memory is None
        ):
            if self.agent and self.agent.verbose:
                self._printer.print(
                    content="Long term memory is enabled, but entity memory is not enabled. Please configure entity memory or set memory=True to automatically enable it.",
                    color="bold_yellow",
                )

    


    def _create_graph_memory(self, output: AgentFinish) -> None:
        """Create and save a graph memory item if conditions are met.

        - task.description 를 query/task key로 저장
        - output.text 를 observation/answer로 metadata에 넣음
        - triplets는 (가능하면) 기존에 구축된 그래프/백엔드에서 가져오거나,
        없으면 저장을 스킵(또는 빈 triplets로 저장)
        """
        if not (self.crew and self.agent and self.task):
            return

        if not hasattr(self.crew, "_graph_memory") or not self.crew._graph_memory:
            return

        try:
            if f"Action: {sanitize_tool_name('Delegate work to coworker')}" in output.text:
                return
        except Exception:
            # sanitize_tool_name 이 없거나 예외나면 그냥 계속 진행
            pass

        try:
            
            # 1) triplets 수집
            triplets: list[Any] = []

            # (A) storage에서 기존 triplet 로드 
            storage = getattr(self.crew._graph_memory, "storage", None)
            if storage is not None and hasattr(storage, "load_all_triplets"):
                try:
                    loaded_triplets = storage.load_all_triplets()
                    if loaded_triplets:
                        triplets.extend(loaded_triplets)
                        self.agent._logger.log(
                            "info",
                            f"Loaded {len(loaded_triplets)} existing triplets from storage"
                        )
                except Exception as e:
                    self.agent._logger.log(
                        "warning",
                        f"Failed to load existing triplets: {e}",
                    )

            backend = getattr(self.crew._graph_memory, "graph_backend", None)
            if backend is not None and hasattr(backend, "graph") and hasattr(backend.graph, "triplets"):
                backend_triplets = list(getattr(backend.graph, "triplets") or [])
                # 중복 제거하면서 추가
                for t in backend_triplets:
                    if t not in triplets:
                        triplets.append(t)
                
            
            # (C) backend graph가 generate()를 제공하면 output 텍스트에서 triplet 추출 시도
            if backend is not None and hasattr(backend, "graph") and hasattr(backend.graph, "generate"):
                extraction_prompt = (
                    "Extract factual relation triplets from the text.\n"
                    "Return strict JSON only, with this schema:\n"
                    '{"triplets":[{"subject":"...","relation":"...","object":"..."}]}\n'
                    "Rules:\n"
                    "- Keep entities concise.\n"
                    "- Use relation as a short verb phrase.\n"
                    "- Skip uncertain claims.\n"
                    "- If none, return {\"triplets\": []}.\n\n"
                    f"Text:\n{output.text}"
                )
                try:
                    """
                    output.text : 
                    As of 2026, the field of AI Language Learning Models (LLMs) has seen remarkable advancements and developments. Here are ten relevant points regarding these advancements:

                    1. **Increased Model Efficiency**: The latest LLMs have significantly improved their energy efficiency and computational requirements. Techniques such as quantization and model distillation have enabled models to run effectively on lower-spec hardware, making AI tools more accessible.
                    ...
                    """
                    generated = backend.graph.generate(extraction_prompt, jsn=True, t=0.0)
                    parsed = json.loads(generated) if isinstance(generated, str) else generated
                    generated_triplets = parsed.get("triplets", []) if isinstance(parsed, dict) else []
                    for item in generated_triplets:
                        if not isinstance(item, dict):
                            continue
                        subject = str(item.get("subject", "")).strip()
                        relation = str(item.get("relation", "")).strip()
                        obj = str(item.get("object", "")).strip()
                        if subject and relation and obj:
                            new_triplet = [subject, obj, {"label": relation}]
                            if new_triplet not in triplets:
                                triplets.append(new_triplet)
                except Exception as e:
                    self.agent._logger.log(
                        "warning",
                        f"Graph memory triplet extraction call failed: {e}",
                    )

            
            if not triplets:
                g = getattr(self.crew, "_triplet_graph", None) or getattr(self.crew, "_contriever_graph", None)
                if g is not None and hasattr(g, "triplets"):
                    triplets = list(getattr(g, "triplets") or [])
            
            
            
            if not triplets:
                self.agent._logger.log(
                    "warning",
                    "Graph memory triplet extraction failed: no graph backend or extracted triplets available. Skipping graph memory save.",
                )
                return


            metadata = {
                "description": self.task.description,
                "expected_output": getattr(self.task, "expected_output", ""),
                "agent_role": getattr(self.agent, "role", ""),
                "output_text": output.text,
                "messages": getattr(self, "messages", None),
            }

            metadata["quality"] = float(metadata.get("quality", 1.0))

            episodic = {
                "task_description": self.task.description,
                "output": output.text,
            }

            graph_item = GraphMemoryItem(
                task=self.task.description,
                agent=getattr(self.agent, "role", ""),
                expected_output=getattr(self.task, "expected_output", ""),
                datetime=str(time.time()),
                metadata=metadata,
                triplets=triplets,
                episodic=episodic,
            )
            self.crew._graph_memory.save(graph_item)

        except AttributeError as e:
            self.agent._logger.log("error", f"Missing attributes for graph memory: {e}")
        except Exception as e:
            self.agent._logger.log("error", f"Failed to add to graph memory: {e}")

    def _ask_human_input(self, final_answer: str) -> str:
        """Prompt human input with mode-appropriate messaging.

        Note: The final answer is already displayed via the AgentLogsExecutionEvent
        panel, so we only show the feedback prompt here.
        """
        from rich.panel import Panel
        from rich.text import Text

        formatter = event_listener.formatter
        formatter.pause_live_updates()

        try:
            # Training mode prompt (single iteration)
            if self.crew and getattr(self.crew, "_train", False):
                prompt_text = (
                    "TRAINING MODE: Provide feedback to improve the agent's performance.\n\n"
                    "This will be used to train better versions of the agent.\n"
                    "Please provide detailed feedback about the result quality and reasoning process."
                )
                title = "🎓 Training Feedback Required"
            # Regular human-in-the-loop prompt (multiple iterations)
            else:
                prompt_text = (
                    "Provide feedback on the Final Result above.\n\n"
                    "• If you are happy with the result, simply hit Enter without typing anything.\n"
                    "• Otherwise, provide specific improvement requests.\n"
                    "• You can provide multiple rounds of feedback until satisfied."
                )
                title = "💬 Human Feedback Required"

            content = Text()
            content.append(prompt_text, style="yellow")

            prompt_panel = Panel(
                content,
                title=title,
                border_style="yellow",
                padding=(1, 2),
            )
            formatter.console.print(prompt_panel)

            response = input()
            if response.strip() != "":
                formatter.console.print("\n[cyan]Processing your feedback...[/cyan]")
            return response
        finally:
            formatter.resume_live_updates()
