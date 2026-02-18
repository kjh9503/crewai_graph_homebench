from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pathlib import Path
import os
import json
from qa_task.tools.custom_tool import KnowledgeGraphReaderTool, ParagraphRetrievalTool


def _load_question_data(qnum: str) -> list[dict]:
    data_path = Path(__file__).resolve().parents[2] / "data" / f"musique_train_{qnum}.json"
    with data_path.open('r') as f:
        question_data = json.load(f)
    return [
        {
            key: item[key]
            for key in item.keys()
            if key not in ['answer', 'answer_aliases', 'answerable']
        }
        for item in question_data
    ]


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class QaTask():
    """QaTask crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def _get_question_data(self) -> list[dict]:
        qnum = os.environ.get("QNUM", "0")
        if getattr(self, "_question_qnum", None) != qnum:
            self._question_data = _load_question_data(qnum)
            self._question_qnum = qnum
        return self._question_data

    def _get_questions(self) -> list[dict]:
        return [
            {
                "question": item.get("question"),
            }
            for item in self._get_question_data()
        ]
    def _get_decomposed_questions(self) -> list[dict]:
        return [
            {
                "decomposed_questions": ". ".join(
                    [dq.get("question") for dq in (item.get("question_decomposition") or [])]
                )
            }
            for item in self._get_question_data()
        ]

    def _get_paragraphs(self) -> list[dict]:
        return [
            {
                "paragraphs": item.get("paragraphs", []),
            }
            for item in self._get_question_data()
        ]

    def _get_knowledge_graph_triples(self) -> str:
        return str(self._get_kg_path())

    def _get_kg_path(self) -> Path:
        qnum = os.environ.get("QNUM", "0")
        mode = os.environ.get("KG_MODE", "merge").strip().lower()
        if mode == "split":
            return Path.cwd() / f"knowledge_graph_{qnum}.json"
        return Path.cwd() / "knowledge_graph.json"

    def _append_knowledge_graph_output(self, output) -> None:
        raw = str(output.raw or "").strip()
        if not raw:
            raise ValueError("knowledge_graph_task produced empty output.")
        try:
            new_data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"knowledge_graph_task produced invalid JSON: {exc}"
            ) from exc
        if not isinstance(new_data, dict):
            raise ValueError("knowledge_graph_task output must be a JSON object.")

        kg_path = self._get_kg_path()
        if kg_path.exists():
            existing_raw = kg_path.read_text().strip()
            if not existing_raw:
                raise ValueError("knowledge_graph.json exists but is empty.")
            try:
                existing_data = json.loads(existing_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"knowledge_graph.json contains invalid JSON: {exc}"
                ) from exc
            if not isinstance(existing_data, dict):
                raise ValueError("knowledge_graph.json must contain a JSON object.")
        else:
            existing_data = {}

        if not existing_data:
            merged = new_data
        else:
            merged = dict(existing_data)
            merged["schema_version"] = merged.get(
                "schema_version", new_data.get("schema_version")
            )
            existing_relations = merged.get("relation_inventory", [])
            new_relations = new_data.get("relation_inventory", [])
            merged["relation_inventory"] = list(
                dict.fromkeys(existing_relations + new_relations)
            )
            merged["steps"] = merged.get("steps", []) + new_data.get("steps", [])
            merged["global_triples"] = merged.get("global_triples", []) + new_data.get(
                "global_triples", []
            )
        kg_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    
    def _get_paragraph_retrieval_tool(self) -> ParagraphRetrievalTool:
        """Create a ParagraphRetrievalTool with current question's paragraphs."""
        question_data = self._get_question_data()
        paragraphs = []
        for item in question_data:
            paragraphs.extend(item.get("paragraphs", []))
        return ParagraphRetrievalTool(paragraphs=paragraphs)
    
    @agent
    def paragraph_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config['paragraph_retriever'], # type: ignore[index]
            verbose=True,
            tools=[self._get_paragraph_retrieval_tool()]
        )
    
    @agent
    def question_answerer(self) -> Agent:
        return Agent(
            config=self.agents_config['question_answerer'], # type: ignore[index]
            verbose=True,
            tools=[self._get_paragraph_retrieval_tool(), KnowledgeGraphReaderTool()]
        )
    
    @agent
    def final_answer_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['final_answer_generator'], # type: ignore[index]
            verbose=True,
            tools=[KnowledgeGraphReaderTool()]
        )
    
    @agent
    def knowledge_graph_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['knowledge_graph_agent'], # type: ignore[index]
            verbose=True
        )

    @agent
    def answering_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['answering_agent'], # type: ignore[index]
            verbose=True,
            tools=[KnowledgeGraphReaderTool()]
        )


    @task
    def retrieve_paragraphs_task(self) -> Task:
        return Task(
            config=self.tasks_config['retrieve_paragraphs_task'], # type: ignore[arg-type]
        )
    
    @task
    def answer_subquestions_task(self) -> Task:
        qnum = os.environ.get("QNUM", "0")
        return Task(
            config=self.tasks_config['answer_subquestions_task'], # type: ignore[arg-type]
            output_file=f"output/answer_subquestions_{qnum}.md"
        )
    
    @task
    def generate_final_answer_task(self) -> Task:
        qnum = os.environ.get("QNUM", "0")
        return Task(
            config=self.tasks_config['generate_final_answer_task'], # type: ignore[arg-type]
            output_file=f"output/answer_{qnum}.md"
        )
    
    @task
    def knowledge_graph_task(self) -> Task:
        return Task(
            config=self.tasks_config['knowledge_graph_task'], # type: ignore[index]
            callback=self._append_knowledge_graph_output
        )

    @task
    def answering_task(self) -> Task:
        qnum = os.environ.get("QNUM", "0")
        return Task(
            config=self.tasks_config['answering_task'], # type: ignore[index],
            output_file=f"output/answer_kg_{qnum}.md"
        )

    @crew
    def crew(self) -> Crew:
        """Creates the QaTask crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
            tracing=False
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
