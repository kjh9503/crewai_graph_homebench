from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pathlib import Path
import os
import json

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


def _load_perception_data(qnum: str) -> dict:
    """Load perception data for a given qnum."""
    data_path = Path(__file__).resolve().parents[2] / "data" / "data_partial" / f"generated_user_story_perceptions_{qnum}.json"
    with data_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _load_core_context() -> str:
    """Load core context (user story)."""
    context_path = Path(__file__).resolve().parents[2] / "data" / "core_context.txt"
    with context_path.open('r', encoding='utf-8') as f:
        return f.read()


def _load_home_status() -> dict:
    """Load home status capabilities."""
    status_path = Path(__file__).resolve().parents[2] / "data" / "home_40_status.json"
    with status_path.open('r', encoding='utf-8') as f:
        return json.load(f)


@CrewBase
class HomebenchGraph():
    """HomebenchGraph crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def _get_perception_data(self) -> dict:
        """Get perception data for current QNUM."""
        qnum = os.environ.get("QNUM", "0")
        if getattr(self, "_perception_qnum", None) != qnum:
            self._perception_data = _load_perception_data(qnum)
            self._perception_qnum = qnum
        return self._perception_data

    def _get_core_context(self) -> str:
        """Get core context (user story)."""
        if not hasattr(self, "_core_context"):
            self._core_context = _load_core_context()
        return self._core_context

    def _get_home_status(self) -> dict:
        """Get home status capabilities."""
        if not hasattr(self, "_home_status"):
            self._home_status = _load_home_status()
        return self._home_status

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def user_profile_interpreter(self) -> Agent:
        return Agent(
            config=self.agents_config['user_profile_interpreter'], # type: ignore[index]
            verbose=True
        )

    @agent
    def context_interpreter(self) -> Agent:
        return Agent(
            config=self.agents_config['context_interpreter'], # type: ignore[index]
            verbose=True
        )

    # @agent
    # def capability_planner(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['capability_planner'], # type: ignore[index]
    #         verbose=True
    #     )

    @agent
    def command_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['command_writer'], # type: ignore[index]
            verbose=True
        )

    @agent
    def command_critic(self) -> Agent:
        return Agent(
            config=self.agents_config['command_critic'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def interpret_user_story_task(self) -> Task:
        return Task(
            config=self.tasks_config['interpret_user_story_task'], # type: ignore[index]
        )

    @task
    def interpret_context_task(self) -> Task:
        return Task(
            config=self.tasks_config['interpret_context_task'], # type: ignore[index]
        )

    # @task
    # def plan_action_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['plan_action_task'], # type: ignore[index]
    #     )

    @task
    def write_command_task(self) -> Task:
        return Task(
            config=self.tasks_config['write_command_task'], # type: ignore[index]
        )

    @task
    def validate_and_finalize_task(self) -> Task:
        qnum = os.environ.get("QNUM", "0")
        return Task(
            config=self.tasks_config['validate_and_finalize_task'], # type: ignore[index]
            output_file=f"output/command_{qnum}.md"
        )

    @crew
    def crew(self) -> Crew:
        """Creates the HomebenchGraph crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            memory=True,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
