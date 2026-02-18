#!/usr/bin/env python
import sys
import os
import warnings
from pathlib import Path

from datetime import datetime

from qa_task.crew import QaTask

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Ensure output directory exists
Path("output").mkdir(exist_ok=True)

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    qnum = os.environ.get("QNUM") or (sys.argv[1] if len(sys.argv) > 1 else "0")
    os.environ["QNUM"] = str(qnum)
    
    qa_task = QaTask()
    inputs = {
        'topic': 'Reading Comprehension Test',
        'current_year': str(datetime.now().year),
        'question': qa_task._get_questions()[0]['question'],
        'question_decomposition': qa_task._get_decomposed_questions()[0]['decomposed_questions'],
        'paragraphs': qa_task._get_paragraphs(),
        'decomposed_questions': qa_task._get_decomposed_questions(),
        'questions': qa_task._get_questions(),
        'knowledge_graph_triples': qa_task._get_knowledge_graph_triples()
    }

    try:
        qa_task.crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        QaTask().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        QaTask().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        QaTask().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = QaTask().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
