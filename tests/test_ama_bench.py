"""Tests for the AMA-Bench adapter: parsing, memory construction, retrieval."""

from __future__ import annotations

from ama_bench.extract import extract_final_answer, parse_answer_blocks
from ama_bench.fold import NovaCodeMemoryBuilder
from ama_bench.retrieve import render_evidence, score_candidates
from ama_bench.steps import parse_trajectory_text, steps_to_text

SAMPLE_TRAJECTORY = """Step 0:
Action: navigate(room=kitchen)
Observation: You are in the kitchen. There is a red apple on the table.

Step 1:
Action: grab(apple)
Observation: You are now holding a red apple.

Step 2:
Action: read_file(path="recipes.txt")
Observation: Recipes mention apples, flour, and sugar.

Step 3:
Action: run_shell(cmd="grep -n apple recipes.txt")
Observation: 3: apple pie recipe requires 4 apples.

Step 4:
Action: write_file(path="summary.txt", content="apple pie needs 4 apples")
Observation: File written.

Step 5:
Action: run_shell(cmd="python -m unittest tests")
Observation: 3 tests passed.
"""


def test_parse_trajectory_text_roundtrip() -> None:
    steps = parse_trajectory_text(SAMPLE_TRAJECTORY)
    assert len(steps) == 6
    assert steps[0].turn_idx == 0
    assert steps[0].action == "navigate(room=kitchen)"
    assert "red apple" in steps[0].observation
    assert steps[5].turn_idx == 5
    assert "3 tests passed" in steps[5].observation
    # Rendered text round-trips through the parser.
    assert len(parse_trajectory_text(steps_to_text(steps))) == 6


def test_parse_trajectory_turn_marker() -> None:
    text = "Turn 3:\nAction: move(east)\nObservation: corridor\nTurn 4:\nAction: move(west)\nObservation: room"
    steps = parse_trajectory_text(text)
    assert [step.turn_idx for step in steps] == [3, 4]


def test_memory_construction_deterministic() -> None:
    builder = NovaCodeMemoryBuilder()
    memory = builder.build(parse_trajectory_text(SAMPLE_TRAJECTORY), task="collect items")
    assert memory.task_state.objective == "collect items"
    # Some trajectory groups must have been folded into task/tool state.
    assert len(memory.trajectory.groups) < 6
    assert memory.stats["groups_folded"] > 0
    assert memory.stats["model_folds"] == 0
    assert memory.stats["fallback_folds"] > 0
    assert memory.stats["task_entries"] > 0 or memory.stats["tool_entries"] > 0


def test_memory_construction_empty() -> None:
    memory = NovaCodeMemoryBuilder().build([], task="nothing")
    assert memory.stats["steps"] == 0
    assert len(memory.trajectory.groups) == 0


def test_retrieval_ranks_apple_evidence_first() -> None:
    builder = NovaCodeMemoryBuilder()
    memory = builder.build(parse_trajectory_text(SAMPLE_TRAJECTORY), task="collect items")
    candidates = score_candidates(memory, "How many apples does the pie recipe require?")
    assert candidates
    top_text = candidates[0]["text"].lower()
    assert "apple" in top_text
    # The recipe answer should surface in the top candidates.
    blob = " ".join(c["text"] for c in candidates).lower()
    assert "pie" in blob or "4 apples" in blob


def test_retrieval_surfaces_key_sequences_for_intent_questions() -> None:
    from coding_agent.structured_context.models import KeySequence

    memory = NovaCodeMemoryBuilder().build(parse_trajectory_text(SAMPLE_TRAJECTORY), task="collect items")
    memory.task_state.key_sequences.append(
        KeySequence(
            id="k-1",
            pattern="grab apple at step 1, then read recipes at step 2",
            intent="collect the apple before checking the recipe",
            step_range="1-2",
        )
    )
    candidates = score_candidates(memory, "Why did the agent pick up the apple first?")
    assert candidates
    top = candidates[0]
    assert top["type"] == "sequence"
    assert "collect the apple" in top["text"].lower()
    assert top["meta"]["step_range"] == "1-2"
    rendered = render_evidence(candidates)
    assert "step_range=1-2" in rendered


def test_render_evidence_shapes() -> None:
    memory = NovaCodeMemoryBuilder().build(parse_trajectory_text(SAMPLE_TRAJECTORY))
    rendered = render_evidence(score_candidates(memory, "apple"))
    assert rendered.startswith("<evidence")
    assert "apple" in rendered.lower()
    empty = render_evidence([], max_total_chars=1_000)
    assert "no relevant evidence" in empty


def test_extract_final_answer_mcq() -> None:
    assert extract_final_answer("###Answer: (A)", mcq_mode=True) == "(A)"
    assert extract_final_answer("##Answer: (A)(C)", mcq_mode=True) == "(A)(C)"
    # First line only for MCQ; markers are stripped.
    assert extract_final_answer("##Answer: (B)\nsome trailing note", mcq_mode=True) == "(B)"
    # Bare option token without a marker.
    assert extract_final_answer("The answer is (D).", mcq_mode=True) == "(D)"
    # Thinking blocks are removed.
    assert extract_final_answer("<think>reasoning</think>\n##Answer: (A)", mcq_mode=True) == "(A)"


def test_extract_final_answer_open_end() -> None:
    text = "##Answer: The apple pie requires 4 apples."
    assert extract_final_answer(text, mcq_mode=False) == "The apple pie requires 4 apples."


def test_parse_answer_blocks() -> None:
    response = (
        "Answer[1]: (A)\n"
        "Answer[2]: (B)(D)\n"
        "Answer[3]: (C)"
    )
    answers = parse_answer_blocks(response, 3, mcq_mode=True)
    assert answers == ["(A)", "(B)(D)", "(C)"]


def test_method_two_stage_interface() -> None:
    from ama_bench.method import NovaCodeMemoryMethod

    method = NovaCodeMemoryMethod()
    memory = method.memory_construction(SAMPLE_TRAJECTORY, task="collect")
    context = method.memory_retrieve(memory, "how many apples?")
    assert isinstance(context, str)
    assert context
    assert not method.requires_embedding


def test_method_batch_prompt() -> None:
    from ama_bench.method import NovaCodeMemoryMethod

    method = NovaCodeMemoryMethod()
    memory = method.memory_construction(SAMPLE_TRAJECTORY, task="collect")
    prompt = method.build_prompt(memory, ["How many apples?", "Which file was written?"], mcq_mode=True)
    assert "Question 1:" in prompt
    assert "Answer[1]" in prompt
    assert "Answer[2]" in prompt
    assert "<evidence" in prompt
