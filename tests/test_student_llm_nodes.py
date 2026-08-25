from types import SimpleNamespace

import langgraph_agent_lab.nodes as nodes
from langgraph_agent_lab.schemas import ClassificationDecision, EvaluationDecision


class FakeStructuredLLM:
    def __init__(self, owner: "FakeLLM", payload: object) -> None:
        self.owner = owner
        self.payload = payload

    def invoke(self, prompt: str) -> object:
        self.owner.prompts.append(prompt)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeLLM:
    def __init__(self, structured_payload: object, answer: str = "grounded answer") -> None:
        self.structured_payload = structured_payload
        self.answer = answer
        self.schemas: list[type[object]] = []
        self.prompts: list[str] = []

    def with_structured_output(self, schema: type[object]) -> FakeStructuredLLM:
        self.schemas.append(schema)
        return FakeStructuredLLM(self, self.structured_payload)

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.answer)


def test_classifier_uses_structured_llm_without_scenario_id(monkeypatch) -> None:
    fake = FakeLLM(ClassificationDecision(route="risky", reason="side effect"))
    monkeypatch.setattr(nodes, "get_llm", lambda **_: fake, raising=False)

    update = nodes.classify_node(
        {"query": "Please refund the customer", "scenario_id": "DO-NOT-PROMPT"}
    )

    assert update["route"] == "risky"
    assert update["risk_level"] == "high"
    assert fake.schemas == [ClassificationDecision]
    assert "risky > tool > missing_info > error > simple" in fake.prompts[0]
    assert "Please refund the customer" in fake.prompts[0]
    assert "DO-NOT-PROMPT" not in fake.prompts[0]


def test_classifier_fallback_is_auditable(monkeypatch) -> None:
    fake = FakeLLM(RuntimeError("provider unavailable"))
    monkeypatch.setattr(nodes, "get_llm", lambda **_: fake, raising=False)

    update = nodes.classify_node({"query": "Delete the customer account"})

    assert update["route"] == "risky"
    assert update["errors"]
    assert update["events"][0]["event_type"] == "fallback"


def test_answer_uses_grounded_workflow_context(monkeypatch) -> None:
    fake = FakeLLM(
        ClassificationDecision(route="simple", reason="unused"),
        answer="Order 123 is shipped according to the tool result.",
    )
    monkeypatch.setattr(nodes, "get_llm", lambda **_: fake, raising=False)

    update = nodes.answer_node(
        {
            "query": "Where is order 123?",
            "tool_results": ["SUCCESS: order 123 shipped"],
            "approval": {"approved": True, "reviewer": "mock-reviewer"},
        }
    )

    assert "Order 123" in update["final_answer"]
    prompt = fake.prompts[-1]
    assert "Where is order 123?" in prompt
    assert "SUCCESS: order 123 shipped" in prompt
    assert "approved" in prompt.lower()


def test_llm_judge_uses_one_structured_verdict_with_explicit_budget(monkeypatch) -> None:
    fake = FakeLLM(EvaluationDecision(verdict="success", reason="tool result is usable"))
    captured: dict[str, object] = {}

    def fake_get_llm(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(nodes, "configured_provider", lambda: "openai", raising=False)
    monkeypatch.setattr(nodes, "get_llm", fake_get_llm, raising=False)

    update = nodes.evaluate_node({"tool_results": ["SUCCESS: found order"]})

    assert update["evaluation_result"] == "success"
    assert fake.schemas == [EvaluationDecision]
    assert len(fake.prompts) == 1
    assert captured["timeout"] == 20.0
    assert captured["max_retries"] == 0
    assert update["events"][0]["metadata"]["mode"] == "llm-as-judge"
    assert update["events"][0]["metadata"]["timeout_seconds"] == 20.0
    assert update["events"][0]["metadata"]["max_retries"] == 0
