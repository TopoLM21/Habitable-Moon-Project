from types import SimpleNamespace

import pytest

from analysis.profile_cpu_stages import Budget, instrument_main


def test_budget_restores_nested_scopes_and_exceptions():
    budget = Budget()
    with budget.scope("physics/other"):
        wrapped = budget.wrap(lambda: 17, "physics/kernel", physics_only=True)
        assert wrapped() == 17
        assert budget.current == "physics/other"
        with pytest.raises(ValueError), budget.scope("physics/failure"):
            raise ValueError("test")
        assert budget.current == "physics/other"
    assert budget.current == "imports_and_instrumentation"
    assert budget.calls["physics/kernel"] == 1
    wrapped()
    assert budget.calls["physics/kernel"] == 1  # gated outside stepping


def test_ast_scopes_preserve_original_statements_and_order(monkeypatch):
    import analysis.profile_cpu_stages as probe

    source = """def main():
    items = [1, 2, 3]
    total = 7
    output.append('setup')
    for dti in items:
        total += dti
        output.append(total)
    output.append('tail')
    return total
"""
    namespace = {"output": []}
    exec(source, namespace)
    base = SimpleNamespace(**namespace)
    monkeypatch.setattr(probe.inspect, "getsource", lambda _: source)
    budget = Budget()
    instrument_main(base, budget)
    assert base.main() == 13
    assert base.output == ["setup", 8, 10, 13, "tail"]
    assert budget.calls == {"initialization/other": 1, "physics/other": 1, "checkpoint_and_output": 1}
