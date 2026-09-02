import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import execution_policy as policy


def test_normal_and_invalid_policy_do_not_change_current_process():
    before = policy.read_process_priority()
    assert policy.apply_process_priority("normal") == before
    with pytest.raises(ValueError, match="priority"):
        policy.apply_process_priority("high")
    assert policy.read_process_priority() == before


def test_lower_priority_is_real_and_idempotent_in_disposable_process():
    before = policy.read_process_priority()
    code = (
        "import json; from execution_policy import apply_process_priority; "
        "a=apply_process_priority('below_normal'); "
        "b=apply_process_priority('below_normal'); print(json.dumps([a,b]))"
    )
    output = subprocess.check_output([sys.executable, "-c", code],
                                     cwd=Path(__file__).resolve().parent.parent, text=True, timeout=15)
    first, second = json.loads(output)
    assert first == second
    assert policy.is_lower_priority(first)
    assert policy.read_process_priority() == before


@pytest.mark.parametrize("initial,expected", [(0, 5), (5, 5), (12, 12)])
def test_unix_policy_never_raises_inherited_priority(monkeypatch, initial, expected):
    value = {"nice": initial}
    def setpriority(kind, who, nice):
        assert (kind, who) == (0, 0)
        value["nice"] = nice
    fake = SimpleNamespace(name="posix", PRIO_PROCESS=0,
        getpriority=lambda kind, who: value["nice"], setpriority=setpriority)
    monkeypatch.setattr(policy, "os", fake)
    assert policy.apply_process_priority("below_normal")["value"] == expected
    assert policy.apply_process_priority("below_normal")["value"] == expected


def test_unsupported_lower_priority_is_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(policy, "os", SimpleNamespace(name="unknown"))
    with pytest.raises(RuntimeError, match="not supported"):
        policy.apply_process_priority("below_normal")
