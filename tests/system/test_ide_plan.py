"""``pycircuit ide plan`` draws a source file from its verified unit."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from pycircuit.cli import main

pytestmark = pytest.mark.system

REPO = Path(__file__).resolve().parents[2]
_HELPERS = {
    "PYCIRCUIT_SOURCE_COMPILER": "pycircuit-source-unit",
    "PYCIRCUIT_LINKER": "pycircuit-link",
    "PYCIRCUIT_EMITTER": "pycircuit-emit",
}


@pytest.fixture(autouse=True)
def _toolchain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = os.environ.get("PYC_TOOLCHAIN_ROOT")
    for variable, name in _HELPERS.items():
        if os.environ.get(variable):
            continue
        candidate = Path(root) / "bin" / name if root else None
        found = str(candidate) if candidate and candidate.is_file() else None
        found = found or shutil.which(name)
        if not found:
            raise AssertionError(f"set PYC_TOOLCHAIN_ROOT or {variable}")
        monkeypatch.setenv(variable, found)
    monkeypatch.setenv("PYCIRCUIT_IDE_CACHE", str(tmp_path / "cache"))


def _plan(tmp_path: Path, source: Path, *extra: str) -> dict:
    output = tmp_path / "plan.json"
    assert main(["ide", "plan", str(source), "-o", str(output), *extra]) == 0
    return json.loads(output.read_text(encoding="utf-8"))


def _families(graph: dict) -> dict[str, str]:
    return {
        node["label"]: node["meta"]["definitionAnnotations"][0]["name"]
        for node in graph["nodes"]
        if node.get("meta", {}).get("definitionAnnotations")
    }


def _links(graph: dict, role: str) -> set[tuple[str, str]]:
    labels = {node["id"]: node["label"] for node in graph["nodes"]}
    ports = {
        port["id"]: (labels[node["id"]], port["name"])
        for node in graph["nodes"]
        for port in node["ports"]
    }
    return {
        (".".join(ports[edge["from"]]), ".".join(ports[edge["to"]]))
        for edge in graph["edges"]
        if edge["meta"]["role"] == role
    }


def test_rules_queues_and_state_accesses(tmp_path: Path) -> None:
    source = REPO / "examples/queue_rule_pipeline/queue_rule_pipeline.py"
    doc = _plan(tmp_path, source, "--hierarchy")
    graph = doc["graph"]

    assert "errors" not in doc
    assert graph["id"] == "wf:QueueRulePipeline"
    assert _families(graph) == {
        "ingress": "queue",
        "egress": "queue",
        "accumulator": "instance",
        "transferred": "state",
        "transfer": "rule",
        "progress": "rule",
        "put": "method",
        "get": "method",
        "snapshot": "method",
    }
    # Each access is drawn the way its data moves.
    assert {
        ("ingress.pop", "transfer.ingress.pop"),
        ("transfer.accumulator.add", "accumulator.add"),
        ("transfer.egress.push", "egress.push"),
        ("transferred.read", "transfer.transferred"),
        ("transfer.transferred", "transferred.write"),
        ("put.ingress.push", "ingress.push"),
        ("egress.pop", "get.egress.pop"),
    } <= _links(graph, "access")
    # The compiler-owned clock and reset are not drawn.
    names = {port["name"] for node in graph["nodes"] for port in node["ports"]}
    assert not names & {"pyc_clk", "pyc_rst", "clk", "rst"}
    # The child module is a drill-down with its own graph.
    (child,) = doc["hierarchy"]["children"]
    assert (child["path"], child["workflowName"]) == (["accumulator"], "Accumulator")
    assert _families(child["graph"]["graph"]) == {
        "total": "state",
        "add": "method",
        "snapshot": "method",
    }


def test_shared_reference_across_source_units(tmp_path: Path) -> None:
    folder = REPO / "examples/sibling_calls"
    system = _plan(tmp_path, folder / "sibling_calls.py")["graph"]
    assert _links(system, "reference") == {
        ("counter.self", "Writer.counter"),
        ("counter.self", "Reader.counter"),
    }

    writer = _plan(tmp_path, folder / "clients.py", "--workflow", "Writer")["graph"]
    assert _families(writer) == {"counter": "reference", "write": "rule"}
    assert _links(writer, "access") == {
        ("write.counter.increment", "counter.increment")
    }


def test_compile_error_is_a_located_diagnostic(tmp_path: Path) -> None:
    source = tmp_path / "src" / "broken.py"
    source.parent.mkdir()
    source.write_text(
        "import pycircuit as ac\n\n\n"
        "@ac.module\n"
        "def Broken(a: ac.u8) -> ac.u8:  # noqa: N802\n"
        "    return a + missing_name\n",
        encoding="utf-8",
    )
    doc = _plan(tmp_path, source, "--best-effort")

    assert doc["partial"] is True and doc["graph"]["nodes"] == []
    (error,) = doc["errors"]
    assert (error["file"], error["line"]) == (str(source), 6)
    assert "missing_name" in error["message"]
