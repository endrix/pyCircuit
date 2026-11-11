"""Source structure -> Dialogram graph documents.

The document is the shape Dialogram's ``cli-plan`` acquisition reads from
``pycircuit ide plan --format graph``: ``{"version", "graph": {"id", "nodes",
"edges", "subgraphs", "meta"}, "partial"?, "errors"?}``, with ``--hierarchy``
adding the ``hierarchy`` outline of nested instance graphs.

One module definition is one graph:

* its data inputs and outputs are ``wf-input`` / ``wf-output`` boundary nodes;
  the clock and reset the compiler adds to every module are not drawn;
* a child instance is a node. Its family comes from the callee's declared
  primitive kind: a FIFO is a ``queue``, a DFF a ``register``, a memory a
  ``memory``; an ordinary module is an ``instance`` the diagram drills into.
  A standard leaf's constant inputs are the source stage's placeholders for
  the connections Rule lowering makes, and are not drawn;
* a module-scope state, a Rule, a Method that no Rule binds, and a module
  reference parameter are nodes of their own families;
* value wires are ``data`` edges from the producing endpoint (an input, an
  instance output, a state, a constant or an expression ``logic`` node);
* each push, pop, Method call, state read and state write of a Rule or Method
  is an ``access`` edge, drawn the way its data moves: towards the target when
  the call passes arguments (or returns nothing), away from it otherwise;
* a reference an instantiation passes (``Writer(counter)``) is a ``reference``
  edge from the instance it names to the receiving instance.

This is read-only navigation data derived from the compiler's description of
the verified unit; it decides no semantics.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from .project import CompileError, Diagnostic, Project, Source, UnitCache

DOC_VERSION = "1.1"
ROOT_SCOPE = "scope:root"
#: Deeper than any real hierarchy; a guard against a definition nesting itself.
MAX_DEPTH = 32
DEFAULT_HIERARCHY_BUDGET = 500
#: The compiler names instances the source leaves anonymous with this
#: reserved prefix (``__pyc_call_0``); the diagram shows the callee instead.
GENERATED_PREFIX = "__pyc_"


def short_name(symbol: str) -> str:
    return symbol.rsplit(".", 1)[-1]


def family_of(primitive: str | None) -> str:
    if primitive is None:
        return "instance"
    if primitive.startswith("fifo"):
        return "queue"
    if primitive.startswith("dff"):
        return "register"
    if "mem" in primitive:
        return "memory"
    return "primitive"


def node_id(name: str) -> str:
    return f"node:{ROOT_SCOPE}:{name}"


class Design:
    """The module definitions a file's import closure describes, by symbol."""

    def __init__(self, project: Project, cache: UnitCache) -> None:
        self.project = project
        self.cache = cache
        self.modules: dict[str, tuple[dict[str, Any], Source]] = {}
        self.owners: dict[str, dict[str, str]] = {}
        self.diagnostics: list[Diagnostic] = []

    def load(self, source: Source) -> list[dict[str, Any]]:
        """Compile ``source`` and its imports; return the modules it defines."""
        self.cache.compile(source)
        return self._describe(source)

    def _describe(self, source: Source) -> list[dict[str, Any]]:
        structure = self.cache.describe(source)
        for module in structure["modules"]:
            self.modules[module["symbol"]] = (module, source)
        for entry in structure["imports"]:
            if entry.get("source"):
                self.owners.setdefault(entry["symbol"], entry["source"])
        return list(structure["modules"])

    def definition(self, symbol: str) -> tuple[dict[str, Any], Source] | None:
        if symbol not in self.modules and symbol in self.owners:
            source = self.project.owner_source(self.owners[symbol])
            if source is not None and source in self.cache.keys:
                try:
                    self._describe(source)
                except CompileError as failure:
                    self.diagnostics.extend(failure.diagnostics)
        return self.modules.get(symbol)

    @staticmethod
    def location(loc: dict[str, Any] | None, source: Source) -> dict[str, Any] | None:
        if not loc:
            return None
        path = source.file(loc["path"])
        if not path.is_file():
            return None
        return {"file": str(path), "line": loc["line"], "column": loc["column"]}


class _Graph:
    """One graph under construction: nodes with their ports, and edges."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.ports: dict[str, dict[str, dict[str, Any]]] = {}
        self.owner: dict[str, str] = {}
        self.edges: dict[str, dict[str, Any]] = {}

    def node(self, ident: str, kind: str, label: str, **fields: Any) -> str:
        if ident not in self.nodes:
            node = {"id": ident, "kind": kind, "label": label, "scope": ROOT_SCOPE}
            node.update({k: v for k, v in fields.items() if v is not None})
            self.nodes[ident] = node
            self.ports[ident] = {}
        return ident

    def port(
        self,
        node: str,
        key: str,
        direction: str,
        *,
        name: str | None = None,
        value_type: str | None = None,
        role: str = "data",
    ) -> str:
        ident = f"port:{node}:{key}"
        if ident not in self.ports[node]:
            port: dict[str, Any] = {
                "id": ident,
                "name": name or key,
                "direction": direction,
                "role": role,
            }
            if value_type:
                port["type"] = value_type
            self.ports[node][ident] = port
            self.owner[ident] = node
        return ident

    def edge(self, source: str, target: str, role: str, **meta: Any) -> None:
        ident = f"edge:{source}->{target}"
        if ident in self.edges:
            return
        self.edges[ident] = {
            "id": ident,
            "from": source,
            "to": target,
            "fromNode": self.owner[source],
            "toNode": self.owner[target],
            "scope": ROOT_SCOPE,
            "meta": {"role": role, **{k: v for k, v in meta.items() if v}},
        }

    def finish(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nodes = []
        for ident, node in self.nodes.items():
            nodes.append({**node, "ports": list(self.ports[ident].values())})
        return nodes, list(self.edges.values())


class GraphBuilder:
    def __init__(self, design: Design) -> None:
        self.design = design

    def document(self, symbol: str) -> dict[str, Any]:
        module, source = self.design.modules[symbol]
        return _ModuleGraph(self.design, module, source).document()

    def drill_down(self, symbol: str) -> list[tuple[str, str]]:
        """``(instance name, callee symbol)`` for each child with a graph."""
        module, _ = self.design.modules[symbol]
        children = []
        for instance in module["instances"]:
            callee = instance["callee"]
            if instance["primitive"] is None and self.design.definition(callee):
                children.append((instance["name"], callee))
        return children

    def hierarchy(
        self, symbol: str, budget: int = DEFAULT_HIERARCHY_BUDGET
    ) -> dict[str, Any]:
        """The root's document plus ``hierarchy``, the ``{graph, hierarchy}``
        envelope Dialogram's hierarchy cache reads: every nested instance with
        its own document, breadth first under ``budget``. An instance past the
        budget, past :data:`MAX_DEPTH` or of a definition on its own lineage is
        listed ``truncated`` without a graph."""
        root = self._entry(symbol, [], None)
        graphs = 1
        queue = deque([(symbol, root, (symbol,))])
        while queue:
            parent, entry, lineage = queue.popleft()
            for name, callee in self.drill_down(parent):
                path = [*entry["path"], name]
                if len(path) > MAX_DEPTH or callee in lineage or graphs >= budget:
                    entry["children"].append(
                        {
                            "path": path,
                            "workflowName": short_name(callee),
                            "sourcePath": str(self.design.modules[callee][1].path),
                            "nodeId": node_id(name),
                            "children": [],
                            "truncated": True,
                        }
                    )
                    continue
                child = self._entry(callee, path, node_id(name))
                graphs += 1
                entry["children"].append(child)
                queue.append((callee, child, (*lineage, callee)))
        exported = root.pop("graph")
        return {**exported, "hierarchy": root}

    def _entry(self, symbol: str, path: list[str], ident: str | None) -> dict[str, Any]:
        doc = self.document(symbol)
        entry: dict[str, Any] = {
            "path": path,
            "workflowName": short_name(symbol),
            "sourcePath": str(self.design.modules[symbol][1].path),
            "children": [],
            "graph": doc,
            "nodeCount": sum(
                1
                for n in doc["graph"]["nodes"]
                if n["kind"] not in ("wf-input", "wf-output")
            ),
        }
        if ident:
            entry["nodeId"] = ident
        return entry


class _ModuleGraph:
    def __init__(self, design: Design, module: dict[str, Any], source: Source) -> None:
        self.design = design
        self.module = module
        self.source = source
        self.graph = _Graph()
        self.logic = {entry["id"]: entry for entry in module["logic"]}
        self.constants = 0
        bound = {rule["method"]: rule["name"] for rule in module["rules"]}
        self.actor_of = {name: rule for name, rule in bound.items() if name}
        self.actor_of.update({rule["name"]: rule["name"] for rule in module["rules"]})
        self.used_outputs = self._used_outputs()

    def document(self) -> dict[str, Any]:
        m = self.module
        self._boundary()
        for reference in m["references"]:
            self._reference(reference)
        for instance in m["instances"]:
            self._instance_node(instance)
        for state in m["states"]:
            self.graph.node(
                node_id(state["name"]),
                "state",
                state["name"],
                type=state["type"],
                meta=self._meta("state", state.get("location")),
            )
        for rule in m["rules"]:
            meta = self._meta("rule", rule.get("location"))
            if rule.get("method") and rule["method"] != rule["name"]:
                meta["method"] = rule["method"]
            self.graph.node(node_id(rule["name"]), "rule", rule["name"], meta=meta)
        for method in m["methods"]:
            if method["name"] not in self.actor_of:
                self.graph.node(
                    node_id(method["name"]),
                    "method",
                    method["name"],
                    type=_signature(method),
                    meta=self._meta("method", method.get("location")),
                )
        for instance in m["instances"]:
            self._instance_wires(instance)
        methods = {method["name"]: method for method in m["methods"]}
        for rule in m["rules"]:
            bound = methods.get(rule.get("method") or "", {})
            captures = [*rule["captures"], *bound.get("captures", [])]
            self._captures(node_id(rule["name"]), captures)
        for method in m["methods"]:
            if method["name"] not in self.actor_of:
                self._captures(node_id(method["name"]), method["captures"])
        for result in m["results"]:
            source = self._endpoint(result["from"])
            if source:
                self.graph.edge(
                    source,
                    self.graph.port(f"wf:output:{result['port']}", "in", "in"),
                    "data",
                )
        for access in m["accesses"]:
            self._access(access)

        nodes, edges = self.graph.finish()
        meta: dict[str, Any] = {
            "file": str(self.source.path),
            "unitKind": m["root_kind"],
            "symbol": m["symbol"],
        }
        where = self.design.location(m.get("location"), self.source)
        if where:
            meta["source"] = where
        if m["parameters"]:
            meta["parameters"] = [{"name": name} for name in m["parameters"]]
        graph = {
            "id": f"wf:{m['name']}",
            "nodes": nodes,
            "edges": edges,
            "subgraphs": [
                {
                    "id": ROOT_SCOPE,
                    "kind": "root",
                    "parent": None,
                    "controlNodeId": None,
                    "children": [],
                    "nodes": [n["id"] for n in nodes],
                    "edges": [e["id"] for e in edges],
                }
            ],
            "meta": meta,
        }
        doc: dict[str, Any] = {"version": DOC_VERSION, "graph": graph}
        if self.design.diagnostics:
            doc["partial"] = True
            doc["errors"] = [d.to_json() for d in self.design.diagnostics]
        return doc

    # -- nodes ------------------------------------------------------------------

    def _meta(self, family: str, loc: dict[str, Any] | None) -> dict[str, Any]:
        meta: dict[str, Any] = {"definitionAnnotations": [{"name": family}]}
        where = self.design.location(loc, self.source)
        if where:
            meta["source"] = where
        return meta

    def _boundary(self) -> None:
        for port in self.module["inputs"]:
            if port.get("domain"):
                continue
            ident = self.graph.node(
                f"wf:input:{port['name']}", "wf-input", port["name"]
            )
            self.graph.port(
                ident, "out", "out", name=port["name"], value_type=port["type"]
            )
        for port in self.module["outputs"]:
            ident = self.graph.node(
                f"wf:output:{port['name']}", "wf-output", port["name"]
            )
            self.graph.port(
                ident, "in", "in", name=port["name"], value_type=port["type"]
            )

    def _reference(self, reference: dict[str, Any]) -> None:
        meta = self._meta("reference", self.module.get("location"))
        callee = reference.get("callee")
        if callee:
            meta["referencedEntityName"] = short_name(callee)
            self._referenced_source(meta, callee)
        self.graph.node(
            node_id(reference["name"]),
            "reference",
            reference["name"],
            type=short_name(callee) if callee else None,
            meta=meta,
        )

    def _referenced_source(self, meta: dict[str, Any], callee: str) -> None:
        found = self.design.definition(callee)
        if found:
            where = self.design.location(found[0].get("location"), found[1])
            if where:
                meta["referencedSource"] = where

    def _instance_node(self, instance: dict[str, Any]) -> None:
        family = family_of(instance["primitive"])
        name, callee = instance["name"], instance["callee"]
        meta = self._meta(family, instance.get("location"))
        meta["referencedEntityName"] = short_name(callee)
        if instance["parameters"]:
            meta["parameters"] = instance["parameters"]
        if instance["primitive"] is not None:
            meta["primitive"] = instance["primitive"]
        elif self.design.definition(callee):
            meta["networkInstance"] = True
            self._referenced_source(meta, callee)
        else:
            meta["external"] = True
        label = short_name(callee) if name.startswith(GENERATED_PREFIX) else name
        self.graph.node(
            node_id(name), "instance", label, type=short_name(callee), meta=meta
        )

    def _instance_wires(self, instance: dict[str, Any]) -> None:
        ident = node_id(instance["name"])
        leaf = instance["primitive"] is not None
        for port in instance["inputs"]:
            if port.get("domain"):
                continue
            endpoint = port.get("from")
            if "constant" in port or (leaf and self._is_constant(endpoint)):
                if leaf:
                    continue
                value = port.get("constant") or self._constant_text(endpoint)
                endpoint = {"kind": "constant", "value": value}
            target = self.graph.port(ident, port["port"], "in", value_type=port["type"])
            source = self._endpoint(endpoint)
            if source:
                self.graph.edge(source, target, "data")
        for port in instance["outputs"]:
            if not leaf or (instance["name"], port["port"]) in self.used_outputs:
                self.graph.port(ident, port["port"], "out", value_type=port["type"])
        for binding in instance["references"]:
            target = self.graph.port(ident, binding["name"], "in", role="reference")
            receiver = binding["receiver"]
            if receiver["kind"] == "reference":
                holder, rest = receiver["name"], receiver["path"]
            elif receiver["path"]:
                holder, rest = receiver["path"][0], receiver["path"][1:]
            else:
                continue
            if node_id(holder) not in self.graph.nodes:
                continue
            source = self.graph.port(node_id(holder), "self", "out", role="reference")
            self.graph.edge(
                source,
                target,
                "reference",
                label=".".join(rest) if rest else None,
                unresolved=receiver.get("unresolved"),
            )

    # -- wires ------------------------------------------------------------------

    def _used_outputs(self) -> set[tuple[str, str]]:
        used: set[tuple[str, str]] = set()

        def note(endpoint: dict[str, Any] | None) -> None:
            if endpoint and endpoint.get("kind") == "instance":
                used.add((endpoint["name"], endpoint["port"]))

        for instance in self.module["instances"]:
            for port in instance["inputs"]:
                note(port.get("from"))
        for result in self.module["results"]:
            note(result["from"])
        for entry in self.module["logic"]:
            for endpoint in entry["inputs"]:
                note(endpoint)
        return used

    def _captures(self, actor: str, captures: list[dict[str, Any]]) -> None:
        """An input per enclosing value the body reads, named after it."""
        for endpoint in captures:
            source = self._endpoint(endpoint)
            if not source:
                continue
            name = self._describe(endpoint)
            target = self.graph.port(actor, f"capture:{name}", "in", name=name)
            self.graph.edge(source, target, "data")

    def _describe(self, endpoint: dict[str, Any]) -> str:
        kind = endpoint["kind"]
        if kind in ("input", "state"):
            return endpoint["name"]
        if kind in ("instance", "rule"):
            holder = self.graph.nodes[node_id(endpoint["name"])]["label"]
            return f"{holder}.{endpoint['port']}"
        if kind == "logic":
            return self.logic[endpoint["id"]]["operation"]
        return str(endpoint.get("value", kind))

    def _is_constant(self, endpoint: dict[str, Any] | None) -> bool:
        return bool(
            endpoint
            and endpoint.get("kind") == "logic"
            and not self.logic[endpoint["id"]]["inputs"]
        )

    def _constant_text(self, endpoint: dict[str, Any] | None) -> str:
        return self.logic[endpoint["id"]]["operation"] if endpoint else "?"

    def _endpoint(self, endpoint: dict[str, Any] | None) -> str | None:
        """The port that drives a wire from ``endpoint``, creating it if needed."""
        if not endpoint:
            return None
        kind = endpoint.get("kind")
        if kind == "input":
            boundary = f"wf:input:{endpoint['name']}"
            return f"port:{boundary}:out" if boundary in self.graph.nodes else None
        if kind in ("instance", "state", "rule"):
            holder = node_id(endpoint["name"])
            if holder not in self.graph.nodes:
                return None
            port = "read" if kind == "state" else endpoint["port"]
            return self.graph.port(holder, port, "out")
        if kind == "constant":
            self.constants += 1
            ident = self.graph.node(
                node_id(f"constant{self.constants}"),
                "constant",
                endpoint["value"],
                meta={"definitionAnnotations": [{"name": "logic"}]},
            )
            return self.graph.port(ident, "out", "out")
        if kind == "logic":
            return self._logic_node(endpoint["id"])
        return None

    def _logic_node(self, logic_id: str) -> str:
        ident = node_id(logic_id)
        if ident not in self.graph.nodes:
            entry = self.logic[logic_id]
            self.graph.node(
                ident,
                "logic",
                entry["operation"],
                meta=self._meta("logic", entry.get("location")),
            )
            for index, endpoint in enumerate(entry["inputs"]):
                source = self._endpoint(endpoint)
                if source:
                    self.graph.edge(
                        source, self.graph.port(ident, f"in{index}", "in"), "data"
                    )
        return self.graph.port(ident, "out", "out")

    # -- accesses ---------------------------------------------------------------

    def _access(self, access: dict[str, Any]) -> None:
        actor_name = self.actor_of.get(access["actor"], access["actor"])
        actor = node_id(actor_name)
        if actor not in self.graph.nodes:
            return
        operation, target = access["operation"], access["target"]
        if target is None:
            meta = self.graph.nodes[actor].setdefault("meta", {})
            observes = meta.setdefault("observes", [])
            if operation not in observes:
                observes.append(operation)
            return
        if target["kind"] == "state":
            holder, method, label = target["name"], operation, target["name"]
            inward = operation == "write"
        else:
            if target["kind"] == "instance" and target["path"]:
                holder, rest = target["path"][0], target["path"][1:]
            elif target["kind"] == "reference":
                holder, rest = target["name"], target["path"]
            else:  # a Method of this module
                holder = self.actor_of.get(target["method"], target["method"])
                rest = []
            method = ".".join([*rest, target["method"]])
            label = f"{holder}.{method}" if target["kind"] != "self" else method
            inward = target["arguments"] > 0 or target["results"] == 0
        holder_id = node_id(holder)
        if holder_id not in self.graph.nodes:
            return
        if target["kind"] == "self":
            method = "call"
        there = self.graph.port(
            holder_id, method, "in" if inward else "out", role="access"
        )
        here = self.graph.port(
            actor,
            f"{label}:{'out' if inward else 'in'}",
            "out" if inward else "in",
            name=label,
            role="access",
        )
        if inward:
            self.graph.edge(here, there, "access", operation=operation)
        else:
            self.graph.edge(there, here, "access", operation=operation)


def _signature(method: dict[str, Any]) -> str:
    arguments = ", ".join(f"{a['name']}: {a['type']}" for a in method["arguments"])
    results = [r["type"] for r in method["results"]]
    returns = (
        ""
        if not results
        else f" -> {results[0]}" if len(results) == 1 else f" -> ({', '.join(results)})"
    )
    return f"({arguments}){returns}"


def failure_document(
    unit_name: str, file: str, diagnostics: list[Diagnostic]
) -> dict[str, Any]:
    """An empty partial document for a file whose closure did not compile."""
    return {
        "version": DOC_VERSION,
        "partial": True,
        "errors": [d.to_json() for d in diagnostics]
        or [{"message": "compilation failed", "file": file}],
        "graph": {
            "id": f"wf:{unit_name}",
            "nodes": [],
            "edges": [],
            "subgraphs": [
                {
                    "id": ROOT_SCOPE,
                    "kind": "root",
                    "parent": None,
                    "controlNodeId": None,
                    "children": [],
                    "nodes": [],
                    "edges": [],
                }
            ],
            "meta": {"file": file},
        },
    }


def pick_module(
    modules: list[dict[str, Any]], file: Path, requested: str | None
) -> dict[str, Any]:
    """The module a request is about.

    ``requested`` (a short name or a symbol) wins. Otherwise the file's top: the
    module no other module in the file instantiates, a System first, then the
    one named like the file, then the first. A :class:`LookupError` carries
    the candidates.
    """
    names = [m["name"] for m in modules]
    if requested:
        for module in modules:
            if requested in (module["name"], module["symbol"]):
                return module
        raise LookupError(f"no module {requested!r} in {file.name}", names)
    if not modules:
        raise LookupError(f"{file.name} defines no module", names)
    used = {i["callee"] for m in modules for i in m["instances"]}
    tops = [m for m in modules if m["symbol"] not in used] or modules
    systems = [m for m in tops if m["root_kind"] == "system"]
    if len(systems) == 1:
        return systems[0]
    stem = file.stem.replace("_", "").lower()
    for module in tops:
        if module["name"].lower() == stem:
            return module
    return tops[0]


__all__ = ["Design", "GraphBuilder", "failure_document", "pick_module"]
