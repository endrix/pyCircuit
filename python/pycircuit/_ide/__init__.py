"""``pycircuit ide plan``: the graph document an IDE draws a source file from.

The editor (the Dialogram ``ac-ide`` extension) runs this command and reads one
JSON graph document from stdout. The file and its local imports are compiled
through the ordinary source-unit path into a private cache
(``$PYCIRCUIT_IDE_CACHE``, default ``~/.cache/pycircuit/ide``) and the verified
units are described by ``pycircuit-emit --describe-structure``; see
:mod:`pycircuit._ide.graph` for what the document draws.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .graph import (
    DEFAULT_HIERARCHY_BUDGET,
    Design,
    GraphBuilder,
    failure_document,
    pick_module,
)
from .project import CompileError, Diagnostic, UnitCache, discover


def plan_command(
    *,
    file: str,
    workflow: str | None = None,
    hierarchy: bool = False,
    hierarchy_budget: int = DEFAULT_HIERARCHY_BUDGET,
    best_effort: bool = False,
    source_root: str | None = None,
    package_prefix: str | None = None,
    output: str | None = None,
) -> int:
    source = Path(file).resolve()
    project = discover(
        source, Path(source_root) if source_root else None, package_prefix
    )
    try:
        cache = UnitCache(project)
        design = Design(project, cache)
        module = pick_module(design.load(project.source(source)), source, workflow)
    except CompileError as failure:
        return _fail(source, workflow, failure.diagnostics, best_effort, output)
    except LookupError as error:
        message, candidates = error.args[0], (
            error.args[1] if len(error.args) > 1 else []
        )
        if best_effort and not candidates:
            return _fail(
                source, workflow, [Diagnostic(message, str(source))], True, output
            )
        hint = f" (candidates: {', '.join(candidates)})" if candidates else ""
        sys.stderr.write(f"pycircuit ide plan: {message}{hint}\n")
        return 1
    except Exception as error:  # noqa: BLE001 - a missing toolchain, a bad cache
        if not best_effort:
            raise
        diagnostic = Diagnostic(f"{type(error).__name__}: {error}", str(source))
        return _fail(source, workflow, [diagnostic], True, output)
    builder = GraphBuilder(design)
    symbol = module["symbol"]
    doc = (
        builder.hierarchy(symbol, hierarchy_budget)
        if hierarchy
        else builder.document(symbol)
    )
    _write(doc, output)
    return 0


def _fail(
    source: Path,
    workflow: str | None,
    diagnostics: list[Diagnostic],
    best_effort: bool,
    output: str | None,
) -> int:
    if not best_effort:
        for diagnostic in diagnostics:
            where = ":".join(
                str(part)
                for part in (diagnostic.file, diagnostic.line, diagnostic.column)
                if part is not None
            )
            sys.stderr.write(f"{where}: {diagnostic.severity}: {diagnostic.message}\n")
        return 1
    _write(failure_document(workflow or source.stem, str(source), diagnostics), output)
    return 0


def _write(doc: dict[str, object], output: str | None) -> None:
    text = json.dumps(doc, indent=2 if sys.stdout.isatty() else None)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")


__all__ = ["plan_command"]
