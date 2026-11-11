"""Find, compile and describe the sources one diagram needs.

A diagram is drawn from the verified source-unit body, never from the Python
text. This module only works out which sources a file imports, compiles them in
import order through the ordinary ``pycircuit compile`` path into a private
cache, and asks ``pycircuit-emit --describe-structure`` what each unit contains.

Sources import each other as ``<package>.<path>``; the package is a build
setting the sources do not declare. The file's own folder is its source root
and its package is the prefix the folder's sources use for one another. An
import that does not resolve there may name a provider folder beside it whose
name is the package (``functional_float.word_arithmetic`` is
``../functional_float/word_arithmetic.py``), as the example builds lay them out.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .._driver import _native_helper, compile_command

CACHE_ENV = "PYCIRCUIT_IDE_CACHE"
#: Package-prefix inference reads at most this many sources of the root.
_SCAN_LIMIT = 200
_NATIVE_DIAGNOSTIC = re.compile(
    r'loc\("(?P<file>[^"]+)":(?P<line>\d+):(?P<column>\d+)\): '
    r"(?P<severity>error|warning): (?P<message>.*)"
)


@dataclass(frozen=True)
class Diagnostic:
    message: str
    file: str | None = None
    line: int | None = None
    column: int | None = None
    severity: str = "error"

    def to_json(self) -> dict[str, object]:
        out: dict[str, object] = {"message": self.message, "severity": self.severity}
        for key in ("file", "line", "column"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


class CompileError(Exception):
    """A source in the closure did not compile; ``diagnostics`` say where."""

    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        super().__init__("; ".join(d.message for d in diagnostics))
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class Source:
    """One Python source, the root it is captured from and its package."""

    root: Path
    package: str
    path: Path

    @property
    def relative(self) -> str:
        return self.path.relative_to(self.root).as_posix()

    @property
    def key(self) -> str:
        return f"{self.package or '_'}__{self.relative.replace('/', '__')}"

    def file(self, relative: str) -> Path:
        return self.root / relative


def _module_file(root: Path, package: str, module: str) -> Path | None:
    parts = module.split(".")
    prefix = package.split(".") if package else []
    if parts[: len(prefix)] != prefix or len(parts) == len(prefix):
        return None
    candidate = root.joinpath(*parts[len(prefix) :]).with_suffix(".py")
    return candidate if candidate.is_file() else None


def _imported_modules(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return []
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.append(node.module)
            modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


def _infer_package(root: Path) -> str:
    """The prefix the root's sources use to import each other (``""`` if none)."""
    votes: Counter[str] = Counter()
    for index, path in enumerate(sorted(root.rglob("*.py"))):
        if index >= _SCAN_LIMIT:
            break
        for module in _imported_modules(path):
            parts = module.split(".")
            for split in range(len(parts)):
                rest = root.joinpath(*parts[split:]).with_suffix(".py")
                if rest.is_file() and rest != path:
                    votes[".".join(parts[:split])] += 1
    return votes.most_common(1)[0][0] if votes else ""


class Project:
    """The file's own root and package, and the provider roots it reaches."""

    def __init__(self, root: Path, package: str) -> None:
        self.root = root
        self.package = package
        self.roots: dict[str, Path] = {package: root}

    def source(self, path: Path) -> Source:
        return Source(self.root, self.package, path)

    def resolve(self, module: str) -> Source | None:
        """The source a module name denotes: in a known root, or a provider."""
        for package, root in self.roots.items():
            path = _module_file(root, package, module)
            if path is not None:
                return Source(root, package, path)
        package = module.split(".", 1)[0]
        provider = self.root.parent / package
        if package in self.roots or not provider.is_dir():
            return None
        path = _module_file(provider, package, module)
        if path is None:
            return None
        self.roots[package] = provider
        return Source(provider, package, path)

    def imports(self, source: Source) -> list[Source]:
        found: list[Source] = []
        for module in _imported_modules(source.path):
            dependency = self.resolve(module)
            if dependency and dependency != source and dependency not in found:
                found.append(dependency)
        return found

    def closure(self, source: Source) -> list[Source]:
        """``source`` and every source it imports, dependencies first."""
        order: list[Source] = []
        active: list[Source] = []

        def visit(current: Source) -> None:
            if current in order:
                return
            if current in active:
                cycle = " -> ".join(s.path.name for s in [*active, current])
                raise CompileError(
                    [Diagnostic(f"import cycle: {cycle}", str(current.path))]
                )
            active.append(current)
            for dependency in self.imports(current):
                visit(dependency)
            active.pop()
            order.append(current)

        visit(source)
        return order

    def owner_source(self, owner: dict[str, str]) -> Source | None:
        """The source a unit owner ``{package, path}`` names, if known."""
        root = self.roots.get(owner.get("package", ""))
        if root is None:
            return None
        path = root / owner.get("path", "")
        return Source(root, owner["package"], path) if path.is_file() else None


def discover(
    file: Path, source_root: Path | None = None, package: str | None = None
) -> Project:
    root = (source_root or file.parent).resolve()
    return Project(root, _infer_package(root) if package is None else package)


# -- Compiled units ---------------------------------------------------------------


def _stamp(path: Path) -> str:
    stat = path.stat()
    return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"


def _cache_base() -> Path:
    configured = os.environ.get(CACHE_ENV)
    if configured:
        return Path(configured)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "pycircuit" / "ide"


def _replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class UnitCache:
    """Published source units for one project, recompiled when their inputs change.

    A unit's key covers its source bytes, its imports' keys and the toolchain
    that compiled it, so an edited import or a rebuilt compiler recompiles
    exactly the units it reaches.
    """

    def __init__(self, project: Project) -> None:
        self.project = project
        identity = f"{project.root}\0{project.package}".encode()
        self.root = (
            _cache_base() / hashlib.sha256(identity).hexdigest()[:16]
        ).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.emitter = _native_helper("emit")
        package_dir = Path(__file__).resolve().parent.parent
        self.toolchain = "\n".join(
            _stamp(path)
            for path in (
                _native_helper("source-unit"),
                package_dir / "_source_capture.py",
                package_dir / "_source_transport.py",
            )
        )
        self.keys: dict[Source, str] = {}

    def unit_dir(self, source: Source) -> Path:
        return self.root / "units" / source.key

    def compile(self, source: Source) -> None:
        """Compile ``source`` and the sources it imports, dependencies first."""
        for current in self.project.closure(source):
            digest = hashlib.sha256(self.toolchain.encode())
            digest.update(current.path.read_bytes())
            for dependency in sorted(self.project.imports(current), key=str):
                digest.update(self.keys[dependency].encode())
            key = self.keys[current] = digest.hexdigest()
            unit, stamp = self.unit_dir(current), self.root / "keys" / current.key
            fresh = stamp.is_file() and stamp.read_text(encoding="utf-8") == key
            if fresh and (unit / "unit.json").is_file():
                continue
            # A unit's interface names its own imports, so the compiler needs
            # the header of every source the closure reaches.
            reached = self.project.closure(current)[:-1]
            self._compile_one(current, unit, [self.unit_dir(s) for s in reached])
            _replace_text(stamp, key)

    def _compile_one(self, source: Source, unit: Path, interfaces: list[Path]) -> None:
        unit.parent.mkdir(parents=True, exist_ok=True)
        try:
            compile_command(
                source=source.path,
                source_root=source.root,
                output=unit,
                package_prefix=source.package,
                interface_units=interfaces,
                replace=unit.exists(),
            )
        except SyntaxError as error:
            where = source.file(Path(error.filename or source.relative).name)
            diagnostic = Diagnostic(
                f"invalid syntax: {error.msg}", str(where), error.lineno, error.offset
            )
            raise CompileError([diagnostic]) from error
        except Exception as error:  # noqa: BLE001 - every failure is a diagnostic
            raise CompileError(_diagnostics(str(error), source)) from error

    def describe(self, source: Source) -> dict[str, object]:
        """The structure of ``source``'s compiled unit (after :meth:`compile`)."""
        key = f"{self.keys[source]}:{_stamp(self.emitter)}"
        cached = self.root / "structure" / f"{source.key}.json"
        if cached.is_file():
            stored = json.loads(cached.read_text(encoding="utf-8"))
            if stored.get("key") == key:
                return stored["structure"]
        unit = self.unit_dir(source)
        receipt = json.loads((unit / "unit.json").read_text(encoding="utf-8"))
        completed = subprocess.run(
            [
                str(self.emitter),
                "--describe-structure",
                str(unit / receipt["files"]["body"]),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or "structure description failed"
            raise CompileError(_diagnostics(message, source))
        structure = json.loads(completed.stdout)
        _replace_text(cached, json.dumps({"key": key, "structure": structure}))
        return structure


def _diagnostics(text: str, source: Source) -> list[Diagnostic]:
    found = [
        Diagnostic(
            match["message"],
            str(source.file(match["file"])),
            int(match["line"]),
            int(match["column"]),
            match["severity"],
        )
        for match in _NATIVE_DIAGNOSTIC.finditer(text)
    ]
    return found or [Diagnostic(text, str(source.path))]


__all__ = [
    "CompileError",
    "Diagnostic",
    "Project",
    "Source",
    "UnitCache",
    "discover",
]
