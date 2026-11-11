# Editor diagrams

`pycircuit ide plan` prints the graph document an editor draws a source file
from. The Dialogram `ac-ide` extension, which lives in its own repository, runs
it for each diagram it opens. Nothing in the document is read from the Python
text: the file and the local sources it imports are compiled through the
ordinary [source-unit workflow](source-unit-workflow.md), and the verified
units are described by the compiler.

```sh
pycircuit ide plan examples/queue_rule_pipeline/queue_rule_pipeline.py \
  --format graph --hierarchy --best-effort
```

The command draws the file's top module: the one no other module in the file
instantiates, a System first. `--workflow NAME` selects another module.
`--hierarchy` adds the `hierarchy` outline, with one graph for each nested
instance of an ordinary module. `--best-effort` turns a compile failure into an
empty `partial` document. Its `errors` keep the compiler's file, line and
column.

## Finding the sources

The file's folder is the source root, or `--source-root` names another one.
Sources import each other as `<package>.<path>`. The package prefix is a build
setting that the sources do not declare, so the command infers it from how the
root's sources import one another. `--package-prefix` overrides the inferred
prefix. An import that does not resolve in the root may name a provider folder
beside it whose name is the package, as the example builds lay them out:
`functional_float.word_arithmetic` resolves to
`../functional_float/word_arithmetic.py`.

Compiled units are kept in `$PYCIRCUIT_IDE_CACHE`, which defaults to
`~/.cache/pycircuit/ide`. A unit is recompiled when its source changes, when a
source it imports changes, or when the compiler is rebuilt. The first plan of
a large closure therefore compiles the whole closure, and later plans reuse
the cached units.

## What the document draws

`pycircuit-emit --describe-structure UNIT.ac` describes one verified source
unit body in a `pycircuit-source-structure` JSON document. The description
covers module ports, child instances and the wires between them, state owners,
Methods, Rules and the effects each one names. Each graph shows one module
definition:

- Data inputs and outputs are boundary nodes. The compiler-owned clock and
  reset are not drawn.
- Each child instance is a node. A standard leaf's declared primitive kind
  selects its family: a FIFO is a `queue`, a DFF a `register` and a memory a
  `memory`. An ordinary module is an `instance` that the diagram opens. A
  leaf's constant inputs are not drawn: in a source unit they hold the
  places of the connections that Rule lowering makes.
- Module-scope state, Rules, Methods that no Rule binds, and module reference
  parameters are nodes of their own.
- `data` edges carry values between nodes. An expression that feeds an
  instance is drawn as one `logic` node.
- `access` edges connect a Rule or Method to the target of each push, pop,
  Method call, state read and state write. Each edge points the way the data
  moves.
- `reference` edges run from an existing instance to every module that it is
  passed to.

The description and the document are read-only navigation data. They carry
no hardware semantics, and no compiler stage consumes them.
