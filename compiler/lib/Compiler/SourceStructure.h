#ifndef ACIR_LIB_COMPILER_SOURCESTRUCTURE_H
#define ACIR_LIB_COMPILER_SOURCESTRUCTURE_H

#include "Dialect/ACIR/ACIRSourceContracts.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"

namespace acir::compiler {

// The `kind` and `version` of the document describeSourceStructure returns.
inline constexpr llvm::StringLiteral kSourceStructureKind =
    "pycircuit-source-structure";
inline constexpr int64_t kSourceStructureVersion = 1;

// Describe the authored structure of one verified source unit body: its
// module definitions with their ports, child instances and the wires between
// them, state owners, Methods, Rules and the effects each Method or Rule names,
// plus the owner and primitive kind of every definition it imports. The
// description is read-only navigation data derived from the IR for editors;
// it is not a hardware artifact, carries no semantics of its own and is
// rejected for any stage other than `source`.
mlir::FailureOr<llvm::json::Value>
describeSourceStructure(mlir::ModuleOp unit, ac::detail::EmitError emitError);

} // namespace acir::compiler

#endif // ACIR_LIB_COMPILER_SOURCESTRUCTURE_H
