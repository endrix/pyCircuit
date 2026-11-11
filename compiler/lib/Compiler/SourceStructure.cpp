#include "SourceStructure.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "pycircuit/Dialect/ACIR/ACIRAttributes.h"
#include "pycircuit/Dialect/ACIR/ACIRDialect.h"
#include "pycircuit/Dialect/ACIR/ACIROps.h"
#include "pycircuit/Dialect/ACIR/ACIRTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <set>
#include <string>

using namespace mlir;

namespace acir::compiler {
namespace {

// -- Spelling -----------------------------------------------------------------

std::string shortName(StringRef qualified) {
  auto dot = qualified.rfind('.');
  return (dot == StringRef::npos ? qualified : qualified.drop_front(dot + 1))
      .str();
}

std::string staticText(Attribute attr);

// A compile-time expression tree in source spelling, as far as it is simple:
// literals, parameter references and arithmetic. Anything else is elided.
std::string treeText(DictionaryAttr tree) {
  auto kind = tree ? tree.getAs<StringAttr>("kind") : StringAttr();
  if (!kind)
    return "?";
  StringRef tag = kind.getValue();
  if (tag == "literal") {
    auto value = tree.getAs<DictionaryAttr>("value");
    Attribute payload = value ? value.get("value") : Attribute();
    if (auto integer = dyn_cast_or_null<ac::MathIntAttr>(payload))
      return integer.getValue().getValue().str();
    if (auto boolean = dyn_cast_or_null<BoolAttr>(payload))
      return boolean.getValue() ? "True" : "False";
    if (auto integer = dyn_cast_or_null<IntegerAttr>(payload))
      return std::to_string(integer.getInt());
    return "?";
  }
  if (tag == "reference") {
    auto ref = tree.getAs<DictionaryAttr>("ref");
    Attribute name = ref ? ref.get("name") : Attribute();
    if (auto text = dyn_cast_or_null<StringAttr>(name))
      return text.getValue().str();
    if (auto parts = dyn_cast_or_null<ArrayAttr>(name)) {
      SmallVector<StringRef> names;
      for (Attribute part : parts)
        if (auto text = dyn_cast<StringAttr>(part))
          names.push_back(text.getValue());
      return llvm::join(names, ".");
    }
    return "?";
  }
  if (tag == "unary") {
    auto op = tree.getAs<StringAttr>("operator");
    return (op ? op.getValue().str() : std::string("?")) +
           staticText(tree.get("operand"));
  }
  if (tag == "binary") {
    auto op = tree.getAs<StringAttr>("operator");
    return "(" + staticText(tree.get("lhs")) + " " +
           (op ? op.getValue().str() : std::string("?")) + " " +
           staticText(tree.get("rhs")) + ")";
  }
  return "...";
}

std::string staticText(Attribute attr) {
  if (auto expr = dyn_cast_or_null<ac::StaticExprAttr>(attr))
    return treeText(expr.getTree());
  if (auto tree = dyn_cast_or_null<DictionaryAttr>(attr))
    return treeText(tree);
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attr))
    return std::to_string(integer.getInt());
  return "?";
}

// A hardware type in the spelling the source uses (`u8`, `bits[W]`, a Struct
// name, `table[4, u8]`), without the source locations the IR attaches to it.
std::string typeText(Type type) {
  if (auto bits = dyn_cast<ac::BitsType>(type)) {
    std::string width = staticText(bits.getWidth());
    bool literal = !width.empty() && llvm::all_of(width, llvm::isDigit);
    if (literal && width.size() <= 2 && std::stoi(width) >= 1 &&
        std::stoi(width) <= 64)
      return "u" + width;
    return "bits[" + width + "]";
  }
  if (auto record = dyn_cast<ac::StructType>(type))
    return shortName(record.getName().getValue());
  if (auto choice = dyn_cast<ac::EnumType>(type))
    return shortName(choice.getName().getValue());
  if (auto table = dyn_cast<ac::TableType>(type)) {
    SmallVector<std::string> dimensions;
    for (Attribute dimension : table.getShape())
      dimensions.push_back(staticText(dimension));
    return "table[" + llvm::join(dimensions, ", ") + ", " +
           typeText(table.getElementType()) + "]";
  }
  if (isa<ac::MathIntType>(type))
    return "int";
  if (auto parameter = dyn_cast<ac::TypeParamType>(type))
    return parameter.getName().getValue().str();
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text.substr(0, text.find('<'));
}

llvm::json::Value location(Location loc) {
  std::optional<FileLineColRange> found;
  loc->walk([&](Location inner) {
    if (auto file = dyn_cast<FileLineColRange>(inner)) {
      found = file;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!found)
    return nullptr;
  return llvm::json::Object{
      {"path", found->getFilename().getValue()},
      {"line", static_cast<int64_t>(found->getStartLine())},
      {"column", static_cast<int64_t>(found->getStartColumn())}};
}

llvm::json::Array stringArray(ArrayAttr array) {
  llvm::json::Array values;
  if (array)
    for (Attribute element : array)
      if (auto text = dyn_cast<StringAttr>(element))
        values.push_back(text.getValue());
  return values;
}

std::string stringAt(ArrayAttr array, unsigned index, StringRef fallback) {
  if (array && index < array.size())
    if (auto text = dyn_cast<StringAttr>(array[index]))
      return text.getValue().str();
  return (fallback + llvm::Twine(index)).str();
}

// -- Definitions --------------------------------------------------------------

// What a callee publishes, for naming an instance's ports.
struct Signature {
  ArrayAttr inputNames, outputNames;
  std::optional<std::string> primitive;
};

Signature signatureOf(Operation *definition) {
  Signature signature;
  if (!definition)
    return signature;
  signature.inputNames = definition->getAttrOfType<ArrayAttr>("input_names");
  signature.outputNames = definition->getAttrOfType<ArrayAttr>("output_names");
  if (auto primitive = definition->getAttrOfType<StringAttr>("primitive_kind"))
    signature.primitive = primitive.getValue().str();
  return signature;
}

llvm::json::Array signaturePorts(Operation *definition, bool inputs) {
  llvm::json::Array ports;
  auto type = definition->getAttrOfType<TypeAttr>("function_type");
  auto function =
      type ? dyn_cast<FunctionType>(type.getValue()) : FunctionType();
  if (!function)
    return ports;
  auto names = definition->getAttrOfType<ArrayAttr>(inputs ? "input_names"
                                                           : "output_names");
  TypeRange types = inputs ? function.getInputs() : function.getResults();
  for (auto [index, element] : llvm::enumerate(types))
    ports.push_back(llvm::json::Object{
        {"name", stringAt(names, index, inputs ? "input" : "output")},
        {"type", typeText(element)}});
  return ports;
}

llvm::json::Array parameterNames(Operation *definition) {
  llvm::json::Array names;
  if (auto parameters = definition->getAttrOfType<ArrayAttr>("parameters"))
    for (Attribute parameter : parameters)
      if (auto entry = dyn_cast<DictionaryAttr>(parameter))
        if (auto name = entry.getAs<StringAttr>("name"))
          names.push_back(name.getValue());
  return names;
}

// The module references a definition takes (`Writer(counter: Counter)`).
llvm::json::Array referenceParameters(Operation *definition) {
  llvm::json::Array references;
  if (auto formals =
          definition->getAttrOfType<ArrayAttr>("ac.reference_parameters"))
    for (Attribute formal : formals)
      if (auto entry = dyn_cast<DictionaryAttr>(formal)) {
        auto callee = entry.getAs<FlatSymbolRefAttr>("callee");
        references.push_back(llvm::json::Object{
            {"name", entry.getAs<StringAttr>("name").getValue()},
            {"callee", callee ? llvm::json::Value(callee.getValue())
                              : llvm::json::Value(nullptr)}});
      }
  return references;
}

// -- One module ---------------------------------------------------------------

class ModuleDescriber {
public:
  ModuleDescriber(ac::ModuleOp module, const llvm::StringMap<Operation *> &defs)
      : module(module), definitions(defs) {
    if (auto domain = module->getAttrOfType<DictionaryAttr>("ac.domain_inputs"))
      for (StringRef role : {StringRef("clock"), StringRef("reset")})
        if (auto ordinal = domain.getAs<IntegerAttr>(role))
          domainInputs[ordinal.getInt()] = role.str();
  }

  llvm::json::Object describe() {
    Operation *op = module.getOperation();
    StringRef symbol = op->getAttrOfType<StringAttr>("sym_name").getValue();
    auto rootKind = op->getAttrOfType<StringAttr>("ac.root_kind");
    llvm::json::Array inputs = signaturePorts(op, /*inputs=*/true);
    for (auto [index, port] : llvm::enumerate(inputs))
      if (auto it = domainInputs.find(index); it != domainInputs.end())
        (*port.getAsObject())["domain"] = it->second;

    llvm::json::Array instances, states, methods, rules, accesses, outputs;
    Block &body = op->getRegion(0).front();
    for (Operation &child : body) {
      if (auto instance = dyn_cast<ac::InstanceOp>(child))
        instances.push_back(describeInstance(instance));
      else if (auto state = dyn_cast<ac::SourceStateOp>(child))
        states.push_back(llvm::json::Object{
            {"name", state.getInstanceName()},
            {"type", typeText(state->getResult(0).getType())},
            {"location", location(state.getLoc())}});
      else if (auto method = dyn_cast<ac::SourceMethodOp>(child)) {
        methods.push_back(describeMethod(method));
        collectAccesses(&child, method.getSymName(), accesses);
      } else if (auto rule = dyn_cast<ac::SourceTransactionOp>(child)) {
        rules.push_back(describeRule(rule, symbol));
        collectAccesses(&child,
                        rule->getAttrOfType<StringAttr>("name").getValue(),
                        accesses, /*skipBinding=*/true);
      } else if (isa<ac::YieldOp>(child)) {
        auto names = op->getAttrOfType<ArrayAttr>("output_names");
        for (auto [index, value] : llvm::enumerate(child.getOperands()))
          outputs.push_back(llvm::json::Object{
              {"port", stringAt(names, index, "output")},
              {"from", endpoint(value)}});
      }
    }

    return llvm::json::Object{
        {"symbol", symbol},
        {"name", shortName(symbol)},
        {"root_kind", rootKind ? rootKind.getValue() : StringRef("module")},
        {"location", location(op->getLoc())},
        {"parameters", parameterNames(op)},
        {"references", referenceParameters(op)},
        {"inputs", std::move(inputs)},
        {"outputs", signaturePorts(op, /*inputs=*/false)},
        {"instances", std::move(instances)},
        {"states", std::move(states)},
        {"methods", std::move(methods)},
        {"rules", std::move(rules)},
        {"accesses", std::move(accesses)},
        {"results", std::move(outputs)},
        {"logic", std::move(logic)}};
  }

private:
  Operation *lookup(StringRef symbol) const {
    auto it = definitions.find(symbol);
    return it == definitions.end() ? nullptr : it->second;
  }

  llvm::json::Object describeInstance(ac::InstanceOp instance) {
    StringRef callee =
        instance->getAttrOfType<FlatSymbolRefAttr>("callee").getValue();
    Operation *definition = lookup(callee);
    Signature signature = signatureOf(definition);
    llvm::json::Array inputs, outputs, parameters;
    for (auto [index, value] : llvm::enumerate(instance->getOperands())) {
      llvm::json::Object port{
          {"port", stringAt(signature.inputNames, index, "input")},
          {"type", typeText(value.getType())}};
      llvm::json::Value from = endpoint(value);
      const llvm::json::Object *source = from.getAsObject();
      std::optional<StringRef> kind =
          source ? source->getString("kind") : std::nullopt;
      if (kind == "domain")
        port["domain"] = source->getString("name")->str();
      else if (kind == "constant")
        port["constant"] = source->getString("value")->str();
      else
        port["from"] = std::move(from);
      inputs.push_back(std::move(port));
    }
    for (auto [index, result] : llvm::enumerate(instance->getResults()))
      outputs.push_back(llvm::json::Object{
          {"port", stringAt(signature.outputNames, index, "output")},
          {"type", typeText(result.getType())}});
    ArrayAttr formals = definition
                            ? definition->getAttrOfType<ArrayAttr>("parameters")
                            : ArrayAttr();
    if (auto actuals = instance->getAttrOfType<ArrayAttr>("parameters"))
      for (auto [index, actual] : llvm::enumerate(actuals)) {
        std::string name = "parameter" + std::to_string(index);
        if (formals && index < formals.size())
          if (auto entry = dyn_cast<DictionaryAttr>(formals[index]))
            if (auto formal = entry.getAs<StringAttr>("name"))
              name = formal.getValue().str();
        parameters.push_back(
            llvm::json::Object{{"name", name}, {"value", staticText(actual)}});
      }
    llvm::json::Array references;
    if (auto bindings =
            instance->getAttrOfType<ArrayAttr>("ac.reference_bindings"))
      for (Attribute binding : bindings)
        if (auto entry = dyn_cast<DictionaryAttr>(binding))
          references.push_back(llvm::json::Object{
              {"name", entry.getAs<StringAttr>("name").getValue()},
              {"receiver",
               receiver(entry.getAs<DictionaryAttr>("receiver"))}});
    llvm::json::Object described{
        {"name", instance.getInstanceName()},
        {"references", std::move(references)},
        {"callee", callee},
        {"location", location(instance.getLoc())},
        {"parameters", std::move(parameters)},
        {"inputs", std::move(inputs)},
        {"outputs", std::move(outputs)}};
    described["primitive"] = signature.primitive
                                 ? llvm::json::Value(*signature.primitive)
                                 : llvm::json::Value(nullptr);
    return described;
  }

  // The instance a reference binding passes: a formal of this module, or a
  // child instance path whose steps are source occurrences. Steps resolve
  // through every definition body this unit holds; a step into an imported
  // definition stays unresolved.
  llvm::json::Value receiver(DictionaryAttr receiver) {
    if (!receiver)
      return llvm::json::Object{{"kind", "unknown"}};
    llvm::json::Array names;
    Operation *scope = module.getOperation();
    bool resolved = true;
    if (auto steps = receiver.getAs<ArrayAttr>("instance_path"))
      for (Attribute step : steps) {
        ac::InstanceOp found;
        if (scope)
          for (auto child : scope->getRegion(0).getOps<ac::InstanceOp>())
            if (child->getAttr("occurrence") == step) {
              found = child;
              break;
            }
        if (!found) {
          resolved = false;
          break;
        }
        names.push_back(found.getInstanceName());
        Operation *next = lookup(
            found->getAttrOfType<FlatSymbolRefAttr>("callee").getValue());
        scope = isa_and_nonnull<ac::ModuleOp>(next) ? next : nullptr;
      }
    llvm::json::Object described{{"path", std::move(names)}};
    if (auto reference = receiver.getAs<StringAttr>("reference")) {
      described["kind"] = "reference";
      described["name"] = reference.getValue();
    } else {
      described["kind"] = "instance";
    }
    if (!resolved)
      described["unresolved"] = true;
    return described;
  }

  llvm::json::Object describeMethod(ac::SourceMethodOp method) {
    llvm::json::Array arguments, results;
    auto names = method->getAttrOfType<ArrayAttr>("argument_names");
    auto function = method.getFunctionType();
    for (auto [index, type] : llvm::enumerate(function.getInputs()))
      arguments.push_back(llvm::json::Object{
          {"name", stringAt(names, index, "argument")},
          {"type", typeText(type)}});
    for (Type type : function.getResults())
      results.push_back(llvm::json::Object{{"type", typeText(type)}});
    return llvm::json::Object{
        {"name", method.getSymName()},
        {"location", location(method.getLoc())},
        {"arguments", std::move(arguments)},
        {"results", std::move(results)},
        {"captures", usedCaptures(method, function.getNumInputs())}};
  }

  // The enclosing values a Method or Rule body reads. Both regions take the
  // explicit inputs followed by one argument per capture; a capture neither
  // region uses is not a dependency.
  llvm::json::Array usedCaptures(Operation *op, unsigned explicitInputs) {
    llvm::json::Array captures;
    for (auto [index, capture] : llvm::enumerate(op->getOperands())) {
      if (isa<ac::SourceTransactionOp>(op) && index < 2)
        continue; // clock and reset
      unsigned position =
          explicitInputs + index - (isa<ac::SourceTransactionOp>(op) ? 2 : 0);
      bool used = llvm::any_of(op->getRegions(), [&](Region &region) {
        return !region.empty() && position < region.getNumArguments() &&
               !region.getArgument(position).use_empty();
      });
      if (used)
        captures.push_back(endpoint(capture));
    }
    return captures;
  }

  // A Rule registers a transaction whose body calls the module's own Method
  // of the same declaration; that Method holds the Rule's effects.
  llvm::json::Object describeRule(ac::SourceTransactionOp rule,
                                  StringRef symbol) {
    llvm::json::Value bodyMethod = nullptr;
    rule->walk([&](ac::SourceMethodCallOp call) {
      auto path = call->getAttrOfType<ArrayAttr>("instance_path");
      auto callee = call->getAttrOfType<SymbolRefAttr>("callee");
      if ((!path || path.empty()) && callee &&
          callee.getRootReference().getValue() == symbol) {
        bodyMethod = callee.getLeafReference().getValue();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    llvm::json::Array results;
    for (Type type : rule->getResultTypes())
      results.push_back(llvm::json::Object{{"type", typeText(type)}});
    return llvm::json::Object{
        {"name", rule->getAttrOfType<StringAttr>("name").getValue()},
        {"method", std::move(bodyMethod)},
        {"location", location(rule.getLoc())},
        {"results", std::move(results)},
        {"captures", usedCaptures(rule, 0)}};
  }

  void collectAccesses(Operation *actor, StringRef name,
                       llvm::json::Array &accesses, bool skipBinding = false) {
    StringRef symbol =
        module->getAttrOfType<StringAttr>("sym_name").getValue();
    actor->walk([&](Operation *op) {
      if (op == actor)
        return;
      llvm::json::Value target = nullptr;
      std::string operation;
      if (auto call = dyn_cast<ac::SourceMethodCallOp>(op)) {
        auto path = call->getAttrOfType<ArrayAttr>("instance_path");
        auto callee = call->getAttrOfType<SymbolRefAttr>("callee");
        if (!callee)
          return;
        auto reference = call->getAttrOfType<StringAttr>("ac.reference");
        bool self = !reference && (!path || path.empty());
        if (self && skipBinding &&
            callee.getRootReference().getValue() == symbol)
          return;
        operation = "call";
        target = llvm::json::Object{
            {"kind", reference ? "reference" : self ? "self" : "instance"},
            {"name", reference ? llvm::json::Value(reference.getValue())
                               : llvm::json::Value(nullptr)},
            {"path", stringArray(path)},
            {"method", callee.getLeafReference().getValue()},
            {"arguments", static_cast<int64_t>(call.getArguments().size())},
            {"results", static_cast<int64_t>(call->getNumResults())}};
      } else if (auto read = dyn_cast<ac::SourceStateReadOp>(op)) {
        operation = "read";
        target = llvm::json::Object{{"kind", "state"},
                                    {"name", read.getState()}};
      } else if (auto write = dyn_cast<ac::SourceStateWriteOp>(op)) {
        operation = "write";
        target = llvm::json::Object{{"kind", "state"},
                                    {"name", write.getState()}};
      } else if (auto observe = dyn_cast<ac::SourceObserveOp>(op)) {
        operation = observe.getKind().str();
      } else if (auto expect = dyn_cast<ac::SourceExpectOp>(op)) {
        operation = expect.getKind().str();
      } else {
        return;
      }
      accesses.push_back(
          llvm::json::Object{{"actor", name},
                             {"operation", operation},
                             {"target", std::move(target)},
                             {"location", location(op->getLoc())}});
    });
  }

  // Where a wire comes from: a module input (or its hidden clock/reset), an
  // instance output, a state owner, a constant, or a pure expression drawn as
  // one logic node whose inputs are the endpoints the expression reads.
  llvm::json::Value endpoint(Value value) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (argument.getOwner()->getParentOp() != module.getOperation())
        return llvm::json::Object{{"kind", "unknown"}};
      unsigned index = argument.getArgNumber();
      if (auto it = domainInputs.find(index); it != domainInputs.end())
        return llvm::json::Object{{"kind", "domain"}, {"name", it->second}};
      return llvm::json::Object{
          {"kind", "input"},
          {"name", stringAt(module->getAttrOfType<ArrayAttr>("input_names"),
                            index, "input")}};
    }
    Operation *definition = value.getDefiningOp();
    if (auto instance = dyn_cast<ac::InstanceOp>(definition)) {
      Signature signature = signatureOf(lookup(
          instance->getAttrOfType<FlatSymbolRefAttr>("callee").getValue()));
      return llvm::json::Object{
          {"kind", "instance"},
          {"name", instance.getInstanceName()},
          {"port", stringAt(signature.outputNames,
                            cast<OpResult>(value).getResultNumber(),
                            "output")}};
    }
    if (auto rule = dyn_cast<ac::SourceTransactionOp>(definition))
      return llvm::json::Object{
          {"kind", "rule"},
          {"name", rule->getAttrOfType<StringAttr>("name").getValue()},
          {"port", "result" + std::to_string(
                                  cast<OpResult>(value).getResultNumber())}};
    if (auto state = dyn_cast<ac::SourceStateOp>(definition))
      return llvm::json::Object{{"kind", "state"},
                                {"name", state.getInstanceName()}};
    if (isConstant(definition))
      return llvm::json::Object{{"kind", "constant"},
                                {"value", constantText(definition)}};
    return llvm::json::Object{{"kind", "logic"}, {"id", logicFor(definition)}};
  }

  static bool isConstant(Operation *op) {
    return isa<ac::BitsConstantOp, ac::MathConstantOp>(op);
  }

  static std::string constantText(Operation *op) {
    return staticText(op->getAttr("value"));
  }

  std::string logicFor(Operation *root) {
    if (auto it = logicIds.find(root); it != logicIds.end())
      return it->second;
    std::string id = "logic" + std::to_string(logicIds.size());
    logicIds[root] = id;
    llvm::json::Array inputs;
    llvm::DenseSet<Operation *> visited;
    std::set<std::string> seen;
    SmallVector<Operation *> stack{root};
    while (!stack.empty()) {
      Operation *op = stack.pop_back_val();
      if (!visited.insert(op).second)
        continue;
      for (Value operand : op->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        bool leaf = !producer || isa<ac::InstanceOp, ac::SourceStateOp,
                                     ac::SourceTransactionOp>(producer);
        if (!leaf && isConstant(producer))
          continue;
        if (!leaf) {
          stack.push_back(producer);
          continue;
        }
        llvm::json::Value source = endpoint(operand);
        const llvm::json::Object *object = source.getAsObject();
        if (object && object->getString("kind") == "domain")
          continue;
        std::string key;
        llvm::raw_string_ostream stream(key);
        stream << source;
        if (seen.insert(key).second)
          inputs.push_back(std::move(source));
      }
    }
    StringRef name = root->getName().getStringRef();
    name.consume_front("ac.");
    logic.push_back(llvm::json::Object{{"id", id},
                                       {"operation", name},
                                       {"location", location(root->getLoc())},
                                       {"inputs", std::move(inputs)}});
    return id;
  }

  ac::ModuleOp module;
  const llvm::StringMap<Operation *> &definitions;
  llvm::DenseMap<unsigned, std::string> domainInputs;
  llvm::DenseMap<Operation *, std::string> logicIds;
  llvm::json::Array logic;
};

} // namespace

FailureOr<llvm::json::Value>
describeSourceStructure(mlir::ModuleOp unit, ac::detail::EmitError emitError) {
  auto stage = unit->getAttrOfType<StringAttr>("ac.stage");
  if (!stage || stage.getValue() != "source") {
    emitError() << "structure description requires a source unit body "
                   "(ac.stage = \"source\")";
    return failure();
  }
  llvm::StringMap<Operation *> definitions;
  for (Operation &op : unit.getBody()->getOperations())
    if (isa<ac::ModuleOp, ac::ModuleImportOp>(op))
      if (auto symbol = op.getAttrOfType<StringAttr>("sym_name"))
        definitions[symbol.getValue()] = &op;

  llvm::json::Array modules, imports;
  for (Operation &op : unit.getBody()->getOperations()) {
    if (auto module = dyn_cast<ac::ModuleOp>(op)) {
      modules.push_back(ModuleDescriber(module, definitions).describe());
      continue;
    }
    auto import = dyn_cast<ac::ModuleImportOp>(op);
    if (!import)
      continue;
    llvm::json::Object entry{
        {"symbol", import->getAttrOfType<StringAttr>("sym_name").getValue()}};
    Signature signature = signatureOf(&op);
    entry["primitive"] = signature.primitive
                             ? llvm::json::Value(*signature.primitive)
                             : llvm::json::Value(nullptr);
    if (auto owner = import->getAttrOfType<DictionaryAttr>("source_owner"))
      if (auto package = owner.getAs<StringAttr>("package"))
        if (auto path = owner.getAs<StringAttr>("path"))
          entry["source"] = llvm::json::Object{{"package", package.getValue()},
                                               {"path", path.getValue()}};
    imports.push_back(std::move(entry));
  }

  llvm::json::Object source;
  if (auto owner = unit->getAttrOfType<DictionaryAttr>("ac.source_owner"))
    for (StringRef key : {StringRef("package"), StringRef("path")})
      if (auto value = owner.getAs<StringAttr>(key))
        source[key] = value.getValue();
  return llvm::json::Value(llvm::json::Object{
      {"kind", kSourceStructureKind},
      {"version", kSourceStructureVersion},
      {"source", std::move(source)},
      {"modules", std::move(modules)},
      {"imports", std::move(imports)}});
}

} // namespace acir::compiler
