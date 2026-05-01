#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <string>

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONCOSTMODEL
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-cost-model"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

enum class Opcode : uint8_t {
  // Arithmetic (scalar/vector)
  IAdd, // integer add
  ISub,
  IMul,
  IDiv,

  FAdd,
  FSub,
  FMul,
  FDiv,

  FFma, // float fused multiply-add
  HFma, // half precision FMA

  FMin,
  FMax,

  FSqrt,
  FRsqrt,
  FExp,
  FSilu, // sigmoid * x

  // Type conversion
  I2F,
  F2I,
  I2I,
  F2F,

  // Memory ops
  Ldg,    // global -> reg
  Lds,    // shared -> reg
  Ldgsts, // global -> shared (cp.async equivalent)
  Sts,    // reg -> shared
  Stg,    // reg -> global
  Tmastg, // shared -> global (TMA store)

  // Tensor Core / MMA
  Hmma,  // tensor core fp16
  Imma,  // tensor core int
  Wgmma, // Hopper warpgroup MMA

  // Control / sync
  Barrier,   // __syncthreads / barrier
  AsyncWait, // cp.async.wait_group
  MBarrier,  // mbarrier

  // Special / fallback
  Default
};
static const std::unordered_map<Opcode, int> independent_cpi_lut = {
    // Arithmetic
    {Opcode::IAdd, 2},
    {Opcode::IMul, 2},
    {Opcode::FAdd, 2},
    {Opcode::FMul, 2},
    {Opcode::FFma, 2},
    {Opcode::HFma, 2},
    {Opcode::FDiv, 253},
    {Opcode::FMin, 2},
    {Opcode::FMax, 2},
    {Opcode::FSqrt, 40},
    {Opcode::FRsqrt, 18},
    {Opcode::FExp, 43},
    {Opcode::FSilu, 300},

    // Casts
    {Opcode::I2F, 2},
    {Opcode::F2I, 2},
    {Opcode::I2I, 2},
    {Opcode::F2F, 2},

    // Memory
    {Opcode::Ldg, 2},
    {Opcode::Lds, 2},
    {Opcode::Ldgsts, 2},
    {Opcode::Sts, 2},
    {Opcode::Stg, 2},
    {Opcode::Tmastg, 2},

    // Tensor core
    {Opcode::Hmma, 8},
    {Opcode::Imma, 8},
    {Opcode::Wgmma, 8},

    // Sync
    {Opcode::Barrier, 0}, // sync has no issue cost, only stall
    {Opcode::AsyncWait, 0},
    {Opcode::MBarrier, 0},

    // Default fallback
    {Opcode::Default, 2},
};

static const std::unordered_map<Opcode, int> dependent_cpi_lut = {
    // Arithmetic
    {Opcode::IAdd, 5},
    {Opcode::IMul, 4},
    {Opcode::FAdd, 4},
    {Opcode::FMul, 4},
    {Opcode::FFma, 4},
    {Opcode::HFma, 4},
    {Opcode::FDiv, 253},
    {Opcode::FMin, 10},
    {Opcode::FMax, 10},
    {Opcode::FSqrt, 40},
    {Opcode::FRsqrt, 18},
    {Opcode::FExp, 43},
    {Opcode::FSilu, 300},

    // Casts
    {Opcode::I2F, 23},
    {Opcode::F2I, 23},
    {Opcode::I2I, 23},
    {Opcode::F2F, 6},

    // Memory
    {Opcode::Ldg, 280},    // ~L2/DRAM latency
    {Opcode::Lds, 30},     // shared memory
    {Opcode::Ldgsts, 280}, // async global->shared
    {Opcode::Sts, 19},
    {Opcode::Stg, 280},
    {Opcode::Tmastg, 280},

    // Tensor core
    {Opcode::Hmma, 16},
    {Opcode::Imma, 16},
    {Opcode::Wgmma, 16}, // approximation (handled specially sometimes)

    // Sync
    {Opcode::Barrier, 0},   // handled via pipeline flush logic
    {Opcode::AsyncWait, 0}, // modeled explicitly elsewhere
    {Opcode::MBarrier, 0},

    // Default fallback
    {Opcode::Default, 4},
};

int get_latency(const std::unordered_map<Opcode, int> &lut, Opcode op) {
  auto it = lut.find(op);
  if (it != lut.end())
    return it->second;
  return lut.at(Opcode::Default);
}

int64_t getNumElements(Operation *op) {
  if (op->getNumResults() == 0)
    return 1;

  auto type = op->getResult(0).getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    int64_t num = 1;
    for (int64_t d : tensorType.getShape())
      num *= d;
    return num;
  }
  return 1;
}

int64_t getBitWidthSafe(Type t) {
  if (auto tt = dyn_cast<RankedTensorType>(t))
    t = tt.getElementType();
  if (auto it = dyn_cast<IntegerType>(t))
    return it.getWidth();
  if (auto ft = dyn_cast<FloatType>(t))
    return ft.getWidth();
  return 32;
}

int64_t estimateElementwiseInsts(Operation *op) {
  int64_t elems = getNumElements(op);

  int64_t vecWidth = 1;

  // crude heuristic: assume 128-bit vectorization
  auto type = op->getResult(0).getType();
  if (auto tt = dyn_cast<RankedTensorType>(type)) {
    int bitwidth = getBitWidthSafe(tt);
    vecWidth = std::max<int64_t>(1, 128 / bitwidth);
  }

  return (elems + vecWidth - 1) / vecWidth;
}

int64_t estimateCopyInsts(Operation *op, Opcode opcode) {
  int64_t elems = getNumElements(op);

  auto type = op->getOperand(0).getType();
  int bitwidth = 32;

  if (auto tt = dyn_cast<RankedTensorType>(type))
    bitwidth = getBitWidthSafe(tt);

  int64_t elemsPerInst = std::max<int64_t>(1, 128 / bitwidth);
  if (opcode == Opcode::Ldgsts)
    elemsPerInst = 16 / (bitwidth / 8);

  return std::max<int64_t>(1, (elems + elemsPerInst - 1) / elemsPerInst);
}

int64_t estimateMmaInsts(Operation *op) {
  auto type = op->getResult(0).getType();
  auto tensorType = dyn_cast<RankedTensorType>(type);

  if (!tensorType)
    return 1;

  auto shape = tensorType.getShape();
  int64_t M = shape[0];
  int64_t N = shape[1];

  auto aType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  int64_t K = aType.getShape()[1];

  // Assume 16x16x16 tiles
  int64_t mt = 16, nt = 16, kt = 16;

  return (M / mt) * (N / nt) * (K / kt);
}

bool isFloat(Type t) {
  if (auto tt = dyn_cast<RankedTensorType>(t))
    return tt.getElementType().isFloat();
  return t.isFloat();
}

bool isInt(Type t) {
  if (auto tt = dyn_cast<RankedTensorType>(t))
    return tt.getElementType().isInteger();
  return t.isInteger();
}

class LatencyModel {
private:
  std::unordered_map<Operation *, int64_t> ops_on_the_fly;
  llvm::DenseMap<Value, Operation *>
      var2op; // last op that writes to a variable
public:
  LatencyModel() : ops_on_the_fly(), var2op() {}
  int64_t visit(Operation *op) {
    if (!op)
      return 0;
    int64_t cycles = 0;

    if (op->getDialect()->getNamespace() == "arith")
      cycles += visitArithOp(op);
    else if (auto forOp = dyn_cast<scf::ForOp>(op))
      return visitFor(forOp);
    else if (auto yield = dyn_cast<scf::YieldOp>(op))
      return 0;
    else {
      auto opNamespace = op->getDialect()->getNamespace();
      if (opNamespace == "tt" || opNamespace == "ttg") {
        cycles += visitTritonOp(op);
      }
    }

    // walk regions
    for (Region &r : op->getRegions()) {
      for (Block &b : r) {
        for (Operation &nested : b) {
          cycles += visit(&nested);
        }
      }
    }
    return cycles;
  }
  int64_t visitFor(scf::ForOp forOp) {
    int64_t bodyCost = 0;

    for (Operation &nested : forOp.getBody()->without_terminator()) {
      bodyCost += visit(&nested);
    }

    // Try constant trip count
    auto lb = forOp.getLowerBound();
    auto ub = forOp.getUpperBound();
    auto step = forOp.getStep();

    int64_t tripCount = 8; // fallback like Python

    if (auto lbC = getConstantIntValue(lb))
      if (auto ubC = getConstantIntValue(ub))
        if (auto stepC = getConstantIntValue(step))
          tripCount = (*ubC - *lbC) / *stepC;

    return bodyCost * tripCount;
  }
  int64_t visitArithmetic(Operation *op, Opcode opcode, int64_t numElems) {
    int64_t cycles = 0;

    // RAW dependency
    for (Value operand : op->getOperands()) {
      if (var2op.count(operand)) {
        Operation *producer = var2op[operand];
        if (ops_on_the_fly.count(producer)) {
          cycles = std::max(cycles, ops_on_the_fly[producer]);
        }
      }
    }

    int issue_latency = get_latency(independent_cpi_lut, opcode);
    int64_t issue_cycles = issue_latency * numElems;
    cycles += issue_cycles;

    // Update in-flight ops
    std::vector<Operation *> toErase;
    for (auto &[o, remain] : ops_on_the_fly) {
      if (remain > cycles) {
        remain -= cycles;
      } else {
        toErase.push_back(o);
      }
    }
    for (auto *o : toErase)
      ops_on_the_fly.erase(o);

    int dep_latency = get_latency(dependent_cpi_lut, opcode);
    if (dep_latency > issue_latency) {
      ops_on_the_fly[op] = dep_latency - issue_latency;
    }

    // Track outputs
    for (Value res : op->getResults()) {
      var2op[res] = op;
    }

    return cycles;
  }
  int64_t visitArithOp(Operation *op) {
    if (isa<arith::ConstantOp>(op)) {
      return 0;
    }
    if (auto addf = dyn_cast<arith::AddFOp>(op)) {
      return visitArithmetic(op, Opcode::FAdd, estimateElementwiseInsts(op));
    }

    if (auto addi = dyn_cast<arith::AddIOp>(op)) {
      return visitArithmetic(op, Opcode::IAdd, estimateElementwiseInsts(op));
    }

    if (auto subf = dyn_cast<arith::SubFOp>(op)) {
      return visitArithmetic(op, Opcode::FAdd,
                             estimateElementwiseInsts(op)); // PTX uses add
    }

    if (auto subi = dyn_cast<arith::SubIOp>(op)) {
      return visitArithmetic(op, Opcode::IAdd, estimateElementwiseInsts(op));
    }

    if (auto mulf = dyn_cast<arith::MulFOp>(op)) {
      return visitArithmetic(op, Opcode::FMul, estimateElementwiseInsts(op));
    }

    if (auto muli = dyn_cast<arith::MulIOp>(op)) {
      return visitArithmetic(op, Opcode::IMul, estimateElementwiseInsts(op));
    }

    if (auto divf = dyn_cast<arith::DivFOp>(op)) {
      return visitArithmetic(op, Opcode::FDiv, estimateElementwiseInsts(op));
    }
    if (auto divi = dyn_cast<arith::DivSIOp>(op)) {
      return visitArithmetic(op, Opcode::IDiv, estimateElementwiseInsts(op));
    }

    if (auto rem = dyn_cast<arith::RemSIOp>(op)) {
      return visitArithmetic(op, Opcode::IDiv, estimateElementwiseInsts(op));
    }

    if (auto min = dyn_cast<arith::MinSIOp>(op)) {
      return visitArithmetic(op, Opcode::IAdd, estimateElementwiseInsts(op));
    }

    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      return visitArithmetic(op, Opcode::IAdd, estimateElementwiseInsts(op));
    }
    if (auto andi = dyn_cast<arith::AndIOp>(op)) {
      return visitArithmetic(op, Opcode::IAdd, estimateElementwiseInsts(op));
    }
    if (auto cast = dyn_cast<arith::ExtFOp>(op)) {
      return visitArithmetic(op, Opcode::F2F, estimateElementwiseInsts(op));
    }

    if (auto cast = dyn_cast<arith::TruncFOp>(op)) {
      return visitArithmetic(op, Opcode::F2F, estimateElementwiseInsts(op));
    }

    if (auto cast = dyn_cast<arith::SIToFPOp>(op)) {
      return visitArithmetic(op, Opcode::I2F, estimateElementwiseInsts(op));
    }

    if (auto cast = dyn_cast<arith::FPToSIOp>(op)) {
      return visitArithmetic(op, Opcode::F2I, estimateElementwiseInsts(op));
    }
    LLVM_DEBUG({ DBGS() << "Unhandled Arith op: " << op->getName() << "\n"; });
    return 0;
  }
  int64_t visitTritonOp(Operation *op) {
    if (auto load = dyn_cast<triton::LoadOp>(op))
      return visitCopy(op, Opcode::Ldg);

    if (auto store = dyn_cast<triton::StoreOp>(op))
      return visitCopy(op, Opcode::Stg);

    if (auto dot = dyn_cast<triton::DotOp>(op))
      return visitMma(op);
    // Async copy (important!)
    if (op->getName().getStringRef().contains("async_copy")) {
      return visitCopy(op, Opcode::Ldgsts);
    }

    // Reductions (simplified)
    if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
      // Python model treats reduce specially
      // For now: approximate as repeated adds
      return visitArithmetic(op, Opcode::FAdd, getNumElements(op));
    }

    // Layout / address-only ops (zero cost)
    if (isa<triton::BroadcastOp>(op) || isa<triton::ExpandDimsOp>(op) ||
        isa<triton::SplatOp>(op)) {
      return 0;
    }

    // Control / sync
    if (op->getName().getStringRef().contains("barrier")) {
      // flush pipeline
      int64_t cycles = 0;
      for (auto &[o, remain] : ops_on_the_fly)
        cycles = std::max(cycles, remain);
      ops_on_the_fly.clear();
      return cycles;
    }

    if (op->getName().getStringRef().contains("wait")) {
      return 0; // handled implicitly later
    }
    // Function container
    if (isa<triton::FuncOp>(op)) {
      return 0;
    }

    // Program ID (thread/block index)
    if (isa<triton::GetProgramIdOp>(op)) {
      return 0;
    }

    // Range creation
    if (isa<triton::MakeRangeOp>(op)) {
      return 0;
    }

    // Pointer arithmetic
    if (op->getName().getStringRef().contains("addptr")) {
      return 0;
    }

    // Layout conversions
    if (op->getName().getStringRef().contains("convert_layout")) {
      return 0;
    }
    // return
    if (isa<triton::ReturnOp>(op)) {
      return 0;
    }

    // Fallback
    LLVM_DEBUG({ DBGS() << "Unhandled Triton op: " << op->getName() << "\n"; });

    return 0;
  }

  int64_t visitCopy(Operation *op, Opcode opcode) {
    int64_t cycles = 0;

    // RAW dependency (source)
    if (!op->getOperands().empty()) {
      Value src = op->getOperand(0);
      if (var2op.count(src)) {
        Operation *producer = var2op[src];
        if (ops_on_the_fly.count(producer)) {
          cycles = ops_on_the_fly[producer];
        }
      }
    }

    int64_t numInsts = estimateCopyInsts(op, opcode);

    int issue_latency = get_latency(independent_cpi_lut, opcode);
    int64_t issue_cycles = issue_latency * numInsts;
    cycles += issue_cycles;

    // Update pipeline
    std::vector<Operation *> toErase;
    for (auto &[o, remain] : ops_on_the_fly) {
      if (remain > cycles) {
        remain -= cycles;
      } else {
        toErase.push_back(o);
      }
    }
    for (auto *o : toErase)
      ops_on_the_fly.erase(o);

    int dep_latency = get_latency(dependent_cpi_lut, opcode);
    ops_on_the_fly[op] = dep_latency - issue_latency;

    for (Value res : op->getResults()) {
      var2op[res] = op;
    }

    return cycles;
  }
  int64_t visitMma(Operation *op) {
    int64_t cycles = 0;

    for (Value operand : op->getOperands()) {
      if (var2op.count(operand)) {
        Operation *producer = var2op[operand];
        if (ops_on_the_fly.count(producer)) {
          cycles = std::max(cycles, ops_on_the_fly[producer]);
        }
      }
    }

    int64_t numInsts = estimateMmaInsts(op);

    int issue_latency = get_latency(independent_cpi_lut, Opcode::Hmma);
    cycles += issue_latency * numInsts;

    int dep_latency = get_latency(dependent_cpi_lut, Opcode::Hmma);
    if (dep_latency > issue_latency) {
      ops_on_the_fly[op] = dep_latency - issue_latency;
    }

    for (Value res : op->getResults()) {
      var2op[res] = op;
    }

    return cycles;
  }
  int64_t predict(Operation *op) { return visit(op); }
};

class CostModelPass : public impl::TritonCostModelBase<CostModelPass> {

public:
  void runOnOperation() override {
    Operation *op = getOperation();
    LDBG("Running Cost model pass");

    LatencyModel model;
    int64_t cost = model.predict(op);
    LLVM_DEBUG({ DBGS() << "Total cost: " << cost << "\n"; });
  }
};

} // namespace mlir::triton
