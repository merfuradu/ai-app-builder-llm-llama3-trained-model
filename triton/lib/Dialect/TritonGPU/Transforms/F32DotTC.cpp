#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUF32DOTTC
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// nb. We call the trick TF32x3 as C++ disallows varaibles starting with numbers
// Implement 3xTF32 trick https://github.com/NVIDIA/cutlass/discussions/385
// For a, b f32
// dot(a, b, inputPrecision="tf32x3") ->
//  let aBig = f32ToTF32(a), aSmall = a - aBig;
//  let bBig = f32ToTF32(b), bSmall = b - bBig;
//  dot(aSmall, bBig, inputPrecision="tf32") +
//  dot(aBig, bSmall, inputPrecision="tf32") +
//  dot(aBig, bBig, inputPrecision="tf32")
class TF32x3 : public OpRewritePattern<DotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOp dotOp,
                                PatternRewriter &rewriter) const override {

    auto isF32 = [](Value operand) {
      return cast<RankedTensorType>(operand.getType()).getElementType().isF32();
    };

    if (!(dotOp.getInputPrecision() == InputPrecision::TF32x3 &&
          isF32(dotOp.getA()) && isF32(dotOp.getB()))) {
      return failure();
    }

    // Aux functions
    auto f32ToTF32 = [&](Value value) -> Value {
      return rewriter
          .create<ElementwiseInlineAsmOp>(dotOp.getLoc(), value.getType(),
                                          "cvt.rna.tf32.f32 $0, $1;", "=r,r",
                                          /*isPure=*/true, /*pack=*/1,
                                          ArrayRef<Value>{value})
          .getResult()[0];
    };
    auto sub = [&](Value a, Value b) -> Value {
      return rewriter.create<arith::SubFOp>(dotOp.getLoc(), a, b);
    };
    auto dot = [&](Value a, Value b, Value c) -> Value {
      return rewriter.create<DotOp>(dotOp->getLoc(), c.getType(), a, b, c,
                                    InputPrecision::TF32,
                                    dotOp.getMaxNumImpreciseAcc());
    };

    auto aBig = f32ToTF32(dotOp.getA());
    auto aSmall = sub(dotOp.getA(), aBig);

    auto bBig = f32ToTF32(dotOp.getB());
    auto bSmall = sub(dotOp.getB(), bBig);

    auto dot1 = dot(aSmall, bBig, dotOp.getC());
    auto dot2 = dot(aBig, bSmall, dot1);
    auto dot3 = dot(aBig, bBig, dot2);

    rewriter.replaceOp(dotOp, dot3);
    return success();
  }
};

} // anonymous namespace

struct F32DotTCPass : public impl::TritonGPUF32DotTCBase<F32DotTCPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet decomposePatterns(context);
    decomposePatterns.add<TF32x3>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(decomposePatterns))
            .failed()) {
      signalPassFailure();
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
