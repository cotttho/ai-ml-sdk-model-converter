/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

namespace mlir::model_converter_passes {
#define GEN_PASS_DEF_TOSARESCALESIMPLIFYPASS
#include "passes.hpp.inc"
namespace {

std::optional<int64_t> getScalarInt(Value value) {
    auto constOp = value.getDefiningOp<tosa::ConstOp>();
    if (!constOp)
        return std::nullopt;

    auto values = llvm::dyn_cast<DenseIntElementsAttr>(constOp.getValuesAttr());
    if (!values || values.getNumElements() != 1)
        return std::nullopt;

    return (*values.value_begin<APInt>()).getSExtValue();
}

bool isIdentityRescale(tosa::RescaleOp rescaleOp) {
    if (rescaleOp.getPerChannel())
        return false;

    if (rescaleOp.getInput().getType() != rescaleOp.getOutput().getType())
        return false;

    if (rescaleOp.getInputUnsigned() != rescaleOp.getOutputUnsigned())
        return false;

    const std::optional<int64_t> multiplier = getScalarInt(rescaleOp.getMultiplier());
    const std::optional<int64_t> shift = getScalarInt(rescaleOp.getShift());
    const std::optional<int64_t> inputZp = getScalarInt(rescaleOp.getInputZp());
    const std::optional<int64_t> outputZp = getScalarInt(rescaleOp.getOutputZp());
    if (!multiplier || !shift || !inputZp || !outputZp)
        return false;

    if (*inputZp != *outputZp)
        return false;

    if (*shift < 0 || *shift > 62)
        return false;

    return *multiplier == (int64_t{1} << *shift);
}

class RemoveIdentityRescalePattern final : public OpRewritePattern<tosa::RescaleOp> {
  public:
    using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tosa::RescaleOp rescaleOp, PatternRewriter &rewriter) const override {
        if (!isIdentityRescale(rescaleOp))
            return failure();

        rewriter.replaceOp(rescaleOp, rescaleOp.getInput());
        return success();
    }
};

class TosaRescaleSimplifyPass final : public impl::TosaRescaleSimplifyPassBase<TosaRescaleSimplifyPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<RemoveIdentityRescalePattern>(&getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

} // namespace mlir::model_converter_passes
