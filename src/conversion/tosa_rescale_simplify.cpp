/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cstdint>
#include <optional>
#include <utility>

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

std::optional<std::pair<int64_t, int64_t>> getInterpretedIntegerRange(Value value, bool isUnsigned) {
    auto shapedType = llvm::dyn_cast<ShapedType>(value.getType());
    if (!shapedType)
        return std::nullopt;

    auto elementType = llvm::dyn_cast<IntegerType>(shapedType.getElementType());
    if (!elementType)
        return std::nullopt;

    const unsigned width = elementType.getWidth();
    if (width == 0 || width >= 63)
        return std::nullopt;

    if (isUnsigned)
        return std::make_pair(int64_t{0}, (int64_t{1} << width) - 1);

    return std::make_pair(-(int64_t{1} << (width - 1)), (int64_t{1} << (width - 1)) - 1);
}

bool isUnitScale(tosa::RescaleOp rescaleOp) {
    if (rescaleOp.getPerChannel())
        return false;

    const std::optional<int64_t> multiplier = getScalarInt(rescaleOp.getMultiplier());
    const std::optional<int64_t> shift = getScalarInt(rescaleOp.getShift());
    if (!multiplier || !shift)
        return false;

    if (*shift < 0 || *shift > 62)
        return false;

    return *multiplier == (int64_t{1} << *shift);
}

bool unitScaleRebaseCannotClip(tosa::RescaleOp rescaleOp) {
    const std::optional<int64_t> inputZp = getScalarInt(rescaleOp.getInputZp());
    const std::optional<int64_t> outputZp = getScalarInt(rescaleOp.getOutputZp());
    if (!inputZp || !outputZp)
        return false;

    const std::optional<std::pair<int64_t, int64_t>> inputRange =
        getInterpretedIntegerRange(rescaleOp.getInput(), rescaleOp.getInputUnsigned());
    const std::optional<std::pair<int64_t, int64_t>> outputRange =
        getInterpretedIntegerRange(rescaleOp.getOutput(), rescaleOp.getOutputUnsigned());
    if (!inputRange || !outputRange)
        return false;

    const int64_t rebasedMin = inputRange->first - *inputZp + *outputZp;
    const int64_t rebasedMax = inputRange->second - *inputZp + *outputZp;
    return rebasedMin >= outputRange->first && rebasedMax <= outputRange->second;
}

bool isIdentityRescale(tosa::RescaleOp rescaleOp) {
    if (rescaleOp.getInput().getType() != rescaleOp.getOutput().getType())
        return false;

    if (rescaleOp.getInputUnsigned() != rescaleOp.getOutputUnsigned())
        return false;

    const std::optional<int64_t> inputZp = getScalarInt(rescaleOp.getInputZp());
    const std::optional<int64_t> outputZp = getScalarInt(rescaleOp.getOutputZp());
    if (!inputZp || !outputZp)
        return false;

    if (*inputZp != *outputZp)
        return false;

    return isUnitScale(rescaleOp);
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

class FoldUnitScaleRebaseIntoConsumerPattern final : public OpRewritePattern<tosa::RescaleOp> {
  public:
    using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tosa::RescaleOp consumerOp, PatternRewriter &rewriter) const override {
        auto producerOp = consumerOp.getInput().getDefiningOp<tosa::RescaleOp>();
        if (!producerOp || !producerOp.getOutput().hasOneUse())
            return failure();

        if (!isUnitScale(producerOp) || !unitScaleRebaseCannotClip(producerOp))
            return failure();

        const std::optional<int64_t> producerOutputZp = getScalarInt(producerOp.getOutputZp());
        const std::optional<int64_t> consumerInputZp = getScalarInt(consumerOp.getInputZp());
        if (!producerOutputZp || !consumerInputZp || *producerOutputZp != *consumerInputZp)
            return failure();

        if (producerOp.getOutputUnsigned() != consumerOp.getInputUnsigned())
            return failure();

        rewriter.replaceOpWithNewOp<tosa::RescaleOp>(
            consumerOp, consumerOp.getType(), producerOp.getInput(), consumerOp.getMultiplier(), consumerOp.getShift(),
            producerOp.getInputZp(), consumerOp.getOutputZp(), consumerOp.getScale32Attr(),
            consumerOp.getRoundingModeAttr(), consumerOp.getPerChannelAttr(), producerOp.getInputUnsignedAttr(),
            consumerOp.getOutputUnsignedAttr());
        rewriter.eraseOp(producerOp);
        return success();
    }
};

class TosaRescaleSimplifyPass final : public impl::TosaRescaleSimplifyPassBase<TosaRescaleSimplifyPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<FoldUnitScaleRebaseIntoConsumerPattern, RemoveIdentityRescalePattern>(&getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

} // namespace mlir::model_converter_passes
