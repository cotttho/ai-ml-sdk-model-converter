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

#include <algorithm>
#include <cstdint>
#include <limits>
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

std::optional<int64_t> applyScaleToBound(int64_t value, int64_t multiplier, int64_t shift, bool doubleRound) {
    if (shift < 1 || shift > 62)
        return std::nullopt;

    __int128 round = __int128{1} << (shift - 1);
    if (doubleRound && shift > 31)
        round += value >= 0 ? (__int128{1} << 30) : -(__int128{1} << 30);

    __int128 result = static_cast<__int128>(value) * static_cast<__int128>(multiplier);
    result += round;
    result >>= shift;

    if (result < std::numeric_limits<int32_t>::min() || result > std::numeric_limits<int32_t>::max())
        return std::nullopt;

    return static_cast<int64_t>(result);
}

std::optional<int64_t> getPowerOfTwoExponent(int64_t value) {
    if (value <= 0 || (value & (value - 1)) != 0)
        return std::nullopt;

    int64_t exponent = 0;
    while (value > 1) {
        value >>= 1;
        ++exponent;
    }
    return exponent;
}

std::optional<int64_t> getExactLeftShift(tosa::RescaleOp rescaleOp) {
    if (rescaleOp.getPerChannel())
        return std::nullopt;

    const std::optional<int64_t> multiplier = getScalarInt(rescaleOp.getMultiplier());
    const std::optional<int64_t> shift = getScalarInt(rescaleOp.getShift());
    if (!multiplier || !shift || *shift < 1 || *shift > 62)
        return std::nullopt;

    const std::optional<int64_t> exponent = getPowerOfTwoExponent(*multiplier);
    if (!exponent || *exponent <= *shift)
        return std::nullopt;

    return *exponent - *shift;
}

bool scalarRescaleCannotClip(tosa::RescaleOp rescaleOp) {
    if (rescaleOp.getPerChannel())
        return false;

    const std::optional<int64_t> multiplier = getScalarInt(rescaleOp.getMultiplier());
    const std::optional<int64_t> shift = getScalarInt(rescaleOp.getShift());
    const std::optional<int64_t> inputZp = getScalarInt(rescaleOp.getInputZp());
    const std::optional<int64_t> outputZp = getScalarInt(rescaleOp.getOutputZp());
    if (!multiplier || !shift || !inputZp || !outputZp)
        return false;

    if (*multiplier < 0)
        return false;

    const std::optional<std::pair<int64_t, int64_t>> inputRange =
        getInterpretedIntegerRange(rescaleOp.getInput(), rescaleOp.getInputUnsigned());
    const std::optional<std::pair<int64_t, int64_t>> outputRange =
        getInterpretedIntegerRange(rescaleOp.getOutput(), rescaleOp.getOutputUnsigned());
    if (!inputRange || !outputRange)
        return false;

    const bool doubleRound = rescaleOp.getScale32() && rescaleOp.getRoundingMode() == tosa::RoundingMode::DOUBLE_ROUND;
    const std::optional<int64_t> scaledLower =
        applyScaleToBound(inputRange->first - *inputZp, *multiplier, *shift, doubleRound);
    const std::optional<int64_t> scaledUpper =
        applyScaleToBound(inputRange->second - *inputZp, *multiplier, *shift, doubleRound);
    if (!scaledLower || !scaledUpper)
        return false;

    const int64_t rescaledMin = std::min(*scaledLower, *scaledUpper) + *outputZp;
    const int64_t rescaledMax = std::max(*scaledLower, *scaledUpper) + *outputZp;
    return rescaledMin >= outputRange->first && rescaledMax <= outputRange->second;
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

Value createScalarShiftConst(PatternRewriter &rewriter, Location loc, Value originalShift, int64_t value) {
    auto shiftType = llvm::cast<ShapedType>(originalShift.getType());
    auto elementType = llvm::cast<IntegerType>(shiftType.getElementType());
    auto shiftAttr = DenseElementsAttr::get(shiftType, rewriter.getIntegerAttr(elementType, value));
    return tosa::ConstOp::create(rewriter, loc, shiftType, shiftAttr);
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

class FoldExactLeftShiftProducerIntoConsumerPattern final : public OpRewritePattern<tosa::RescaleOp> {
  public:
    using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tosa::RescaleOp consumerOp, PatternRewriter &rewriter) const override {
        auto producerOp = consumerOp.getInput().getDefiningOp<tosa::RescaleOp>();
        if (!producerOp || !producerOp.getOutput().hasOneUse())
            return failure();

        if (consumerOp.getPerChannel() || consumerOp.getRoundingMode() != tosa::RoundingMode::SINGLE_ROUND)
            return failure();

        const std::optional<int64_t> producerLeftShift = getExactLeftShift(producerOp);
        const std::optional<int64_t> consumerShift = getScalarInt(consumerOp.getShift());
        if (!producerLeftShift || !consumerShift)
            return failure();

        const int64_t adjustedConsumerShift = *consumerShift - *producerLeftShift;
        if (adjustedConsumerShift < 1 || adjustedConsumerShift > 62)
            return failure();

        if (!scalarRescaleCannotClip(producerOp))
            return failure();

        const std::optional<int64_t> producerOutputZp = getScalarInt(producerOp.getOutputZp());
        const std::optional<int64_t> consumerInputZp = getScalarInt(consumerOp.getInputZp());
        if (!producerOutputZp || !consumerInputZp || *producerOutputZp != *consumerInputZp)
            return failure();

        if (producerOp.getOutputUnsigned() != consumerOp.getInputUnsigned())
            return failure();

        rewriter.setInsertionPoint(consumerOp);
        Value adjustedShift =
            createScalarShiftConst(rewriter, consumerOp.getLoc(), consumerOp.getShift(), adjustedConsumerShift);
        rewriter.replaceOpWithNewOp<tosa::RescaleOp>(
            consumerOp, consumerOp.getType(), producerOp.getInput(), consumerOp.getMultiplier(), adjustedShift,
            producerOp.getInputZp(), consumerOp.getOutputZp(), consumerOp.getScale32Attr(),
            consumerOp.getRoundingModeAttr(), consumerOp.getPerChannelAttr(), producerOp.getInputUnsignedAttr(),
            consumerOp.getOutputUnsignedAttr());
        rewriter.eraseOp(producerOp);
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

class FoldUnitScaleConsumerIntoProducerPattern final : public OpRewritePattern<tosa::RescaleOp> {
  public:
    using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tosa::RescaleOp consumerOp, PatternRewriter &rewriter) const override {
        auto producerOp = consumerOp.getInput().getDefiningOp<tosa::RescaleOp>();
        if (!producerOp || !producerOp.getOutput().hasOneUse())
            return failure();

        if (!isUnitScale(consumerOp) || !unitScaleRebaseCannotClip(consumerOp))
            return failure();

        if (!scalarRescaleCannotClip(producerOp))
            return failure();

        const std::optional<int64_t> producerOutputZp = getScalarInt(producerOp.getOutputZp());
        const std::optional<int64_t> consumerInputZp = getScalarInt(consumerOp.getInputZp());
        if (!producerOutputZp || !consumerInputZp || *producerOutputZp != *consumerInputZp)
            return failure();

        if (producerOp.getOutputUnsigned() != consumerOp.getInputUnsigned())
            return failure();

        rewriter.replaceOpWithNewOp<tosa::RescaleOp>(
            consumerOp, consumerOp.getType(), producerOp.getInput(), producerOp.getMultiplier(), producerOp.getShift(),
            producerOp.getInputZp(), consumerOp.getOutputZp(), producerOp.getScale32Attr(),
            producerOp.getRoundingModeAttr(), producerOp.getPerChannelAttr(), producerOp.getInputUnsignedAttr(),
            consumerOp.getOutputUnsignedAttr());
        rewriter.eraseOp(producerOp);
        return success();
    }
};

class TosaRescaleSimplifyPass final : public impl::TosaRescaleSimplifyPassBase<TosaRescaleSimplifyPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<FoldExactLeftShiftProducerIntoConsumerPattern, FoldUnitScaleConsumerIntoProducerPattern,
                     FoldUnitScaleRebaseIntoConsumerPattern, RemoveIdentityRescalePattern>(&getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

} // namespace mlir::model_converter_passes
