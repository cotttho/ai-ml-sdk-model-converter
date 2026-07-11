/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::model_converter_passes {
#define GEN_PASS_DEF_ASSIGNGRAPHARMINTERFACEVARABIPASS
#include "passes.hpp.inc"
namespace {

constexpr uint32_t graphARMDescriptorSet = 0;

class AssignGraphARMInterfaceVarABIPass
    : public impl::AssignGraphARMInterfaceVarABIPassBase<AssignGraphARMInterfaceVarABIPass> {
  public:
    using impl::AssignGraphARMInterfaceVarABIPassBase<
        AssignGraphARMInterfaceVarABIPass>::AssignGraphARMInterfaceVarABIPassBase;

    void runOnOperation() override {
        if (failed(assignInterfaceVarABI(getOperation()))) {
            signalPassFailure();
        }
    }

  private:
    uint32_t getBindingId(const DenseMap<Value, uint32_t> &bindingIds, Value value) {
        auto it = bindingIds.find(value);
        assert(it != bindingIds.end() && "expected binding for graph interface value");
        return it->second;
    }

    spirv::InterfaceVarABIAttr getInterfaceVarABIAttr(uint32_t bindingId) {
        return spirv::InterfaceVarABIAttr::get(graphARMDescriptorSet, bindingId, std::nullopt, &getContext());
    }

    LogicalResult assignInterfaceVarABI(vgf::SequenceOp sequenceOp) {
        auto isSequenceInputOperand = [&](Value operand) {
            return llvm::is_contained(sequenceOp.getArguments(), operand);
        };

        Operation *sequenceOutputOp = sequenceOp.front().getTerminator();
        auto isSequenceOutputOperand = [&](Value operand) {
            return llvm::is_contained(sequenceOutputOp->getOperands(), operand);
        };

        DenseMap<Value, uint32_t> bindingIds;
        auto assignBindingIfUnset = [&](Value operand, uint32_t bindingId) {
            bindingIds.try_emplace(operand, bindingId);
        };

        // First: Resolve sequence inputs.
        for (BlockArgument operand : sequenceOp.getArguments()) {
            sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
                Operation *runSegmentOp = segmentOp->getNextNode();
                if (runSegmentOp && llvm::is_contained(runSegmentOp->getOperands(), operand)) {
                    assignBindingIfUnset(operand, static_cast<uint32_t>(operand.getArgNumber()));
                }
            });
        }

        // Second: Resolve intermediate values.
        auto bindingId = static_cast<uint32_t>(sequenceOp.getNumArguments());
        WalkResult sequenceWalkResult = sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
            Operation *runSegmentOp = segmentOp->getNextNode();
            if (!runSegmentOp) {
                return WalkResult::advance();
            }

            for (Value operand : runSegmentOp->getOperands()) {
                if (!isSequenceInputOperand(operand) && !bindingIds.contains(operand)) {
                    bindingIds[operand] = bindingId++;
                }
            }
            for (Value result : runSegmentOp->getResults()) {
                if (!isSequenceOutputOperand(result) && !bindingIds.contains(result)) {
                    bindingIds[result] = bindingId++;
                }
            }
            return WalkResult::advance();
        });
        if (sequenceWalkResult.wasInterrupted()) {
            return failure();
        }

        // Third: Resolve sequence outputs.
        for (Value operand : sequenceOutputOp->getOperands()) {
            sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
                Operation *runSegmentOp = segmentOp->getNextNode();
                if (runSegmentOp && llvm::is_contained(runSegmentOp->getResults(), operand) &&
                    !bindingIds.contains(operand)) {
                    bindingIds[operand] = bindingId++;
                }
            });
        }

        sequenceWalkResult = sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
            if (segmentOp.getSegmentType() != vgf::SegmentTypeEnum::GRAPH) {
                return WalkResult::advance();
            }

            Operation *runSegmentOp = segmentOp->getNextNode();
            if (!runSegmentOp) {
                segmentOp.emitError("expected graph segment to be followed by a segment run op");
                return WalkResult::interrupt();
            }

            SmallVector<spirv::GraphARMOp, 1> graphOps;
            segmentOp.walk([&](spirv::GraphARMOp graphOp) { graphOps.push_back(graphOp); });
            if (graphOps.empty()) {
                segmentOp.emitError("expected graph segment to contain a spirv.ARM.Graph");
                return WalkResult::interrupt();
            }
            if (graphOps.size() != 1) {
                segmentOp.emitError("expected graph segment to contain a single spirv.ARM.Graph");
                return WalkResult::interrupt();
            }
            spirv::GraphARMOp graphOp = graphOps.front();

            if (runSegmentOp->getNumOperands() != graphOp.getNumArguments()) {
                graphOp.emitError("segment run operand count does not match graph arguments");
                return WalkResult::interrupt();
            }
            if (runSegmentOp->getNumResults() != graphOp.getNumResults()) {
                graphOp.emitError("segment run result count does not match graph results");
                return WalkResult::interrupt();
            }

            for (auto [argIndex, operand] : llvm::enumerate(runSegmentOp->getOperands())) {
                graphOp.setArgAttr(static_cast<unsigned>(argIndex), spirv::getInterfaceVarABIAttrName(),
                                  getInterfaceVarABIAttr(getBindingId(bindingIds, operand)));
            }

            for (auto [resultIndex, result] : llvm::enumerate(runSegmentOp->getResults())) {
                graphOp.setResultAttr(static_cast<unsigned>(resultIndex), spirv::getInterfaceVarABIAttrName(),
                                     getInterfaceVarABIAttr(getBindingId(bindingIds, result)));
            }

            return WalkResult::advance();
        });

        return sequenceWalkResult.wasInterrupted() ? failure() : success();
    }
};

} // namespace
} // namespace mlir::model_converter_passes
