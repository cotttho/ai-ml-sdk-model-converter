//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --assign-grapharm-interface-var-abi %s | FileCheck %s

vgf.sequence @main(%arg0: tensor<1xi8>) -> tensor<1xi8> {
  vgf.segment @single_partition_0(%arg1: tensor<1xi8>) -> tensor<1xi8> attributes {segment_type = 0 : i32} {
    spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
      // CHECK-LABEL: spirv.ARM.Graph @single_partition_0
      // CHECK: spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>
      // CHECK: spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>
      spirv.ARM.Graph @single_partition_0(%arg2: !spirv.arm.tensor<1xi8>) -> !spirv.arm.tensor<1xi8> attributes {entry_point = true} {
        spirv.ARM.GraphOutputs %arg2 : !spirv.arm.tensor<1xi8>
      }
    }
    vgf.segment_output
  }
  %0 = vgf.run_segment @single_partition_0 :(%arg0) (tensor<1xi8>) -> tensor<1xi8>
  vgf.sequence_output %0 : tensor<1xi8>
}

vgf.sequence @multi_segment(%arg0: tensor<1xi8>) -> tensor<1xi8> {
  vgf.segment @multi_partition_0(%arg1: tensor<1xi8>) -> tensor<1xi8> attributes {segment_type = 0 : i32} {
    spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
      // CHECK-LABEL: spirv.ARM.Graph @multi_partition_0
      // CHECK: spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>
      // CHECK: spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>
      spirv.ARM.Graph @multi_partition_0(%arg2: !spirv.arm.tensor<1xi8>) -> !spirv.arm.tensor<1xi8> attributes {entry_point = true} {
        spirv.ARM.GraphOutputs %arg2 : !spirv.arm.tensor<1xi8>
      }
    }
    vgf.segment_output
  }
  %0 = vgf.run_segment @multi_partition_0 :(%arg0) (tensor<1xi8>) -> tensor<1xi8>
  vgf.segment @multi_partition_1(%arg1: tensor<1xi8>) -> tensor<1xi8> attributes {segment_type = 0 : i32} {
    spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
      // CHECK-LABEL: spirv.ARM.Graph @multi_partition_1
      // CHECK: spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>
      // CHECK: spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>
      spirv.ARM.Graph @multi_partition_1(%arg2: !spirv.arm.tensor<1xi8>) -> !spirv.arm.tensor<1xi8> attributes {entry_point = true} {
        spirv.ARM.GraphOutputs %arg2 : !spirv.arm.tensor<1xi8>
      }
    }
    vgf.segment_output
  }
  %1 = vgf.run_segment @multi_partition_1 :(%0) (tensor<1xi8>) -> tensor<1xi8>
  vgf.sequence_output %1 : tensor<1xi8>
}
