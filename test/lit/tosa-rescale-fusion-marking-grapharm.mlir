//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter --dump-mlir --input %s --output %t.vgf 2>&1 | FileCheck %s

module attributes {tosa.description = "TOSA rescale fusion GraphARM propagation test"} {
  // CHECK-LABEL: IR Dump Before SerializeVGFPass
  // CHECK: vgf.segment @graph_partition_0
  // CHECK-SAME: segment_type = 0 : i32
  // CHECK: spirv.Tosa.Rescale {{[^{}]*}} : (
  // CHECK: spirv.Tosa.Rescale {{[^{}]*}} : (
  // CHECK-NOT: vgf.shader_placeholder
  // CHECK: Successfully saved vgf output
  func.func @main(%arg0: tensor<1x4xi8>) -> tensor<1x4xi32> {
    %mult_shift = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_shift = "tosa.const"() {values = dense<10> : tensor<1xi8>} : () -> tensor<1xi8>
    %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %intermediate_zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %producer = tosa.rescale %arg0, %mult_shift, %shift_shift, %input_zp, %intermediate_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x4xi32>

    %mult_consumer = "tosa.const"() {values = dense<1965721573> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_consumer = "tosa.const"() {values = dense<35> : tensor<1xi8>} : () -> tensor<1xi8>
    %out = tosa.rescale %producer, %mult_consumer, %shift_consumer, %intermediate_zp, %intermediate_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x4xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x4xi32>
    return %out : tensor<1x4xi32>
  }
}
