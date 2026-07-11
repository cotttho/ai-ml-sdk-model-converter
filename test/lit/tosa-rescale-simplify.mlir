//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --tosa-rescale-simplify %s | FileCheck %s

module attributes {tosa.description = "TOSA rescale simplify test"} {
  // CHECK-LABEL: func.func @main(
  // CHECK-NOT: dense<1073741824>
  // CHECK: %[[MULT:.*]] = "tosa.const"() <{values = dense<536870912> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: %[[ZP:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: tosa.rescale %arg0, %[[MULT]], %[[SHIFT]], %[[ZP]], %[[ZP]]
  // CHECK-NOT: tosa.rescale
  func.func @main(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
    %mult_identity = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_identity = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
    %zp = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %identity = tosa.rescale %arg0, %mult_identity, %shift_identity, %zp, %zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4xi8>

    %mult_half = "tosa.const"() {values = dense<536870912> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_half = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
    %non_identity = tosa.rescale %identity, %mult_half, %shift_half, %zp, %zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4xi8>
    return %non_identity : tensor<1x4xi8>
  }
}
