//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --tosa-rescale-fusion-marking %s | FileCheck %s

module attributes {tosa.description = "TOSA rescale fusion marking test"} {
  // CHECK-LABEL: func.func @left_shift_rounding_candidate(
  // CHECK: %[[PRODUCER:.*]] = tosa.rescale %arg0
  // CHECK-SAME: rescale_fusion_kind = "two_stage_scalar_left_shift_rounding"
  // CHECK-SAME: rescale_fusion_priority = "high"
  // CHECK-SAME: rescale_fusion_role = "producer"
  // CHECK: tosa.rescale %[[PRODUCER]]
  // CHECK-SAME: rescale_fusion_kind = "two_stage_scalar_left_shift_rounding"
  // CHECK-SAME: rescale_fusion_priority = "high"
  // CHECK-SAME: rescale_fusion_role = "consumer"
  func.func @left_shift_rounding_candidate(%arg0: tensor<1x4xi8>) -> tensor<1x4xi32> {
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

  // CHECK-LABEL: func.func @intermediate_clamp_candidate(
  // CHECK: %[[PRODUCER:.*]] = tosa.rescale %arg0
  // CHECK-SAME: rescale_fusion_kind = "two_stage_scalar_with_intermediate_clamp"
  // CHECK-SAME: rescale_fusion_priority = "medium"
  // CHECK-SAME: rescale_fusion_role = "producer"
  // CHECK: tosa.rescale %[[PRODUCER]]
  // CHECK-SAME: rescale_fusion_kind = "two_stage_scalar_with_intermediate_clamp"
  // CHECK-SAME: rescale_fusion_priority = "medium"
  // CHECK-SAME: rescale_fusion_role = "consumer"
  func.func @intermediate_clamp_candidate(%arg0: tensor<1x4xi8>) -> tensor<1x4xi16> {
    %mult_large = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_large = "tosa.const"() {values = dense<23> : tensor<1xi8>} : () -> tensor<1xi8>
    %zero_i8 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %producer = tosa.rescale %arg0, %mult_large, %shift_large, %zero_i8, %zero_i8 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4xi8>

    %mult_half = "tosa.const"() {values = dense<536870912> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_half = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
    %zero_i16 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %out = tosa.rescale %producer, %mult_half, %shift_half, %zero_i8, %zero_i16 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi16>) -> tensor<1x4xi16>
    return %out : tensor<1x4xi16>
  }

  // CHECK-LABEL: func.func @per_channel_candidate(
  // CHECK: %[[PRODUCER:.*]] = tosa.rescale %arg0
  // CHECK-SAME: rescale_fusion_kind = "two_stage_per_channel_backend_analysis"
  // CHECK-SAME: rescale_fusion_priority = "medium"
  // CHECK-SAME: rescale_fusion_role = "producer"
  // CHECK: tosa.rescale %[[PRODUCER]]
  // CHECK-SAME: rescale_fusion_kind = "two_stage_per_channel_backend_analysis"
  // CHECK-SAME: rescale_fusion_priority = "medium"
  // CHECK-SAME: rescale_fusion_role = "consumer"
  func.func @per_channel_candidate(%arg0: tensor<1x2x2x4xi8>) -> tensor<1x2x2x4xi8> {
    %mult_pc = "tosa.const"() {values = dense<[1073741824, 1073741824, 1073741824, 1073741824]> : tensor<4xi32>} : () -> tensor<4xi32>
    %shift_pc = "tosa.const"() {values = dense<[30, 30, 30, 30]> : tensor<4xi8>} : () -> tensor<4xi8>
    %zero_i8 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %producer = tosa.rescale %arg0, %mult_pc, %shift_pc, %zero_i8, %zero_i8 {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x2x2x4xi8>, tensor<4xi32>, tensor<4xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x2x2x4xi8>

    %mult_half = "tosa.const"() {values = dense<536870912> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_half = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
    %out = tosa.rescale %producer, %mult_half, %shift_half, %zero_i8, %zero_i8 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x2x2x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x2x2x4xi8>
    return %out : tensor<1x2x2x4xi8>
  }

  // CHECK-LABEL: func.func @exact_left_shift_fold_candidate_not_marked(
  // CHECK-NOT: rescale_fusion_kind
  func.func @exact_left_shift_fold_candidate_not_marked(%arg0: tensor<1x4xi8>) -> tensor<1x4xi32> {
    %mult_shift = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_shift = "tosa.const"() {values = dense<10> : tensor<1xi8>} : () -> tensor<1xi8>
    %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %intermediate_zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %producer = tosa.rescale %arg0, %mult_shift, %shift_shift, %input_zp, %intermediate_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x4xi32>

    %mult_consumer = "tosa.const"() {values = dense<1237519284> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_consumer = "tosa.const"() {values = dense<51> : tensor<1xi8>} : () -> tensor<1xi8>
    %out = tosa.rescale %producer, %mult_consumer, %shift_consumer, %intermediate_zp, %intermediate_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x4xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x4xi32>
    return %out : tensor<1x4xi32>
  }
}
