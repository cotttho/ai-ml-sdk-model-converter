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

  // CHECK-LABEL: func.func @unit_rebase_fold(
  // CHECK-NOT: dense<1073741824>
  // CHECK: %[[MULT:.*]] = "tosa.const"() <{values = dense<536870912> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: %[[SIGNED_ZP:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: %[[OUTPUT_ZP:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: tosa.rescale %arg0, %[[MULT]], %[[SHIFT]], %[[SIGNED_ZP]], %[[OUTPUT_ZP]]
  // CHECK-SAME: input_unsigned = false
  // CHECK-SAME: output_unsigned = false
  // CHECK-NOT: tosa.rescale
  func.func @unit_rebase_fold(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
    %mult_identity = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_identity = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
    %signed_zp = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %unsigned_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %rebased = tosa.rescale %arg0, %mult_identity, %shift_identity, %signed_zp, %unsigned_zp {input_unsigned = false, output_unsigned = true, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4xi8>

    %mult_half = "tosa.const"() {values = dense<536870912> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_half = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
    %out = tosa.rescale %rebased, %mult_half, %shift_half, %unsigned_zp, %unsigned_zp {input_unsigned = true, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4xi8>
    return %out : tensor<1x4xi8>
  }

  // CHECK-LABEL: func.func @unit_consumer_rebase_fold(
  // CHECK-NOT: dense<1073741824>
  // CHECK-DAG: %[[MULT:.*]] = "tosa.const"() <{values = dense<268435456> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[INPUT_ZP:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[OUTPUT_ZP:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: tosa.rescale %arg0, %[[MULT]], %[[SHIFT]], %[[INPUT_ZP]], %[[OUTPUT_ZP]]
  // CHECK-SAME: output_unsigned = true
  // CHECK-NOT: tosa.rescale
  func.func @unit_consumer_rebase_fold(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
    %mult_quarter = "tosa.const"() {values = dense<268435456> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_quarter = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
    %input_zp = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %producer_zp = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %producer = tosa.rescale %arg0, %mult_quarter, %shift_quarter, %input_zp, %producer_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4xi8>

    %mult_identity = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_identity = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
    %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %out = tosa.rescale %producer, %mult_identity, %shift_identity, %producer_zp, %output_zp {input_unsigned = false, output_unsigned = true, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4xi8>
    return %out : tensor<1x4xi8>
  }

  // CHECK-LABEL: func.func @unit_consumer_rebase_keeps_clipping_producer(
  // CHECK: tosa.rescale
  // CHECK: tosa.rescale
  func.func @unit_consumer_rebase_keeps_clipping_producer(%arg0: tensor<1x4xi8>) -> tensor<1x4xi16> {
    %mult_large = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_large = "tosa.const"() {values = dense<23> : tensor<1xi8>} : () -> tensor<1xi8>
    %zero_i8 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %producer = tosa.rescale %arg0, %mult_large, %shift_large, %zero_i8, %zero_i8 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4xi8>

    %mult_identity = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %shift_identity = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
    %zero_i16 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %out = tosa.rescale %producer, %mult_identity, %shift_identity, %zero_i8, %zero_i16 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi16>) -> tensor<1x4xi16>
    return %out : tensor<1x4xi16>
  }

  // CHECK-LABEL: func.func @left_shift_producer_fold(
  // CHECK-NOT: dense<1073741824>
  // CHECK-DAG: %[[MULT:.*]] = "tosa.const"() <{values = dense<1237519284> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<31> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[INPUT_ZP:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[OUTPUT_ZP:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: tosa.rescale %arg0, %[[MULT]], %[[SHIFT]], %[[INPUT_ZP]], %[[OUTPUT_ZP]]
  // CHECK-SAME: input_unsigned = false
  // CHECK-SAME: output_unsigned = false
  // CHECK-NOT: tosa.rescale
  func.func @left_shift_producer_fold(%arg0: tensor<1x4xi8>) -> tensor<1x4xi32> {
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

  // CHECK-LABEL: func.func @left_shift_producer_rounding_sensitive_keep(
  // CHECK-DAG: %[[PROD_MULT:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[PROD_SHIFT:.*]] = "tosa.const"() <{values = dense<10> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[CONS_MULT:.*]] = "tosa.const"() <{values = dense<1965721573> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[CONS_SHIFT:.*]] = "tosa.const"() <{values = dense<35> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[INPUT_ZP:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[INTERMEDIATE_ZP:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[PRODUCER:.*]] = tosa.rescale %arg0, %[[PROD_MULT]], %[[PROD_SHIFT]], %[[INPUT_ZP]], %[[INTERMEDIATE_ZP]]
  // CHECK: tosa.rescale %[[PRODUCER]], %[[CONS_MULT]], %[[CONS_SHIFT]], %[[INTERMEDIATE_ZP]], %[[INTERMEDIATE_ZP]]
  func.func @left_shift_producer_rounding_sensitive_keep(%arg0: tensor<1x4xi8>) -> tensor<1x4xi32> {
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

  // CHECK-LABEL: func.func @identity_clamp_remove(
  // CHECK-NOT: tosa.clamp
  func.func @identity_clamp_remove(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
    %out = tosa.clamp %arg0 {max_val = 127 : i8, min_val = -128 : i8} : (tensor<1x4xi8>) -> tensor<1x4xi8>
    return %out : tensor<1x4xi8>
  }

  // CHECK-LABEL: func.func @activation_clamp_keep(
  // CHECK: tosa.clamp
  func.func @activation_clamp_keep(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
    %out = tosa.clamp %arg0 {max_val = 127 : i8, min_val = 0 : i8} : (tensor<1x4xi8>) -> tensor<1x4xi8>
    return %out : tensor<1x4xi8>
  }
}
