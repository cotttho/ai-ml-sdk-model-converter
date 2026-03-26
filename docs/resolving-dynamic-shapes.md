# Resolving Dynamic Shapes in TFLite Model Conversion

Some TFLite models (e.g. YOLOv4) contain dynamic dimensions — typically a variable batch size or a data-dependent output dimension like the number of detections. The TOSA specification requires static shapes, and the VGF serializer will crash if any tensor dimension is unresolved.

This document describes the current manual workflow and a proposal to automate it.

## Pipeline Overview

```
TFLite (.tflite) → tosa-converter-for-tflite → TOSA MLIR → model-converter → VGF
```

Dynamic dimensions can appear at two points:

| Source | Example | Resolvable by shape inference? |
|--------|---------|-------------------------------|
| Variable input dims (batch) | `?x416x416x3` | Yes, if pinned to a concrete value |
| Data-dependent output dims | `1x?x4` (detection count) | No — depends on runtime values |

## Current Manual Workflow

### Step 1: Pin Input Shapes

Use `tosa-converter-for-tflite` with `override_tflite_input_shape`:

```python
from _tosa_converter_for_tflite_wrapper import (
    _tflite_flatbuffer_to_tosa_mlir,
    TosaConverterOutputFormat,
    TosaConverterDebugInfo,
)

_tflite_flatbuffer_to_tosa_mlir(
    "model.tflite",
    "model.tosa.mlir",
    TosaConverterOutputFormat.Text,
    TosaConverterDebugInfo.Disabled,
    {"input_1": [1, 416, 416, 3]},
)
```

> Requires a Python venv matching the wheel version:
> ```
> python3.12 -m venv /tmp/tosa_venv
> /tmp/tosa_venv/bin/pip install ~/tosa-converter-for-tflite/dist/*.whl
> ```

### Step 2: Fix Remaining Dynamic Output Dimensions

Find the actual output shapes from TFLite:

```python
import tensorflow.lite as tflite
interp = tflite.Interpreter(model_path="model.tflite")
interp.allocate_tensors()
for o in interp.get_output_details():
    print(f"{o['name']}: shape={o['shape']}")
```

Replace `?` in the TOSA MLIR with the concrete values:

```python
with open("model.tosa.mlir") as f:
    content = f.read()

content = content.replace("tensor<1x?x4xui8>",  "tensor<1x2535x4xui8>")
content = content.replace("tensor<1x?x80xui8>", "tensor<1x2535x80xui8>")
content = content.replace("tensor<1x?x4xi8>",   "tensor<1x2535x4xi8>")
content = content.replace("tensor<1x?x80xi8>",  "tensor<1x2535x80xi8>")

with open("model_fixed.tosa.mlir", "w") as f:
    f.write(content)
```

### Step 3: Convert to VGF

```bash
model-converter -i model_fixed.tosa.mlir -o model.vgf
```

### Verification

```bash
# Extract SPIR-V and check for unshaped tensor types
spirv-dis /tmp/graph.spv | grep OpTypeTensorARM
# Every line should have 3 operands (elem_type, rank, shape_constant)

# Check for INT64_MIN sentinels in VGF dump
grep -- "-9223372036854775808" model_dump.json
```

## Proposal: Automate in the Model-Converter

The model-converter is the right place to automate this because it's the final step before VGF serialization and already has the infrastructure (`--require-static-shape`, `TosaShapedVerificationPass`).

### Change 1: Add `TosaInferShapesPass` to the pipeline

The model-converter does not currently run shape inference. Adding it resolves all shapes that are statically determinable from the input:

```cpp
// In compiler.cpp SetPassManager(), after createTosaConvertIntegerTypeToSignless:
{
    OpPassManager &funcNestedPM = _pm.nest<func::FuncOp>();
    funcNestedPM.addPass(mlir::tosa::createTosaConvertIntegerTypeToSignless());
    funcNestedPM.addPass(createDenseResourceInlinerPass());
    funcNestedPM.addPass(mlir::tosa::createTosaInferShapesPass());  // <-- ADD THIS
}
```

This is safe to run unconditionally — it's a no-op when all shapes are already static (like EfficientDet).

### Change 2: Add `--shape-override` CLI option

The `TosaShapedVerificationPass` already hints at this flag in its error message. Implement it to override input tensor dimensions before shape inference runs:

```
model-converter -i model.tosa.mlir -o model.vgf \
    --shape-override "input_1=1,416,416,3"
```

This would:
1. Parse the override string into a map of `{tensor_name → dimensions}`
2. Run a pass (similar to `tosa-converter-for-tflite`'s `OverrideTFLiteInputShape`) that updates the function signature and block argument types
3. Then `TosaInferShapesPass` propagates the concrete shapes through the graph

### Change 3: Resolve data-dependent output dims from SPIR-V

After `TosaToSPIRV` conversion, the internal SPIR-V operations have fully resolved shapes even when the MLIR function signature still has `?` dims. For example, the final RESCALE produces `!spirv.arm.tensor<1x2535x4xi8>` but the output is cast back to `tensor<1x?x4xi8>`.

A post-SPIRV-conversion pass could:
1. Walk the graph outputs and find the concrete shapes from the SPIR-V tensor types
2. Update the `tensor.cast` ops and function return types to use the resolved shapes
3. Eliminate the dynamic dims before VGF serialization

This would handle the YOLOv4 case fully automatically without any user intervention.

### Priority

| Change | Effort | Impact |
|--------|--------|--------|
| 1. Add `TosaInferShapesPass` | Minimal (one line) | Resolves most dynamic dims when inputs are static |
| 2. `--shape-override` | Medium (new pass + CLI) | Handles dynamic input dims (batch size) |
| 3. Resolve from SPIR-V | Medium (new pass) | Handles data-dependent output dims |

Change 1 alone would have fixed the YOLOv4 model if the TOSA MLIR had been produced with `override_tflite_input_shape` (step 1 of the manual workflow). Changes 2+3 together would make the entire workflow automatic.

## Example: YOLOv4

| Tensor | Original | After override | After inference |
|--------|----------|---------------|-----------------|
| input_1 | `?x416x416x3` | `1x416x416x3` | `1x416x416x3` |
| Identity (boxes) | `1x?x4` | `1x?x4` | `1x2535x4` (via SPIR-V) |
| Identity_1 (scores) | `1x?x80` | `1x?x80` | `1x2535x80` (via SPIR-V) |

## Emulation Layer Fallback

The emulation layer includes runtime shape inference (`shape_inference.hpp`) that handles models with rank-only tensor types. This serves as a safety net but static shapes are preferred — they produce smaller SPIR-V, avoid sentinel dimension handling, and eliminate inference errors at runtime.
