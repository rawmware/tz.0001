# TZ-0.1 model build

This directory is the reproducible build path for a real TZ checkpoint. The target base is `Qwen/Qwen3-0.6B` (Apache-2.0). TZ-specific training data, adapters, merged weights, quantized MLC weights and WebGPU libraries are separate artifacts.

## 1. Train
Use a CUDA machine for practical training. Create a virtual environment, install `requirements.txt`, then run:

`python model/tz-0.1/train_tz.py --data model/tz-0.1/data/tz_sft.jsonl --out dist/tz-0.1-adapter`

The included JSONL is only a smoke-test seed. A production checkpoint needs a substantially larger, reviewed instruction dataset with evaluation splits and license/provenance records.

## 2. Merge adapter
`python model/tz-0.1/merge_adapter.py --adapter dist/tz-0.1-adapter --out dist/TZ-0.1-HF`

## 3. Compile for WebGPU
MLC requires a source build with its WASM/WebGPU toolchain. After that environment is installed:

`bash model/tz-0.1/build_webgpu.sh dist/TZ-0.1-HF dist/TZ-0.1-q4f16_1-MLC dist/TZ-0.1-q4f16_1-webgpu.wasm`

Defaults are q4f16_1, a 2048-token context window and 256-token prefill chunks to keep the phone memory target conservative.

## 4. Integrity manifest
`python model/tz-0.1/make_integrity.py`

Use the resulting hashes in the WebLLM `ModelRecord.integrity` field with `onFailure: error` before promoting TZ-0.1 to production.

## Distribution
Apache-2.0 permits modification and redistribution, including commercial derivatives, but upstream license and notice obligations still apply. Do not describe unmodified upstream weights as exclusively owned by TZ. TZ-specific code, training data you own, adapters and modifications can have their own terms subject to upstream obligations.
