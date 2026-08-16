#!/usr/bin/env bash
set -euo pipefail
HF_MODEL="${1:-dist/TZ-0.1-HF}"
OUT="${2:-dist/TZ-0.1-q4f16_1-MLC}"
LIB="${3:-dist/TZ-0.1-q4f16_1-webgpu.wasm}"
QUANT="${TZ_QUANT:-q4f16_1}"
CTX="${TZ_CONTEXT:-2048}"
PREFILL="${TZ_PREFILL_CHUNK:-256}"
command -v mlc_llm >/dev/null || { echo 'mlc_llm not found. Build/install MLC LLM from source with the WebGPU/WASM environment first.' >&2; exit 2; }
mkdir -p "$OUT" "$(dirname "$LIB")"
mlc_llm convert_weight "$HF_MODEL" --quantization "$QUANT" -o "$OUT"
mlc_llm gen_config "$HF_MODEL" --quantization "$QUANT" --context-window-size "$CTX" --prefill-chunk-size "$PREFILL" -o "$OUT"
mlc_llm compile "$OUT/mlc-chat-config.json" --device webgpu -o "$LIB"
echo "MLC weights: $OUT"
echo "WebGPU library: $LIB"
