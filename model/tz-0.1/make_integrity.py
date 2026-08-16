#!/usr/bin/env python3
import argparse, base64, hashlib, json
from pathlib import Path

def sri(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return 'sha256-'+base64.b64encode(h.digest()).decode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-dir',default='dist/TZ-0.1-q4f16_1-MLC'); p.add_argument('--wasm',default='dist/TZ-0.1-q4f16_1-webgpu.wasm'); p.add_argument('--out',default='dist/TZ-0.1-integrity.json'); args=p.parse_args()
    root=Path(args.model_dir); tokenizer={}
    for name in ['tokenizer.json','tokenizer.model','tokenizer_config.json']:
        q=root/name
        if q.exists(): tokenizer[name]=sri(q)
    config=root/'mlc-chat-config.json'; wasm=Path(args.wasm)
    result={'config':sri(config) if config.exists() else None,'model_lib':sri(wasm) if wasm.exists() else None,'tokenizer':tokenizer,'onFailure':'error'}
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
