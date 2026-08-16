#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    p=argparse.ArgumentParser(); p.add_argument('--base',default='Qwen/Qwen3-0.6B'); p.add_argument('--adapter',default='dist/tz-0.1-adapter'); p.add_argument('--out',default='dist/TZ-0.1-HF'); args=p.parse_args()
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    base=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=dtype,device_map='auto' if torch.cuda.is_available() else None,trust_remote_code=False)
    model=PeftModel.from_pretrained(base,args.adapter).merge_and_unload(); tokenizer=AutoTokenizer.from_pretrained(args.adapter,use_fast=True,trust_remote_code=False)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); model.save_pretrained(out,safe_serialization=True,max_shard_size='2GB'); tokenizer.save_pretrained(out)
    notice={'model':'TZ-0.1','derived_from':args.base,'base_license':'Apache-2.0','note':'Retain upstream Apache-2.0 license and required notices when distributing this derivative.'}; (out/'TZ-NOTICE.json').write_text(json.dumps(notice,indent=2),encoding='utf-8')
if __name__=='__main__': main()
