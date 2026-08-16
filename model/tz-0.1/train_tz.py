#!/usr/bin/env python3
import argparse, json, os, random
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

SYSTEM='You are TZ, a fast local-first assistant. Give direct, useful answers. Prefer concise answers unless detail is requested. Never claim cloud access, browsing, APIs, or external tools unless the session explicitly says they are active. Protect user privacy and clearly distinguish local execution from network-backed services.'

def read_jsonl(path: Path):
    rows=[]
    with path.open('r',encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            obj=json.loads(line)
            if not isinstance(obj.get('messages'),list): raise ValueError(f'line {n}: messages must be a list')
            rows.append(obj)
    if not rows: raise ValueError('dataset is empty')
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',default='Qwen/Qwen3-0.6B')
    p.add_argument('--data',default='model/tz-0.1/data/tz_sft.jsonl')
    p.add_argument('--out',default='dist/tz-0.1-adapter')
    p.add_argument('--epochs',type=float,default=3.0)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-seq',type=int,default=2048)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--grad-accum',type=int,default=8)
    args=p.parse_args()
    set_seed(args.seed); random.seed(args.seed)
    tokenizer=AutoTokenizer.from_pretrained(args.base,use_fast=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    def normalize(row):
        msgs=row['messages']
        if not msgs or msgs[0].get('role')!='system': msgs=[{'role':'system','content':SYSTEM}]+msgs
        text=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
        return {'text':text}
    rows=[normalize(r) for r in read_jsonl(Path(args.data))]
    ds=Dataset.from_list(rows).shuffle(seed=args.seed)
    use_cuda=torch.cuda.is_available()
    compute=torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=compute) if use_cuda else None
    model=AutoModelForCausalLM.from_pretrained(args.base,device_map='auto' if use_cuda else None,torch_dtype=compute if use_cuda else torch.float32,quantization_config=quant,trust_remote_code=False)
    model.config.use_cache=False
    lora=LoraConfig(r=32,lora_alpha=64,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    cfg=SFTConfig(output_dir=args.out,num_train_epochs=args.epochs,learning_rate=args.lr,per_device_train_batch_size=args.batch,gradient_accumulation_steps=args.grad_accum,warmup_ratio=0.05,logging_steps=5,save_strategy='epoch',max_seq_length=args.max_seq,dataset_text_field='text',packing=True,fp16=bool(use_cuda and compute==torch.float16),bf16=bool(use_cuda and compute==torch.bfloat16),gradient_checkpointing=True,report_to='none',seed=args.seed)
    trainer=SFTTrainer(model=model,args=cfg,train_dataset=ds,processing_class=tokenizer,peft_config=lora)
    trainer.train(); trainer.save_model(args.out); tokenizer.save_pretrained(args.out)
    meta={'name':'TZ-0.1','base':args.base,'dataset':str(args.data),'examples':len(rows),'seed':args.seed,'method':'QLoRA SFT','system_prompt':SYSTEM}
    Path(args.out).mkdir(parents=True,exist_ok=True); (Path(args.out)/'tz-training.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
if __name__=='__main__': main()
