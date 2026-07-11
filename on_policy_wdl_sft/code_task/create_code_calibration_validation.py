#!/usr/bin/env python3
"""Create the deterministic 16/16/32 code validation calibration shard."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import pandas as pd

SOURCES=(('HumanEval+',Path('/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet'),16),('MBPP+',Path('/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet'),16),('LiveCodeBench',Path('/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet'),32))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); frames=[]; sources=[]
 for name,path,count in SOURCES:
  df=pd.read_parquet(path); selected=df.iloc[:count].copy(); frames.append(selected); sources.append({'name':name,'path':str(path),'sha256':sha(path),'row_indices':list(range(count)),'row_count':count})
 merged=pd.concat(frames,ignore_index=True); a.output.parent.mkdir(parents=True,exist_ok=True); merged.to_parquet(a.output,index=False)
 manifest={'schema_version':1,'evidence_class':'infrastructure_calibration','output':str(a.output),'output_sha256':sha(a.output),'row_count':len(merged),'sources':sources,'max_response_length':8192}
 a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n'); print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__': main()
