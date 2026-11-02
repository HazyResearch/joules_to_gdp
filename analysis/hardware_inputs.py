"""Normalize recorded physical request and batch intervals for hardware costing."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from build_pass1_primary_results import ROOT, load_module

def jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(s) for s in handle if s.strip()]


def batch_record(key,b):
    return dict(batch_key=key,batch_start_s=b['batch_prompt_start_time_s'],batch_end_s=b['batch_prompt_end_time_s'],batch_energy_j=b['batch_gpu_energy_joules'],physical_request_count=b['batch_size'])


def request_record(key,t,task,repeat):
    return dict(batch_key=key,physical_request_id=key+':'+str(t['query_id']),task_id=str(task),repeat=int(repeat),start_s=t['prompt_start_time_s'],end_s=t['prompt_end_time_s'])


def available_inputs(p):
    data={}
    q=pd.read_csv(p.SOURCE_PATHS['glm_query']).query("dataset == 'GDPVal'")
    b=pd.read_csv(p.SOURCE_PATHS['glm_batch'])
    requests=pd.DataFrame([dict(batch_key=r.batch_key,physical_request_id=r.query_id,task_id=r.judge_task_id,repeat=r.repeat_index,start_s=r.prompt_start_time_s,end_s=r.prompt_end_time_s) for r in q.itertuples()])
    batches=pd.DataFrame([dict(batch_key=r.batch_key,batch_start_s=r.prompt_start_time_s,batch_end_s=r.prompt_end_time_s,batch_energy_j=r.gpu_energy_joules,physical_request_count=r.attempts) for r in b.itertuples()])
    data['glm52_h200']=(requests,batches)
    h100=load_module('migration_h100',ROOT/'cluster/h100_qwen_metrics.py')
    traces=jsonl(h100.TRACE_PATH)
    bs=[]
    for path in h100.BATCH_PATHS:
        name=path.parents[3].name
        bs += [batch_record(name+':'+b['batch_id'],b) for b in jsonl(path)]
    rs=[request_record(t['source_name']+':'+t['batch_id'],t,t['task_id'],t['repeat_index']) for t in traces]
    data['qwen_h100']=(pd.DataFrame(rs),pd.DataFrame(bs))
    directory=ROOT/'outputs/019fde63-1f71-7461-a23f-2f017c2c95cd/qwen36_h200_paper_five_metrics_20260815/inputs'
    traces=jsonl(directory/'canonical_traces.jsonl')
    bs=[]
    for path in sorted(directory.glob('*.batches.jsonl')):
        name=path.name.removesuffix('.batches.jsonl')
        bs += [batch_record(name+':'+b['batch_id'],b) for b in jsonl(path)]
    rs=[request_record(t['source_name']+':'+t['batch_id'],t,t['task_id'],t['repeat_index']) for t in traces]
    data['qwen_h200']=(pd.DataFrame(rs),pd.DataFrame(bs))
    directory=ROOT/'outputs/019fde63-1f71-7461-a23f-2f017c2c95cd/qwen36_b200_gdp_structured_value_energy_20260815/inputs'
    traces={(int(t['batch_index']),int(t['batch_position'])):t for t in jsonl(directory/'traces.jsonl')}
    manifest=[m for m in jsonl(directory/'comparison_manifest.unblinded.jsonl') if m['condition']=='without_rubric' and m['baseline']=='human']
    bs=[batch_record(b['batch_id'],b) for b in jsonl(directory/'batches.jsonl')]
    rs=[]
    for m in manifest:
        t=traces[divmod(int(m['source_query_id'][1:]),20)]
        rs.append(request_record(t['batch_id'],t,m['task_id'],m['repeat']))
    data['qwen_b200']=(pd.DataFrame(rs),pd.DataFrame(bs))
    return data
