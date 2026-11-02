"""Reproducible NVJ/TSJ/NVD numerical overlay.

In total mode the input builder supplies validated GPU+CPU+DRAM energy for all
models and hardware; the GLM headline uses active-request attribution. Serving
costs use duration-based request allocations and the selected reference tariffs.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import build_pass1_primary_results as primary
from rental_cost import occupied_shares, glm_prompt_window_costs, OTHER_MODEL_TARIFF

ROOT = primary.ROOT
OUT = ROOT / 'reports/paper_net_metrics_20260919'
FIELDS = ['generated_value_usd', 'rental_cost_usd', 'net_value_usd', 'energy_mj',
          'hours_saved', 'nvj', 'tsj', 'nvd']


def jl(path):
    with Path(path).open() as handle:
        return [json.loads(x) for x in handle if x.strip()]


def jwrite(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, default=str)+'\n')


def local_source_path(archived_path):
    """Resolve a frozen absolute archive path inside this checkout."""
    raw = str(archived_path)
    marker = '/reports/'
    if marker not in raw:
        raise ValueError(f'archive path does not contain {marker}: {raw}')
    return ROOT / ('reports/' + raw.split(marker, 1)[1])


def costs_for_other_model(metrics_module, model):
    cfg = metrics_module.MODEL_CONFIGS[model]
    sources = json.loads(cfg['source_outputs'].read_text())
    stage = json.loads((cfg['source_outputs'].parent/'source_stage_summary.json').read_text())
    inventory = json.loads((cfg['source_outputs'].parent/'frozen_source_inventory.json').read_text())
    hashes = {r['path']: r['sha256'] for r in inventory['files'] if r['role']=='canonical_metadata'}
    rows = []
    for s in sources:
        path = local_source_path(s['metadata_path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(s['metadata_path'])]
        m = json.loads(path.read_text())
        q = int(s['query_id'][1:])
        rows.append(dict(task_id=str(s['task_id']), repeat=int(s['repeat']),
            query_id=s['query_id'], source=s['run_name'], batch_index=q//20,
            latency_s=float(m['wall_clock_s']), metadata_path=str(path.relative_to(ROOT)),
            hf_repo=stage['hf_repo_id'], hf_revision=stage['hf_revision']))
    a = pd.DataFrame(rows)
    a['batch_key'] = a.source+':'+a.batch_index.astype(str)
    assert len(a)==880 and a.groupby('batch_key').size().eq(20).all()
    a['allocated_seconds'] = np.nan
    a['timing_method'] = 'duration-based allocation from task metadata'
    audit = []
    for key, g in a.groupby('batch_key'):
        end = g.latency_s.to_numpy()
        shares, idle = occupied_shares(np.zeros(len(g)), end, 0., float(end.max()))
        duration = float(end.max())
        if model=='gemma4_31b':
            # Batch terminal offsets plus measured durations recover launch times.
            shard = 'rows-000-459' if '000-459' in g.source.iloc[0] else 'rows-460-879'
            bs = jl(ROOT/'reports/gemma4_31b_norubric_major_category_value_energy_20260725/source_telemetry'/shard/'batches.jsonl')
            b = next(b for b in bs if int(b['batch_index'])==int(g.batch_index.iloc[0]))
            ends = np.array([b['prompt_finish_offsets_s'][q] for q in g.query_id])
            starts = ends-end
            assert starts.min()>-0.05 and starts.max()<1, (key, starts.min(), starts.max())
            starts = np.maximum(starts, 0.)
            duration = b['batch_prompt_end_time_s']-b['batch_prompt_start_time_s']
            shares, idle = occupied_shares(starts, np.minimum(ends,duration), 0., duration)
            a.loc[g.index,'timing_method'] = 'batch terminal offsets minus measured HF run duration'
        shares += idle*shares/shares.sum()
        a.loc[g.index,'allocated_seconds'] = shares
        audit.append(dict(model=model,batch_key=key,charged_seconds=duration,
            allocated_seconds=float(shares.sum()),timing_method=a.loc[g.index,'timing_method'].iloc[0]))
    # Bundled AWS p5.48xlarge: all eight H100s, 192 vCPU and 2048 GiB RAM.
    tariff=json.loads(OTHER_MODEL_TARIFF.read_text())
    rate=float(next(r for r in tariff['rows'] if r['Instance Type']=='p5.48xlarge')['PricePerUnit'])
    # Warm local NVMe model cache; explicit 32-GiB boot volume and one IPv4.
    hourly=rate+32*0.08/744+0.005
    # One delivery of each final artifact; retain failures' costs as well.
    manifest=jl(cfg['source_outputs'].parent/'comparison_manifest.unblinded.jsonl')
    manifest=[r for r in manifest if r['baseline']=='human' and r['condition']=='without_rubric']
    deliveries={ (str(r['task_id']),int(r['repeat'])):sum(f['size_bytes'] for f in r['agent_files']) for r in manifest }
    assert len(deliveries)==880
    a['delivery_usd']=[deliveries[(r.task_id,r.repeat)]/1e9*0.09 for r in a.itertuples()]
    a['rental_cost_usd']=a.allocated_seconds*hourly/3600+a.delivery_usd
    a['rate_usd_per_hour']=hourly
    a['cost_scope']='recorded task-execution windows at the selected rental rate'
    return a, audit


def summarize(metrics_module, inputs, task, *, model, baseline, policy='pass1', groups=True):
    task=task.sort_values(['occupation','task_id']).copy()
    assert len(task)==220 and task.groupby('occupation').size().eq(5).all()
    assert list(task.occupation.unique())==inputs.occupation_order
    task['net_value_usd']=task.generated_value_usd-task.rental_cost_usd
    arrays={}
    for col in FIELDS[:5]:
        p=task[col].to_numpy().reshape(44,5)
        arrays[col]=np.take_along_axis(np.broadcast_to(p,(len(inputs.bootstrap_draws),44,5)),inputs.bootstrap_draws,axis=2).mean(axis=2)
    lookup=inputs.weights.set_index('occupation').loc[inputs.occupation_order]
    records=[]
    selections=[('global','All tasks',task)]
    if groups:
        selections += [('category',str(k),g) for k,g in task.groupby('category')]
        selections += [('occupation',str(k),g) for k,g in task.groupby('occupation')]
    for level,name,g in selections:
        mask=lookup.index.isin(g.occupation.unique())
        w=lookup.loc[mask,'gdp_structure_weight'].to_numpy()
        mass=w.sum()
        w=w/mass
        t=g.assign(w=g.occupation.map(lookup.gdp_structure_weight)/5/mass)
        r=dict(model=model,baseline=baseline,policy=policy,level=level,application=name,
               n_tasks=len(g),gdp_weight_mass=mass,success_rate=float(t.w@t.credit))
        boot={}
        for col in FIELDS[:5]:
            r[col]=float(t.w@t[col])
            boot[col]=arrays[col][:,mask]@w
        r.update(nvj=r['net_value_usd']/r['energy_mj'],tsj=r['hours_saved']/r['energy_mj'],nvd=r['net_value_usd']/r['rental_cost_usd'])
        boot.update(nvj=boot['net_value_usd']/boot['energy_mj'],tsj=boot['hours_saved']/boot['energy_mj'],nvd=boot['net_value_usd']/boot['rental_cost_usd'])
        for col,vals in boot.items():
            r[col+'_ci_low'],r[col+'_ci_high']=np.quantile(vals,[.025,.975])
            r[col+'_bootstrap_sd']=np.std(vals,ddof=1)
        records.append(r)
    result=pd.DataFrame(records)
    for level in result.level.unique():
        m=result.level.eq(level)
        for field in ['nvj','tsj','nvd']:
            result.loc[m,field+'_rank']=result.loc[m,field].rank(ascending=False,method='min').astype(int)
    return result


def prompt_from_attempts(inputs, a, cost, baseline, k=1, threshold=1):
    c=a.merge(cost[['task_id','repeat','rental_cost_usd']],on=['task_id','repeat'],validate='one_to_one')
    c=c[c['repeat']<k].copy()
    c['successful_latency']=c.latency_hours.where(c.success)
    t=c.groupby('task_id',as_index=False).agg(success_count=('success','sum'),
       mean_latency=('successful_latency','mean'),energy_mj=('energy_mj','sum'),rental_cost_usd=('rental_cost_usd','sum'))
    t=inputs.task_meta.merge(t,on='task_id',validate='one_to_one')
    h=t['human_only_hours' if baseline=='human' else 'human_ai_hours']
    t['credit']=t.success_count>=threshold
    t['generated_value_usd']=t.credit*h*t.median_hourly_wage_usd
    t['hours_saved']=np.where(t.credit,h-t.mean_latency,0.)
    return t


def hardware_results_from_frozen_costs(inputs, cost_path):
    """Replay verified task costs when the historical H100 timing archive is absent."""
    cost = pd.read_csv(cost_path, dtype={"task_id": str})
    cost["repeat"] = cost["repeat"].astype(int)
    profiles = {
        'qwen_h100': 'NVIDIA H100',
        'qwen_h200': 'NVIDIA H200',
        'qwen_b200': 'NVIDIA B200',
        'qwen_mi350x': 'AMD MI350X',
    }
    required = {'task_id', 'repeat', 'profile', 'rental_cost_usd'}
    assert required.issubset(cost.columns)
    assert set(cost.profile) == set(profiles)
    assert not cost.duplicated(['profile', 'task_id', 'repeat']).any()
    frames = primary.hardware_attempts(inputs.task_meta)
    outputs = []
    for profile, label in profiles.items():
        part = cost[cost.profile == profile].copy()
        assert len(part) == 880 and part.task_id.nunique() == 220
        assert part.rental_cost_usd.notna().all() and part.rental_cost_usd.ge(0).all()
        t = prompt_from_attempts(inputs, frames[label], part, 'human')
        t.assign(model=profile, baseline='human').to_csv(OUT / (profile + '_human_tasks.csv'), index=False)
        outputs.append(summarize(None, inputs, t, model=profile, baseline='human'))
    result = pd.concat(outputs)
    cost.to_csv(OUT / 'hardware_cost_ledgers.csv', index=False)
    result.to_csv(OUT / 'hardware_net_metrics.csv', index=False)
    return result


def hardware_results(metrics_module, inputs):
    frozen = ROOT / 'reproduction/inputs/hardware_cost_ledgers.csv'
    h100 = primary.load_module('net_h100_probe', ROOT / 'cluster/h100_qwen_metrics.py')
    if not h100.TRACE_PATH.is_file() and frozen.is_file():
        return hardware_results_from_frozen_costs(inputs, frozen)
    from hardware_inputs import available_inputs
    normalized=available_inputs(metrics_module)
    frames=primary.hardware_attempts(inputs.task_meta)
    # The single-GPU reference containers hold the host reservation fixed at
    # the recorded B200 deployment's 32 physical cores and 256 GiB RAM.
    host_hourly=32*0.04716+256*0.007992
    rates={'qwen_h100':3.9492+host_hourly,'qwen_h200':4.5396+host_hourly,'qwen_b200':6.2496+host_hourly}
    labels={'qwen_h100':'NVIDIA H100','qwen_h200':'NVIDIA H200','qwen_b200':'NVIDIA B200'}
    out=[]
    cost_ledgers=[]
    for profile,label in labels.items():
        rs,bs=normalized[profile]
        rs=rs.copy()
        rs['allocated_seconds']=np.nan
        for b in bs.itertuples():
            g=rs[rs.batch_key==b.batch_key]
            assert len(g)==b.physical_request_count
            shares,idle=occupied_shares(g.start_s,g.end_s,b.batch_start_s,b.batch_end_s)
            shares+=idle*shares/shares.sum()
            rs.loc[g.index,'allocated_seconds']=shares
        c=rs.groupby(['task_id','repeat'],as_index=False).allocated_seconds.sum()
        c['rental_cost_usd']=c.allocated_seconds*rates[profile]/3600
        c['profile']=profile
        c['rate_usd_per_hour']=rates[profile]
        c['cost_scope']='prompt windows at the full reserved GPU, CPU, and RAM rate'
        cost_ledgers.append(c)
        a=frames[label]
        t=prompt_from_attempts(inputs,a,c,'human')
        t.assign(model=profile,baseline='human').to_csv(OUT/(profile+'_human_tasks.csv'),index=False)
        out.append(summarize(metrics_module,inputs,t,model=profile,baseline='human'))
    # Apportion each production and repair batch duration across its requests,
    # using their task durations as allocation weights.
    amd=primary.load_module('net_amd',ROOT/'cluster/amd_qwen_metrics.py')
    _,contributions,batches,_=amd.load_energy_ledger(amd.DEFAULT_INPUT_ROOT)
    a=frames['AMD MI350X'].copy()
    con=pd.DataFrame(contributions)
    batch=pd.DataFrame(batches)
    con=con.merge(batch[['source_name','batch_id','wall_clock_s']],on=['source_name','batch_id'],validate='many_to_one')
    con=con.merge(a[['task_id','repeat','latency_seconds']],on=['task_id','repeat'],validate='many_to_one')
    con['allocated_seconds']=np.nan
    for key,g in con.groupby(['source_name','batch_id']):
        duration=float(g.wall_clock_s.iloc[0])
        ends=g.latency_seconds.to_numpy()
        shares,_=occupied_shares(np.zeros(len(g)),ends,0.,float(ends.max()))
        con.loc[g.index,'allocated_seconds']=shares/shares.sum()*duration
    c=con.groupby(['task_id','repeat'],as_index=False).allocated_seconds.sum()
    # Apply the bundled MI350X instance quote to the allocated serving time.
    c['rental_cost_usd']=c.allocated_seconds*3.65/3600
    c['profile']='qwen_mi350x'
    c['rate_usd_per_hour']=3.65
    c['cost_scope']='bundled MI350X instance rate applied to duration-weighted batch allocations'
    cost_ledgers.append(c)
    t=prompt_from_attempts(inputs,a,c,'human')
    t.assign(model='qwen_mi350x',baseline='human').to_csv(OUT/'qwen_mi350x_human_tasks.csv',index=False)
    out.append(summarize(metrics_module,inputs,t,model='qwen_mi350x',baseline='human'))
    pd.concat(cost_ledgers).to_csv(OUT/'hardware_cost_ledgers.csv',index=False)
    pd.concat(out).to_csv(OUT/'hardware_net_metrics.csv',index=False)
    return pd.concat(out)


def main():
    OUT.mkdir(exist_ok=True)
    metrics_module=primary.load_module('net_metrics',primary.METRICS_MODULE)
    inputs=metrics_module.load_inputs(primary.BOOTSTRAP_SAMPLES,primary.BOOTSTRAP_SEED)
    glm_allocation_path=ROOT/'reports/glm_primary_active_energy_20260919/request_allocations.csv'
    glm_validation=json.loads((glm_allocation_path.parent/'validation.json').read_text())
    boundary=os.environ.get('J2G_ENERGY_BOUNDARY','total')
    expected_boundary='GPU+CPU+DRAM measured total' if boundary=='total' else 'GPU'
    if boundary not in {'total','accelerator'} or glm_validation.get('energy_boundary') != expected_boundary:
        raise ValueError('GLM energy attribution was built under a different energy boundary')
    if hashlib.sha256(glm_allocation_path.read_bytes()).hexdigest() != glm_validation.get('request_allocations_sha256'):
        raise ValueError('GLM request allocations changed after validation')
    glm_energy=pd.read_csv(glm_allocation_path)
    glm_cost, glm_cost_audit = glm_prompt_window_costs(
        metrics_module.SOURCE_PATHS['glm_query'], metrics_module.SOURCE_PATHS['glm_batch']
    )
    glm_cost_audit.to_csv(OUT/'glm_prompt_window_cost_reconciliation.csv', index=False)
    model_cost={'glm52_h200':glm_cost}
    audit=[]
    results=[]
    tasks=[]
    retry=[]
    sens=[]
    for model in metrics_module.MODEL_CONFIGS:
        if model!='glm52_h200':
            model_cost[model],ba=costs_for_other_model(metrics_module,model)
            audit+=ba
        model_cost[model].to_csv(OUT/(model+'_cost_ledger.csv'),index=False)
        for baseline in (['human','human_ai'] if model=='glm52_h200' else ['human']):
            a=inputs.attempts[(model,baseline)].copy()
            if model=='glm52_h200':
                a=a.drop(columns=['energy_mj'],errors='ignore').merge(glm_energy[['task_id','repeat','active_energy_mj','equal_energy_mj']],on=['task_id','repeat'],validate='one_to_one')
                a['energy_mj']=a.active_energy_mj
            elif 'energy_mj' not in a:
                lookup=inputs.models[model].set_index('task_id').energy_all_four_runs_mj/4
                a['energy_mj']=a.task_id.map(lookup)
            t=prompt_from_attempts(inputs,a,model_cost[model],baseline)
            t.assign(model=model,baseline=baseline).to_csv(OUT/(model+'_'+baseline+'_tasks.csv'),index=False)
            r=summarize(metrics_module,inputs,t,model=model,baseline=baseline)
            results.append(r)
            tasks.append(t.assign(model=model,baseline=baseline))
            # Explicit accounting stress, not a confidence interval.
            for factor in [1,1.25,2]:
                x=t.copy()
                x.rental_cost_usd*=factor
                z=summarize(metrics_module,inputs,x,model=model,baseline=baseline,groups=False)
                sens.append(z.assign(cost_multiplier=factor))
            if model=='glm52_h200':
                # Charge the same recorded prompt-window allocation for every selected run.
                old=a.copy()
                old.energy_mj=old.equal_energy_mj
                for k in [1,2,3,4]:
                    rt=prompt_from_attempts(inputs,a,model_cost[model],baseline,k=k)
                    retry.append(summarize(metrics_module,inputs,rt,model=model,baseline=baseline,policy='pass'+str(k),groups=False))
                for threshold in [1,2,3,4]:
                    rt=prompt_from_attempts(inputs,a,model_cost[model],baseline,k=4,threshold=threshold)
                    retry.append(summarize(metrics_module,inputs,rt,model=model,baseline=baseline,policy='threshold'+str(threshold),groups=False))
                eq=prompt_from_attempts(inputs,old,model_cost[model],baseline)
                sr=summarize(metrics_module,inputs,eq,model=model,baseline=baseline,policy='equal_counterfactual')
                sr.to_csv(OUT/(baseline+'_energy_counterfactual.csv'),index=False)
    allresults=pd.concat(results,ignore_index=True)
    allresults.to_csv(OUT/'all_net_metrics.csv',index=False,float_format='%.15g')
    pd.concat(tasks).to_csv(OUT/'all_net_task_inputs.csv',index=False)
    pd.concat(retry).to_csv(OUT/'glm_retry_net_metrics.csv',index=False)
    pd.concat(sens).to_csv(OUT/'rental_cost_sensitivity.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'other_model_batch_cost_reconciliation.csv',index=False)
    for level in ['category','global','occupation']:
        allresults[allresults.level==level].to_csv(OUT/(level+'_net_metrics.csv'),index=False)
    # Conditional energy-attribution comparison keeps the exact same cost numerator.
    correlations=[]
    for baseline in ['human','human_ai']:
        new=allresults.query('model == "glm52_h200" and baseline == @baseline and level == "category"').set_index('application').sort_index()
        old=pd.read_csv(OUT/(baseline+'_energy_counterfactual.csv')).query('level == "category"').set_index('application').loc[new.index]
        for field in ['energy_mj','nvj','tsj']:
            correlations.append(dict(baseline=baseline,metric=field,pearson=float(pearsonr(old[field],new[field]).statistic),spearman=float(spearmanr(old[field],new[field]).statistic),ranks_changed=int((old[field].rank()!=new[field].rank()).sum())))
        for col in ['nvj','tsj','energy_mj']:
            new[col+'_equal']=old[col]
            new[col+'_equal_rank']=old[col].rank(ascending=col=='energy_mj',method='min')
        new.to_csv(OUT/(baseline+'_attribution_comparison.csv'))
    pd.DataFrame(correlations).to_csv(OUT/'net_energy_correlations.csv',index=False)
    hw=hardware_results(metrics_module,inputs)
    validation=dict(energy_boundary=expected_boundary,glm_energy_total_mj=float(glm_energy.active_energy_mj.sum()),glm_all_attempt_cost_usd=float(glm_cost.rental_cost_usd.sum()),
        model_task_counts={m:len(d) for m,d in pd.concat(tasks).query('baseline == "human"').groupby('model')},
        max_other_batch_seconds_residual=max(abs(r['charged_seconds']-r['allocated_seconds']) for r in audit),
        source_metadata_hashes_verified=True,
        energy_update_scope=('all model and hardware metrics use validated GPU+CPU+DRAM; GLM headline pass1 uses active-request allocation'
                             if expected_boundary == 'GPU+CPU+DRAM measured total' else
                             'Accelerator energy; GLM headline pass1 uses active-request allocation'),
        estimates_not_invoices=True,glm_cost_scope='recorded_prompt_window_only',
        bootstrap_scope='10,000 paired occupation-stratified task draws, conditional on energy and rental assumptions')
    jwrite('net_validation.json',validation)
    print(allresults.query('level == "global"')[['model','baseline','generated_value_usd','rental_cost_usd','energy_mj','nvj','tsj','nvd']].to_string(index=False))
    print(hw.query('level == "global"')[['model','rental_cost_usd','nvj','tsj','nvd']].to_string(index=False))


if __name__=='__main__':
    main()
