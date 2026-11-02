#!/usr/bin/env python3
"""Audit and compare the three archived GLM judging pipelines; no API calls.

Run from any directory: python3 analysis/compare_glm_judges.py
Dependencies: numpy and scipy. All source archives remain unchanged.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/glm_judge_consistency_20260919"
SOURCES = {
    "Gemini": ROOT / "reports/gdpval_pairwise_gemini31_flash_lite_20260716",
    "Haiku": ROOT / "reports/gdpval_pairwise_claude_haiku45_20260716",
    "GPT": ROOT / "reports/gdpval_pairwise_glm52_20260715",
}
MODELS = ["models/gemini-3.1-flash-lite", "claude-haiku-4-5-20251001", "gpt-5-mini-2025-08-07"]
JUDGES = list(SOURCES)
GROUPS = list(itertools.product(["without_rubric", "with_rubric"], ["human", "human_ai"]))
SEED = 20260919
N_BOOT = 10000


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def estimate(value, samples):
    samples = np.asarray(samples)
    assert np.isfinite(value) and np.isfinite(samples).all()
    return {"estimate": float(value), "ci95": np.quantile(samples, [.025, .975]).tolist()}


def kappa_from_counts(counts):
    """Unweighted Cohen kappa with the last two axes the category contingency."""
    total = counts.sum(axis=(-2, -1))
    observed = np.trace(counts, axis1=-2, axis2=-1) / total
    chance = (counts.sum(axis=-1) * counts.sum(axis=-2)).sum(axis=-1) / total**2
    return (observed - chance) / (1 - chance)


def task_counts(x, y, categories):
    return np.stack([(x == a) & (y == b) for a in categories for b in categories], axis=-1).sum(axis=1)


def correlation(x, y, weights):
    """Task-cluster bootstrap Pearson r; x/y have task x comparison-stratum axes."""
    count = weights.sum(axis=1) * x.shape[1]
    ex = weights @ x.sum(axis=1) / count
    ey = weights @ y.sum(axis=1) / count
    exy = weights @ (x*y).sum(axis=1) / count
    vx = weights @ (x*x).sum(axis=1) / count - ex**2
    vy = weights @ (y*y).sum(axis=1) / count - ey**2
    return (exy - ex*ey) / np.sqrt(vx*vy)


def analyze(scores, weights, occupation_indices):
    """scores: task x stratum x repeat x judge, ordered identically for all judges."""
    flat = scores.reshape(len(scores), -1, 3)
    n = flat.shape[0] * flat.shape[1]
    boot_n = weights.sum(axis=1) * flat.shape[1]
    result = {"n_outputs": n, "n_tasks": len(scores), "rates": {}, "pairs": {}}
    for j, judge in enumerate(JUDGES):
        v = flat[..., j]
        rates = {"wins": int((v == 1).sum()), "ties": int((v == .5).sum()), "losses": int((v == 0).sum())}
        for name, a in [("strict_win", v == 1), ("tie_rate", v == .5), ("tie_adjusted", v), ("win_or_tie", v > 0)]:
            rates[name] = estimate(a.mean(), weights @ a.sum(axis=1) / boot_n)
        result["rates"][judge] = rates
    unanimous = (flat[..., 0] == flat[..., 1]) & (flat[..., 1] == flat[..., 2])
    result["three_way_exact_agreement"] = estimate(unanimous.mean(), weights @ unanimous.sum(axis=1) / boot_n)
    success = flat > 0
    unanimous_binary = (success[..., 0] == success[..., 1]) & (success[..., 1] == success[..., 2])
    result["three_way_success_agreement"] = estimate(unanimous_binary.mean(), weights @ unanimous_binary.sum(axis=1) / boot_n)
    counts = np.stack([(flat == v).sum(axis=-1) for v in [0, .5, 1]], axis=-1)
    p = ((counts**2).sum(axis=-1) - 3) / 6
    marg = counts.sum(axis=1)
    pe = ((marg.sum(axis=0)/(n*3))**2).sum()
    boot_pe = ((weights @ marg / (boot_n[:, None]*3))**2).sum(axis=1)
    result["fleiss_kappa"] = estimate((p.mean()-pe)/(1-pe), (weights @ p.sum(axis=1)/boot_n-boot_pe)/(1-boot_pe))
    for a, b in itertools.combinations(range(3), 2):
        x, y = flat[..., a], flat[..., b]
        eq = x == y
        counts = task_counts(x, y, [0, .5, 1])
        contingency = counts.sum(axis=0).reshape(3, 3)
        boot_counts = (weights @ counts).reshape(-1, 3, 3)
        pair = {
            "exact_agreement": estimate(eq.mean(), weights @ eq.sum(axis=1)/boot_n),
            "cohen_kappa": estimate(kappa_from_counts(contingency), kappa_from_counts(boot_counts)),
            "confusion_categories": ["loss", "tie", "win"],
            "confusion_rows_first_judge_columns_second": contingency.tolist(),
            "opposite_win_loss_rate": estimate((abs(x-y) == 1).mean(), weights @ (abs(x-y) == 1).sum(axis=1)/boot_n),
        }
        bx, by = x > 0, y > 0
        bc = task_counts(bx, by, [False, True])
        pair["success_agreement"] = estimate((bx == by).mean(), weights @ (bx == by).sum(axis=1)/boot_n)
        pair["success_cohen_kappa"] = estimate(kappa_from_counts(bc.sum(axis=0).reshape(2,2)), kappa_from_counts((weights @ bc).reshape(-1,2,2)))
        for name, xx, yy in [("tie_adjusted", x, y), ("win_or_tie", bx.astype(float), by.astype(float)), ("strict_win", (x == 1).astype(float), (y == 1).astype(float))]:
            diff = xx-yy
            pair[name+"_difference_first_minus_second"] = estimate(diff.mean(), weights @ diff.sum(axis=1)/boot_n)
            # Four repeat judgments become one task score within each stratum.
            tx = xx.reshape(scores.shape[:3]).mean(axis=2)
            ty = yy.reshape(scores.shape[:3]).mean(axis=2)
            pc = float(np.corrcoef(tx.ravel(), ty.ravel())[0,1])
            pair[name+"_task_pearson"] = estimate(pc, correlation(tx, ty, weights))
            pair[name+"_task_spearman"] = float(spearmanr(tx.ravel(), ty.ravel()).statistic)
            ox = np.array([tx[idx].mean(axis=0) for idx in occupation_indices])
            oy = np.array([ty[idx].mean(axis=0) for idx in occupation_indices])
            pair[name+"_occupation_pearson"] = float(np.corrcoef(ox.ravel(), oy.ravel())[0,1])
            pair[name+"_occupation_spearman"] = float(spearmanr(ox.ravel(), oy.ravel()).statistic)
            pair[name+"_task_mean_absolute_difference"] = float(abs(tx-ty).mean())
        result["pairs"][f"{JUDGES[a]} vs {JUDGES[b]}"] = pair
    return result










def main():
    OUT.mkdir(parents=True,exist_ok=True)
    maps={}
    audit={"source_files":{},"checks":[],"seed":SEED,"bootstrap_samples":N_BOOT}
    manifests=[]
    for judge,model in zip(JUDGES,MODELS):
        source=SOURCES[judge]
        names=['results.output_level.jsonl','judgments.raw.unblinded.jsonl','results.task_level.jsonl','comparison_manifest.unblinded.jsonl','headline_results.csv']
        audit['source_files'][judge]={str((source/n).relative_to(ROOT)):digest(source/n) for n in names}
        rows=read_rows(source/names[0])
        mapping={r['comparison_id']:r for r in rows}
        assert len(rows)==len(mapping)==3520
        assert {r['api_metadata']['model'] for r in rows}=={model}
        raw={r['comparison_id']:r for r in read_rows(source/names[1])}
        assert mapping==raw
        maps[judge]=mapping
        manifests.append(digest(source/names[3]))
    assert len(set(manifests))==1
    ids=sorted(maps[JUDGES[0]])
    assert all(set(m)==set(ids) for m in maps.values())
    metadata=['task_id','repeat','condition','baseline','agent_position','occupation','sector','blind_id']
    aligned=[]
    for cid in ids:
        first=maps[JUDGES[0]][cid]
        record={'comparison_id':cid,**{k:first[k] for k in metadata}}
        for judge in JUDGES:
            r=maps[judge][cid]
            assert all(r[k]==first[k] for k in metadata)
            expected='tie' if r['winner']=='tie' else 'win' if r['winner']==r['agent_position'] else 'loss'
            assert r['winner'] in ['A','B','tie'] and r['agent_outcome']==expected
            assert r['agent_score']=={'win':1,'tie':.5,'loss':0}[expected]
            record[judge]={'outcome':expected,'score':r['agent_score']}
        aligned.append(record)
    tasks=sorted({r['task_id'] for r in aligned})
    assert len(tasks)==220
    task_idx={t:i for i,t in enumerate(tasks)}
    occupations={r['task_id']:r['occupation'] for r in aligned}
    occ_idx=[np.array([i for i,t in enumerate(tasks) if occupations[t]==o]) for o in sorted(set(occupations.values()))]
    assert len(occ_idx)==44 and all(len(x)==5 for x in occ_idx)
    scores=np.full((220,4,4,3),np.nan)
    positions={}
    for r in aligned:
        t=task_idx[r['task_id']]
        g=GROUPS.index((r['condition'],r['baseline']))
        rep=r['repeat']
        assert rep in range(4) and np.isnan(scores[t,g,rep]).all()
        scores[t,g,rep]=[r[j]['score'] for j in JUDGES]
        positions.setdefault((t,g),[]).append(r['agent_position'])
    assert np.isfinite(scores).all()
    assert all(Counter(p)=={'A':2,'B':2} for p in positions.values())
    # Reconstruct archived task-level and headline results as an independent integrity check.
    import csv
    for j,judge in enumerate(JUDGES):
        for r in read_rows(SOURCES[judge]/'results.task_level.jsonl'):
            g=GROUPS.index((r['condition'],r['baseline']))
            s=scores[task_idx[r['task_id']],g,:,j]
            assert r['repeat_count']==4 and r['task_tie_adjusted_score']==s.mean()
            assert (r['wins'],r['ties'],r['losses'])==tuple(int((s==v).sum()) for v in [1,.5,0])
        with (SOURCES[judge]/'headline_results.csv').open() as f:
            for r in csv.DictReader(f):
                g=[f'{c}_vs_{b}' for c,b in GROUPS].index(r['comparison'])
                s=scores[:,g,:,j]
                assert int(r['n'])==s.size
                assert all(int(r[k])==int((s==v).sum()) for k,v in [('wins',1),('ties',.5),('losses',0)])
                np.testing.assert_allclose(float(r['win_or_tie_rate']),(s>0).mean())
    rng=np.random.default_rng(SEED)
    weights=np.zeros((N_BOOT,220))
    for idx in occ_idx:
        weights[:,idx]=rng.multinomial(len(idx),np.full(len(idx),1/len(idx)),size=N_BOOT)
    results={}
    for g,(c,b) in enumerate(GROUPS):
        results[f'{c}_vs_{b}']=analyze(scores[:,g:g+1],weights,occ_idx)
    results['pooled']=analyze(scores,weights,occ_idx)
    audit['checks']=['All three source manifest hashes identical','All 3,520 comparison IDs matched without duplicates','All per-verdict model identities verified','Metadata and candidate orientation identical across judges','Raw and output-level verdicts identical within judge','Outcome and score conversion verified','220 tasks; 44 occupations; five tasks per occupation','Exactly four repeats and balanced A/B positions per task-condition-baseline','Task aggregates and headline counts reproduced']
    gptjobs=json.loads((SOURCES['GPT']/'batch_jobs.json').read_text())
    retry_batches={j['batch_id'] for j in gptjobs if j['kind']=='retry'}
    googlejobs=json.loads((SOURCES['Gemini']/'provider_jobs.json').read_text())
    later_google={j['job_name'] for j in googlejobs[2:]}
    gpt_retry={cid for cid,r in maps['GPT'].items() if r['api_metadata']['batch_id'] in retry_batches}
    gem_retry={cid for cid,r in maps['Gemini'].items() if r['api_metadata']['job_name'] in later_google}
    audit['gpt_compact_final_verdicts']=len(gpt_retry)
    audit['gemini_final_verdicts_from_later_jobs']=len(gem_retry)
    excluded=gpt_retry|gem_retry
    sensitivity={}
    for c,b in GROUPS:
        retained=[r for r in aligned if r['condition']==c and r['baseline']==b and r['comparison_id'] not in excluded]
        s=np.array([[r[j]['score'] for j in JUDGES] for r in retained])
        sensitivity[f'{c}_vs_{b}']={'n_retained':len(s),'n_excluded':880-len(s),'win_or_tie_rates':dict(zip(JUDGES,(s>0).mean(axis=0).tolist())),'pairs':{}}
        for a,z in itertools.combinations(range(3),2):
            counts=np.array([((s[:,a]==u)&(s[:,z]==v)).sum() for u in [0,.5,1] for v in [0,.5,1]]).reshape(3,3)
            sensitivity[f'{c}_vs_{b}']['pairs'][f'{JUDGES[a]} vs {JUDGES[z]}']={'exact_agreement':float((s[:,a]==s[:,z]).mean()),'success_agreement':float(((s[:,a]>0)==(s[:,z]>0)).mean()),'cohen_kappa':float(kappa_from_counts(counts))}
    audit['retry_sensitivity_note']='Drops outputs finalized in any GPT retry or Gemini later resubmission; descriptive output-weighted rates on the retained subset, with no equal-task claim or bootstrap intervals.'
    write_json(OUT/'metrics.json',results)
    write_json(OUT/'audit.json',audit)
    write_json(OUT/'retry_exclusion_sensitivity.json',sensitivity)
    (OUT/'aligned_judgments.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in aligned))
    taskrows=[]
    for t,task in enumerate(tasks):
        for g,(c,b) in enumerate(GROUPS):
            taskrows.append({'task_id':task,'occupation':occupations[task],'condition':c,'baseline':b,**{judge:{'win_or_tie':float((scores[t,g,:,j]>0).mean()),'tie_adjusted':float(scores[t,g,:,j].mean())} for j,judge in enumerate(JUDGES)}})
    (OUT/'task_scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in taskrows))
    print(f'\nSaved verified analysis to {OUT}')


if __name__=='__main__':
    main()
