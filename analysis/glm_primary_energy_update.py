#!/usr/bin/env python3
"""GLM primary attribution under the selected measured energy boundary.

Run after the numerical analysis builder. This overlay changes GLM headline
pass@1 attribution and dependent comparisons. In total mode the input builder
has already replaced every model's attempt energy with validated components.
All GLM energy is allocated over 880 requests before selecting repeat zero.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

from active_request_energy import RequestInterval, attribute_batch
from energy_boundary import attempt_energy
import build_pass1_primary_results as primary

ROOT = primary.ROOT
OUT = ROOT / 'reports/glm_primary_active_energy_20260919'


def allocations(metrics_module):
    q = pd.read_csv(metrics_module.SOURCE_PATHS['glm_query']).query("dataset == 'GDPVal'").copy()
    batches = pd.read_csv(metrics_module.SOURCE_PATHS['glm_batch']).query("dataset == 'GDPVal'")
    assert len(q) == 880 and len(batches) == 44
    q['task_id'] = q.judge_task_id.astype(str)
    q['repeat'] = q.repeat_index.astype(int)
    q['equal_energy_mj'] = q.allocated_gpu_energy_joules / 1e6
    q['active_energy_mj'] = np.nan
    audit = []
    for b in batches.itertuples():
        g = q[q.batch_key == b.batch_key]
        assert len(g) == b.attempts == 20 and g.occupation.nunique() == 1
        result = attribute_batch(
            [RequestInterval(r.query_id, r.prompt_start_time_s, r.prompt_end_time_s)
             for r in g.itertuples()],
            batch_start_s=b.prompt_start_time_s, batch_end_s=b.prompt_end_time_s,
            batch_energy_j=b.gpu_energy_joules)
        q.loc[g.index, 'active_energy_mj'] = g.query_id.map(result.request_energy_j) / 1e6
        np.testing.assert_allclose(g.equal_energy_mj, b.gpu_energy_joules / 20 / 1e6, rtol=1e-12)
        audit.append(dict(batch_key=b.batch_key, measured_energy_j=b.gpu_energy_joules,
                          residual_j=result.reconciliation_error_j,
                          no_active_seconds=result.no_active_seconds,
                          no_active_energy_j=result.no_active_energy_j))
    assert not q.duplicated(['task_id', 'repeat']).any()
    assert q.groupby('task_id')['repeat'].apply(lambda x: set(x) == {0, 1, 2, 3}).all()
    assert q.active_energy_mj.notna().all()
    np.testing.assert_allclose(q.active_energy_mj.sum(), q.equal_energy_mj.sum(), rtol=1e-12)
    energy_boundary = os.environ.get('J2G_ENERGY_BOUNDARY', 'total')
    if energy_boundary == 'total':
        recorded = attempt_energy().query("profile == 'glm52_h200'")[
            ['task_id', 'repeat', 'total_energy_j', 'equal_total_energy_j']
        ]
        q['gpu_active_energy_mj'] = q.active_energy_mj
        q['gpu_equal_energy_mj'] = q.equal_energy_mj
        q = q.merge(recorded, on=['task_id', 'repeat'], how='left', validate='one_to_one')
        if q[['total_energy_j', 'equal_total_energy_j']].isna().any().any():
            raise ValueError('Missing measured GPU+CPU+DRAM GLM attempt energy')
        q['active_energy_mj'] = q.pop('total_energy_j') / 1e6
        q['equal_energy_mj'] = q.pop('equal_total_energy_j') / 1e6
    elif energy_boundary != 'accelerator':
        raise ValueError(f'Unknown energy boundary: {energy_boundary}')
    np.testing.assert_allclose(q.active_energy_mj.sum(), q.equal_energy_mj.sum(), rtol=1e-12)
    q['energy_change_pct'] = 100 * (q.active_energy_mj / q.equal_energy_mj - 1)
    q.to_csv(OUT / 'request_allocations.csv', index=False, float_format='%.17g')
    pd.DataFrame(audit).to_csv(OUT / 'batch_reconciliation.csv', index=False)
    return q


def apply_allocations(inputs, q, field):
    lookup = q.set_index(['task_id', 'repeat'])[field]
    for baseline in ('human', 'human_ai'):
        attempts = inputs.attempts[('glm52_h200', baseline)]
        attempts['energy_mj'] = [lookup.loc[(r.task_id, int(r.repeat))] for r in attempts.itertuples()]


def correlation_row(scope, baseline, metric, x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    change = 100 * (y / x - 1)
    return dict(scope=scope, baseline=baseline, metric=metric, n=len(x),
                pearson_r=float(pearsonr(x, y).statistic),
                spearman_rho=float(spearmanr(x, y).statistic),
                kendall_tau_b=float(kendalltau(x, y).statistic),
                median_absolute_change_pct=float(np.median(np.abs(change))),
                min_change_pct=float(change.min()), max_change_pct=float(change.max()))


def rebuild():
    OUT.mkdir(exist_ok=True, parents=True)
    metrics_module = primary.load_module('glm_primary_metrics', primary.METRICS_MODULE)
    inputs = metrics_module.load_inputs(primary.BOOTSTRAP_SAMPLES, primary.BOOTSTRAP_SEED)
    q = allocations(metrics_module)
    category_rows, global_rows, correlations = [], [], []
    for scope, panel in [('all_requests', q), ('recorded_repeat_0', q[q['repeat'] == 0])]:
        correlations.append(correlation_row(scope, 'both', 'energy_mj', panel.equal_energy_mj, panel.active_energy_mj))
    # Store both counterfactuals with identical outcomes, wages, and bootstrap draws.
    results = {}
    for scheme, field in [('equal', 'equal_energy_mj'), ('active', 'active_energy_mj')]:
        apply_allocations(inputs, q, field)
        for baseline, hours in [('human', 'human_only_hours'), ('human_ai', 'human_ai_hours')]:
            prompt = primary.pass1_prompt(inputs, 'glm52_h200', baseline)
            cats, glob, _, gu = primary.analyze_prompt(metrics_module, inputs, prompt, hours)
            cu, _ = metrics_module.bootstrap_metrics(inputs, prompt, 1, hours, 'median_hourly_wage_usd', cats, glob)
            # The frozen helper's attempt-rate bootstrap uses a four-repeat divisor.
            # Correct that field for this explicitly one-repeat panel.
            for column in cu:
                if column.startswith('attempt_win_or_tie_rate_') and not column.endswith('_estimate'):
                    cu[column] *= 4
            combined = metrics_module.combine_point_uncertainty(cats, cu)
            combined['attempts'] = combined.n_unique_prompts
            results[(scheme, baseline)] = (cats, glob, combined)
            combined.assign(scheme=scheme, baseline=baseline).to_csv(
                OUT / f'{baseline}_{scheme}_category_bootstrap.csv', index=False, float_format='%.15g')
            global_rows.append(dict(scheme=scheme, baseline=baseline, **glob))
            if scheme == 'active':
                primary.write_frame(f'glm_{baseline}_category_metrics_pass1', combined)
        # Paired bootstrap differences: task composition resampled identically.
        for baseline, hours in [('human', 'human_only_hours'), ('human_ai', 'human_ai_hours')]:
            prompt = primary.pass1_prompt(inputs, 'glm52_h200', baseline)
            matrices = {}
            for label, column in [('energy', 'energy_all_four_runs_mj')]:
                matrices[label] = metrics_module.prompt_matrix(prompt, inputs.occupation_order, column)
            success = metrics_module.prompt_matrix(prompt, inputs.occupation_order, 'success_count')
            h = metrics_module.prompt_matrix(prompt, inputs.occupation_order, hours)
            wage = metrics_module.prompt_matrix(prompt, inputs.occupation_order, 'median_hourly_wage_usd')
            latency = np.nan_to_num(metrics_module.prompt_matrix(prompt, inputs.occupation_order, 'mean_success_latency_hours'))
            matrices.update(value=success*h*wage, saved=success*(h-latency))
            boot = {}
            for label, matrix in matrices.items():
                sampled = np.take_along_axis(np.broadcast_to(matrix, (10000,)+matrix.shape), inputs.bootstrap_draws, axis=2).sum(axis=2)
                boot[label] = sampled @ inputs.weight_vector / 5
            np.savez_compressed(OUT/f'{baseline}_{scheme}_paired_draws.npz', energy=boot['energy'], dpj=boot['value']/boot['energy'], tsj=boot['saved']/boot['energy'])

    for baseline in ('human', 'human_ai'):
        old, new = (results[(s, baseline)][0].set_index('occupation_category') for s in ('equal', 'active'))
        c = pd.DataFrame(index=old.index)
        for metric in ('energy_mj', 'value_per_mj', 'hours_per_mj'):
            c[metric+'_equal'], c[metric+'_active'] = old[metric], new[metric]
            c[metric+'_change_pct'] = 100*(new[metric]/old[metric]-1)
            c[metric+'_rank_equal'], c[metric+'_rank_active'] = old[metric+'_rank'], new[metric+'_rank']
            correlations.append(correlation_row('occupation_categories', baseline, metric, old[metric], new[metric]))
        category_rows.append(c.reset_index().assign(baseline=baseline))
    category = pd.concat(category_rows)
    category.to_csv(OUT/'category_comparison.csv', index=False, float_format='%.15g')
    pd.DataFrame(global_rows).to_csv(OUT/'global_comparison.csv', index=False, float_format='%.15g')
    pd.DataFrame(correlations).to_csv(OUT/'correlations.csv', index=False, float_format='%.15g')
    paired = []
    for baseline in ('human','human_ai'):
        old, new = (np.load(OUT/f'{baseline}_{s}_paired_draws.npz') for s in ('equal','active'))
        for metric in ('energy','dpj','tsj'):
            values=100*(new[metric]/old[metric]-1)
            paired.append(dict(baseline=baseline,metric=metric,**metrics_module.uncertainty(values)))
    pd.DataFrame(paired).to_csv(OUT/'paired_global_change_percent_bootstrap.csv',index=False)

    # Propagate the GLM primary denominator; other systems retain the selected
    # boundary from the input builder.
    before = pd.read_csv(primary.TABLE_DIR/'five_model_global_system_comparison.csv')
    primary.model_and_reference_tables(metrics_module, inputs)
    # Restore the correctly normalized expert-with-AI category attempt-rate uncertainty.
    primary.write_frame('glm_human_ai_category_metrics_pass1', results[('active','human_ai')][2])
    primary.wage_and_three_way_tables(metrics_module, inputs)
    after = pd.read_csv(primary.TABLE_DIR/'five_model_global_system_comparison.csv')
    pd.testing.assert_frame_equal(before[before.model!='GLM-5.2'].reset_index(drop=True), after[after.model!='GLM-5.2'].reset_index(drop=True))
    total_boundary = os.environ.get('J2G_ENERGY_BOUNDARY', 'total') == 'total'
    audit = dict(primary_method='active-request sharing using constant batch power',
                 energy_boundary='GPU+CPU+DRAM measured total' if total_boundary else 'GPU',
                 request_allocations_sha256=hashlib.sha256((OUT/'request_allocations.csv').read_bytes()).hexdigest(),
                 scope='GLM 8xH200 headline pass@1, both expert conditions and dependent comparisons',
                 retained_method=('dedicated pass@1-4 and threshold counterfactuals use equal split; all headline model and hardware denominators use validated total components'
                                  if total_boundary else 'Accelerator energy: dedicated pass@1-4 and threshold counterfactuals use equal split'),
                 all_880_requests_allocated_before_selection=True,
                 non_glm_model_rows_unchanged=True,
                 source_hashes={str(metrics_module.SOURCE_PATHS[k]):hashlib.sha256(Path(metrics_module.SOURCE_PATHS[k]).read_bytes()).hexdigest() for k in ('glm_query','glm_batch')})
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(pd.DataFrame(correlations).to_string(index=False))
    print(pd.DataFrame(global_rows)[['scheme','baseline','energy_mj','value_per_mj','hours_per_mj']].to_string(index=False))


if __name__ == '__main__':
    rebuild()
