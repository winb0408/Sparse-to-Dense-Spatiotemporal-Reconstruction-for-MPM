# 历史产物归档清单（ARCHIVE_MANIFEST_20261110）

> 由 `python -m spur_mpm.scripts.audit_runs --out-dir docs/archive` 生成，**默认 dry-run**。
> 本文件是只读盘点，不是删除记录；任何删除都要另一份 reviewed allowlist 和 `docs/archive/CLEANUP_REPORT_20261110.md`。

- 采集时间（UTC）：`2026-11-13T14:10:20Z`
- 分支：`feature/phase1-adaptive-prior-pr3-20261112`
- HEAD：`e8b8abd233d8ca18fd69bce92db548bd71cc5ff9`
- 条目：35　合计 6.56 GiB
- 处置分布：`{"keep": 11, "keep-evidence": 21, "review": 3}`
- 类别分布：`{"A": 3, "B": 8, "C": 21, "D": 2, "review": 1}`

## 为什么这份清单里还有负结果

类别 B/C 的目录**不因为结果不好或协议作废就删除**：它们是证伪记录，也是 PR #1 遗留
复核项（B4 旧 checkpoint 重评）唯一能用的权重来源。清单只回答“这些文件是什么、
谁在引用、能不能再生”，删不删是另一个决定。

## 受保护路径（任何 allowlist 都改不动）

- `data_local`
- `runs/E021_ladder`
- `runs/E023_phase1`
- `docs/RESULTS.md`
- `docs/PHASE1_AUDIT_RESPONSE.md`
- `docs/archive`
- `spur_mpm`

## 分类规则

| 前缀 | 类别 | 处置 | 依据 |
|---|---|---|---|
| `runs/E021_ladder` | A | keep | primary supervised ladder; its negative ablations are published evidence |
| `runs/E023_phase1` | A | keep | audit logs, re-scores and full-volume structure exports |
| `runs/E024_dual_prior` | A | keep | PR2 dual-prior ladder; its no-Wiener-gain result is the published evidence for plan SS8.2 (docs/PHASE1_PR2_E024_RESULTS.md) |
| `runs/E025_adaptive_dual_prior` | B | keep | PR3 y-only negative control; kept whatever sign it comes out with |
| `runs/E026_psras_condition` | B | keep | PR3 historical-condition re-check; its matched_synthetic null is evidence (docs/PHASE1_PR3_E026_RESULTS.md) |
| `runs/E001` | B | keep | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E002` | C | keep-evidence | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E009` | C | keep-evidence | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/pilot_S1_fold0` | D | delete | one-off pilot run before the E002 cohort existed |
| `runs/smoke` | D | delete | smoke-test run with random weights; runs/smoke/audit is the data audit |

## 逐条盘点

| 路径 | 类别 | 处置 | 大小 | 文件数 | 已跟踪 | 被引用 | sha256（清单摘要） | 理由 |
|---|---|---|---|---|---|---|---|---|
| `runs/E001_A0_fold0_test` | B | keep | 0.2 MiB | 2 | 2 | 4 | `07a20ec99d5a5945` | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E001_A0_fold1_test` | B | keep | 0.2 MiB | 2 | 2 | 4 | `223c2251a1c77ed4` | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E001_A0_fold2_test` | B | keep | 0.1 MiB | 2 | 2 | 4 | `6924acc4da70fee0` | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E001_A1_fold0_test` | B | keep | 0.2 MiB | 2 | 2 | 4 | `9491d509371dd9d3` | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E001_A1_fold1_test` | B | keep | 0.2 MiB | 2 | 2 | 4 | `fe9a9e46225be7a9` | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E001_A1_fold2_test` | B | keep | 0.1 MiB | 2 | 2 | 4 | `f8e9830e6289ae14` | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E002_S0_fold0` | C | keep-evidence | 49.3 MiB | 8 | 5 | 7 | `e9bdb1bdf8fc1be1` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S0_fold0_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 4 | `9fc52688c8958146` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S0_fold1` | C | keep-evidence | 49.3 MiB | 8 | 5 | 5 | `1a8d0ad305f85448` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S0_fold1_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 4 | `8bea36f5d978ecfc` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S0_fold2` | C | keep-evidence | 49.3 MiB | 8 | 5 | 5 | `b05199766ba8c3fd` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S0_fold2_test` | C | keep-evidence | 0.1 MiB | 2 | 2 | 4 | `027eeea30086c7fa` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S1_fold0` | C | keep-evidence | 49.3 MiB | 8 | 5 | 9 | `f080c39cae53fd20` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S1_fold0_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 4 | `e79c05ef7b991b5a` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S1_fold1` | C | keep-evidence | 49.3 MiB | 8 | 5 | 8 | `821469084807edd3` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S1_fold1_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 4 | `283e9169e5e8a649` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S1_fold2` | C | keep-evidence | 49.3 MiB | 8 | 5 | 8 | `3f4de83b7566047c` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S1_fold2_test` | C | keep-evidence | 0.1 MiB | 2 | 2 | 4 | `9ba2f750515e2bd3` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E009_A0_fold0_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 2 | `60c73629c66eebbe` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_A0_fold1_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 2 | `b26802228fa5626c` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_A0_fold2_test` | C | keep-evidence | 0.1 MiB | 2 | 2 | 2 | `f0ede1504fad5913` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_S1_fold0` | C | keep-evidence | 49.3 MiB | 8 | 5 | 3 | `0879cfd50b51c397` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_S1_fold0_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 2 | `68e702ce5a121db9` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_S1_fold1` | C | keep-evidence | 49.3 MiB | 8 | 5 | 3 | `60cc63bcbef713e0` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_S1_fold1_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 2 | `4385dca13dfeb4be` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_S1_fold2` | C | keep-evidence | 49.3 MiB | 8 | 5 | 3 | `6e212ec506c1052d` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_S1_fold2_test` | C | keep-evidence | 0.1 MiB | 2 | 2 | 2 | `7c2d3d37f29b8bf9` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E021_ladder` | A | keep | 4.21 GiB | 632 | 497 | 21 | `18833de839b732dd` | primary supervised ladder; its negative ablations are published evidence |
| `runs/E023_phase1` | A | keep | 33.1 MiB | 101 | 79 | 10 | `ecacc595a29edcae` | audit logs, re-scores and full-volume structure exports |
| `runs/E024_dual_prior` | A | keep | 1.19 GiB | 102 | 37 | 15 | `219acf67d8770265` | PR2 dual-prior ladder; its no-Wiener-gain result is the published evidence for plan SS8.2 (docs/PHASE1_PR2_E024_RESULTS.md) |
| `runs/E025_adaptive_dual_prior` | B | keep | 0.0 MiB | 2 | 0 | 5 | `f29153ad94ade55f` | PR3 y-only negative control; kept whatever sign it comes out with |
| `runs/E026_psras_condition` | B | keep | 643.2 MiB | 53 | 0 | 5 | `762139f55f00d4d6` | PR3 historical-condition re-check; its matched_synthetic null is evidence (docs/PHASE1_PR3_E026_RESULTS.md) |
| `runs/_smoke_pr3` | review | review | 30.0 MiB | 7 | 0 | 2 | `93a07060a4f985cb` | no rule covers this path; inspect before acting |
| `runs/pilot_S1_fold0` | D | review | 19.8 MiB | 7 | 5 | 5 | `522fbccf2b0e1937` | one-off pilot run before the E002 cohort existed; but 5 tracked file(s) refer to it, e.g. docs/archive/ARCHIVE_MANIFEST_20261110.md |
| `runs/smoke` | D | review | 19.8 MiB | 11 | 9 | 6 | `0adfe3f93fc552ec` | smoke-test run with random weights; runs/smoke/audit is the data audit; but 6 tracked file(s) refer to it, e.g. README.md |

## 引用明细

### `runs/E001_A0_fold0_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E001_baselines/E001_table.md`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`

### `runs/E001_A0_fold1_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E001_baselines/E001_table.md`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`

### `runs/E001_A0_fold2_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E001_baselines/E001_table.md`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`

### `runs/E001_A1_fold0_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E001_baselines/E001_table.md`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`

### `runs/E001_A1_fold1_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E001_baselines/E001_table.md`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`

### `runs/E001_A1_fold2_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E001_baselines/E001_table.md`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`

### `runs/E002_S0_fold0`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`
- `spur_mpm/figures/qualitative/qualitative_M1_FOVII_S1_vessels.json`
- `spur_mpm/scripts/audit_runs.py`
- `spur_mpm/scripts/qualitative_figures.py`

### `runs/E002_S0_fold0_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`

### `runs/E002_S0_fold1`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`
- `spur_mpm/figures/qualitative/qualitative_M2_FOVII_S1_vessels.json`

### `runs/E002_S0_fold1_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`

### `runs/E002_S0_fold2`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`
- `spur_mpm/figures/qualitative/qualitative_M3_FOVI_S1_vessels.json`

### `runs/E002_S0_fold2_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`

### `runs/E002_S1_fold0`

- `archive/selfsup_v1/figures/qualitative_M1_FOVII_S1_vessels_e012.json`
- `archive/selfsup_v1/figures/qualitative_M1_FOVII_S1_vessels_with_dc1.json`
- `archive/selfsup_v1/results/runs_e011/E002_S1_fold0_test_full/summary.json`
- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`
- `spur_mpm/figures/qualitative/qualitative_M1_FOVII_S1_vessels.json`
- `spur_mpm/tests/test_archive_safety.py`

### `runs/E002_S1_fold0_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`

### `runs/E002_S1_fold1`

- `archive/selfsup_v1/figures/qualitative_M2_FOVII_S1_vessels_e012.json`
- `archive/selfsup_v1/figures/qualitative_M2_FOVII_S1_vessels_with_dc1.json`
- `archive/selfsup_v1/results/runs_e011/E002_S1_fold1_test_full/summary.json`
- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`
- `spur_mpm/figures/qualitative/qualitative_M2_FOVII_S1_vessels.json`

### `runs/E002_S1_fold1_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`

### `runs/E002_S1_fold2`

- `archive/selfsup_v1/figures/qualitative_M3_FOVI_S1_vessels_e012.json`
- `archive/selfsup_v1/figures/qualitative_M3_FOVI_S1_vessels_with_dc1.json`
- `archive/selfsup_v1/results/runs_e011/E002_S1_fold2_test_full/summary.json`
- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`
- `spur_mpm/figures/qualitative/qualitative_M3_FOVI_S1_vessels.json`

### `runs/E002_S1_fold2_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`

### `runs/E009_A0_fold0_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`

### `runs/E009_A0_fold1_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`

### `runs/E009_A0_fold2_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`

### `runs/E009_S1_fold0`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/figures/qualitative/qualitative_M1_FOVII_S1_vessels_ky4.json`

### `runs/E009_S1_fold0_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`

### `runs/E009_S1_fold1`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/figures/qualitative/qualitative_M2_FOVII_S1_vessels_ky4.json`

### `runs/E009_S1_fold1_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`

### `runs/E009_S1_fold2`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/figures/qualitative/qualitative_M3_FOVI_S1_vessels_ky4.json`

### `runs/E009_S1_fold2_test`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`

### `runs/E021_ladder`

- `Project log/Project_log.md`
- `docs/CODE_MAP.md`
- `docs/PHASE1_ACCEPTANCE_REPORT.md`
- `docs/PHASE1_AUDIT_RESPONSE.md`
- `docs/PHASE1_PR2_ACCEPTANCE.md`
- `docs/PHASE1_PR2_IMPLEMENTATION_PLAN.md`
- `docs/PHASE1_SUPERVISED_AUDIT_AND_RUNBOOK.md`
- `docs/RESULTS.md`
- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/CLEANUP_REPORT_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/figures/framework/fig2_provenance.json`
- `spur_mpm/scripts/audit_runs.py`
- `spur_mpm/scripts/framework_figure.py`
- `spur_mpm/scripts/ladder_curves.py`
- `spur_mpm/scripts/ladder_report.py`
- `spur_mpm/scripts/run_ladder.py`
- `spur_mpm/scripts/structure_figure.py`
- `spur_mpm/scripts/structure_report.py`
- `spur_mpm/scripts/tolerance_sweep.py`
- `spur_mpm/tests/test_archive_safety.py`

### `runs/E023_phase1`

- `Project log/Project_log.md`
- `docs/PHASE1_ACCEPTANCE_REPORT.md`
- `docs/PHASE1_AUDIT_RESPONSE.md`
- `docs/PHASE1_PR2_IMPLEMENTATION_PLAN.md`
- `docs/RESULTS.md`
- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/CLEANUP_REPORT_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/reval_pre_fix_checkpoints.sh`
- `spur_mpm/scripts/audit_runs.py`

### `runs/E024_dual_prior`

- `Project log/Project_log.md`
- `docs/PHASE1_PR2_ACCEPTANCE.md`
- `docs/PHASE1_PR2_E024_RESULTS.md`
- `docs/PHASE1_PR2_IMPLEMENTATION_PLAN.md`
- `docs/PHASE1_PR2_PR_DRAFT.md`
- `docs/PHASE1_PR2_PSF_ESTIMATE.md`
- `docs/PHASE1_PR2_RUNBOOK.md`
- `docs/RESULTS.md`
- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/run_e024.sh`
- `spur_mpm/scripts/audit_runs.py`
- `spur_mpm/scripts/backfill_model_control.py`
- `spur_mpm/scripts/compare_structure.py`
- `spur_mpm/scripts/estimate_psf.py`

### `runs/E025_adaptive_dual_prior`

- `docs/PHASE1_PR3_IMPLEMENTATION_AND_RUNBOOK.md`
- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/run_e025.sh`
- `spur_mpm/scripts/audit_runs.py`

### `runs/E026_psras_condition`

- `docs/PHASE1_PR3_E026_RESULTS.md`
- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/experiments/run_e026_historical.sh`
- `spur_mpm/scripts/audit_runs.py`

### `runs/_smoke_pr3`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/cleanup_candidates.json`

### `runs/pilot_S1_fold0`

- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/CLEANUP_REPORT_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/scripts/audit_runs.py`
- `spur_mpm/tests/test_archive_safety.py`

### `runs/smoke`

- `README.md`
- `docs/archive/ARCHIVE_MANIFEST_20261110.md`
- `docs/archive/CLEANUP_REPORT_20261110.md`
- `docs/archive/cleanup_candidates.json`
- `spur_mpm/scripts/audit_runs.py`
- `spur_mpm/tests/test_archive_safety.py`

## 恢复方式

本清单不含任何已被删除的**唯一**副本：可删除项（`action=delete`）都是可再生的
smoke/pilot 产物，重建命令见各自 `config.yaml` 与实验脚本。E002/E009 的 checkpoint
只在 PR #1 B4 重评完成后才可能降级，且届时会先写入归档索引再动盘。
