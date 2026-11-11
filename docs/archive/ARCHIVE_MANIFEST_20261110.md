# 历史产物归档清单（ARCHIVE_MANIFEST_20261110）

> 由 `python -m spur_mpm.scripts.audit_runs --out-dir docs/archive` 生成，**默认 dry-run**。
> 本文件是只读盘点，不是删除记录；任何删除都要另一份 reviewed allowlist 和 `docs/archive/CLEANUP_REPORT_20261110.md`。

- 采集时间（UTC）：`2026-11-10T15:01:25Z`
- 分支：`feature/phase1-dual-prior-20261110`
- HEAD：`3e2885a0788b424c7c6132b2f70daa628de1d577`
- 条目：31　合计 4.72 GiB
- 处置分布：`{"delete": 1, "keep": 8, "keep-evidence": 21, "review": 1}`
- 类别分布：`{"A": 2, "B": 6, "C": 21, "D": 2}`

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
| `runs/E001` | B | keep | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E002` | C | keep-evidence | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E009` | C | keep-evidence | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/pilot_S1_fold0` | D | delete | one-off pilot run before the E002 cohort existed |
| `runs/smoke` | D | delete | smoke-test run with random weights; runs/smoke/audit is the data audit |

## 逐条盘点

| 路径 | 类别 | 处置 | 大小 | 文件数 | 已跟踪 | 被引用 | sha256（清单摘要） | 理由 |
|---|---|---|---|---|---|---|---|---|
| `runs/E001_A0_fold0_test` | B | keep | 0.2 MiB | 2 | 2 | 2 | `07a20ec99d5a5945` | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E001_A0_fold1_test` | B | keep | 0.2 MiB | 2 | 2 | 2 | `223c2251a1c77ed4` | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E001_A0_fold2_test` | B | keep | 0.1 MiB | 2 | 2 | 2 | `6924acc4da70fee0` | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E001_A1_fold0_test` | B | keep | 0.2 MiB | 2 | 2 | 2 | `9491d509371dd9d3` | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E001_A1_fold1_test` | B | keep | 0.2 MiB | 2 | 2 | 2 | `fe9a9e46225be7a9` | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E001_A1_fold2_test` | B | keep | 0.1 MiB | 2 | 2 | 2 | `f8e9830e6289ae14` | parameter-free baselines on the old cohort; superseded, still correct |
| `runs/E002_S0_fold0` | C | keep-evidence | 49.3 MiB | 8 | 5 | 4 | `e9bdb1bdf8fc1be1` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S0_fold0_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 2 | `9fc52688c8958146` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S0_fold1` | C | keep-evidence | 49.3 MiB | 8 | 5 | 3 | `1a8d0ad305f85448` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S0_fold1_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 2 | `8bea36f5d978ecfc` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S0_fold2` | C | keep-evidence | 49.3 MiB | 8 | 5 | 3 | `b05199766ba8c3fd` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S0_fold2_test` | C | keep-evidence | 0.1 MiB | 2 | 2 | 2 | `027eeea30086c7fa` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S1_fold0` | C | keep-evidence | 49.3 MiB | 8 | 5 | 6 | `f080c39cae53fd20` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S1_fold0_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 2 | `e79c05ef7b991b5a` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S1_fold1` | C | keep-evidence | 49.3 MiB | 8 | 5 | 6 | `821469084807edd3` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S1_fold1_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 2 | `283e9169e5e8a649` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S1_fold2` | C | keep-evidence | 49.3 MiB | 8 | 5 | 6 | `3f4de83b7566047c` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E002_S1_fold2_test` | C | keep-evidence | 0.1 MiB | 2 | 2 | 2 | `9ba2f750515e2bd3` | pre-Z-boundary-fix split, withdrawn from RESULTS.md; kept because PR #1 item B4 re-scores these checkpoints on the fixed code |
| `runs/E009_A0_fold0_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 0 | `60c73629c66eebbe` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_A0_fold1_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 0 | `b26802228fa5626c` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_A0_fold2_test` | C | keep-evidence | 0.1 MiB | 2 | 2 | 0 | `f0ede1504fad5913` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_S1_fold0` | C | keep-evidence | 49.3 MiB | 8 | 5 | 1 | `0879cfd50b51c397` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_S1_fold0_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 0 | `68e702ce5a121db9` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_S1_fold1` | C | keep-evidence | 49.3 MiB | 8 | 5 | 1 | `60cc63bcbef713e0` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_S1_fold1_test` | C | keep-evidence | 0.2 MiB | 2 | 2 | 0 | `4385dca13dfeb4be` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_S1_fold2` | C | keep-evidence | 49.3 MiB | 8 | 5 | 1 | `6e212ec506c1052d` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E009_S1_fold2_test` | C | keep-evidence | 0.1 MiB | 2 | 2 | 0 | `7c2d3d37f29b8bf9` | same cohort as E002 at k_y=4; kept for the same B4 comparison |
| `runs/E021_ladder` | A | keep | 4.21 GiB | 632 | 497 | 15 | `18833de839b732dd` | primary supervised ladder; its negative ablations are published evidence |
| `runs/E023_phase1` | A | keep | 33.1 MiB | 101 | 79 | 6 | `ecacc595a29edcae` | audit logs, re-scores and full-volume structure exports |
| `runs/pilot_S1_fold0` | D | delete | 19.8 MiB | 7 | 5 | 0 | `522fbccf2b0e1937` | one-off pilot run before the E002 cohort existed |
| `runs/smoke` | D | review | 19.8 MiB | 11 | 9 | 1 | `0adfe3f93fc552ec` | smoke-test run with random weights; runs/smoke/audit is the data audit; but 1 tracked file(s) refer to it, e.g. README.md |

## 引用明细

### `runs/E001_A0_fold0_test`

- `spur_mpm/experiments/E001_baselines/E001_table.md`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`

### `runs/E001_A0_fold1_test`

- `spur_mpm/experiments/E001_baselines/E001_table.md`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`

### `runs/E001_A0_fold2_test`

- `spur_mpm/experiments/E001_baselines/E001_table.md`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`

### `runs/E001_A1_fold0_test`

- `spur_mpm/experiments/E001_baselines/E001_table.md`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`

### `runs/E001_A1_fold1_test`

- `spur_mpm/experiments/E001_baselines/E001_table.md`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`

### `runs/E001_A1_fold2_test`

- `spur_mpm/experiments/E001_baselines/E001_table.md`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`

### `runs/E002_S0_fold0`

- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`
- `spur_mpm/figures/qualitative/qualitative_M1_FOVII_S1_vessels.json`
- `spur_mpm/scripts/qualitative_figures.py`

### `runs/E002_S0_fold0_test`

- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`

### `runs/E002_S0_fold1`

- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`
- `spur_mpm/figures/qualitative/qualitative_M2_FOVII_S1_vessels.json`

### `runs/E002_S0_fold1_test`

- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`

### `runs/E002_S0_fold2`

- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`
- `spur_mpm/figures/qualitative/qualitative_M3_FOVI_S1_vessels.json`

### `runs/E002_S0_fold2_test`

- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`

### `runs/E002_S1_fold0`

- `archive/selfsup_v1/figures/qualitative_M1_FOVII_S1_vessels_e012.json`
- `archive/selfsup_v1/figures/qualitative_M1_FOVII_S1_vessels_with_dc1.json`
- `archive/selfsup_v1/results/runs_e011/E002_S1_fold0_test_full/summary.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`
- `spur_mpm/figures/qualitative/qualitative_M1_FOVII_S1_vessels.json`

### `runs/E002_S1_fold0_test`

- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`

### `runs/E002_S1_fold1`

- `archive/selfsup_v1/figures/qualitative_M2_FOVII_S1_vessels_e012.json`
- `archive/selfsup_v1/figures/qualitative_M2_FOVII_S1_vessels_with_dc1.json`
- `archive/selfsup_v1/results/runs_e011/E002_S1_fold1_test_full/summary.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`
- `spur_mpm/figures/qualitative/qualitative_M2_FOVII_S1_vessels.json`

### `runs/E002_S1_fold1_test`

- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`

### `runs/E002_S1_fold2`

- `archive/selfsup_v1/figures/qualitative_M3_FOVI_S1_vessels_e012.json`
- `archive/selfsup_v1/figures/qualitative_M3_FOVI_S1_vessels_with_dc1.json`
- `archive/selfsup_v1/results/runs_e011/E002_S1_fold2_test_full/summary.json`
- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`
- `spur_mpm/figures/qualitative/qualitative_M3_FOVI_S1_vessels.json`

### `runs/E002_S1_fold2_test`

- `spur_mpm/experiments/E002_supervised/E001_vs_E002.md`
- `spur_mpm/experiments/E002_supervised/E002_table.md`

### `runs/E009_S1_fold0`

- `spur_mpm/figures/qualitative/qualitative_M1_FOVII_S1_vessels_ky4.json`

### `runs/E009_S1_fold1`

- `spur_mpm/figures/qualitative/qualitative_M2_FOVII_S1_vessels_ky4.json`

### `runs/E009_S1_fold2`

- `spur_mpm/figures/qualitative/qualitative_M3_FOVI_S1_vessels_ky4.json`

### `runs/E021_ladder`

- `Project log/Project_log.md`
- `docs/CODE_MAP.md`
- `docs/PHASE1_ACCEPTANCE_REPORT.md`
- `docs/PHASE1_AUDIT_RESPONSE.md`
- `docs/PHASE1_PR2_IMPLEMENTATION_PLAN.md`
- `docs/PHASE1_SUPERVISED_AUDIT_AND_RUNBOOK.md`
- `docs/RESULTS.md`
- `spur_mpm/figures/framework/fig2_provenance.json`
- `spur_mpm/scripts/framework_figure.py`
- `spur_mpm/scripts/ladder_curves.py`
- `spur_mpm/scripts/ladder_report.py`
- `spur_mpm/scripts/run_ladder.py`
- `spur_mpm/scripts/structure_figure.py`
- `spur_mpm/scripts/structure_report.py`
- `spur_mpm/scripts/tolerance_sweep.py`

### `runs/E023_phase1`

- `Project log/Project_log.md`
- `docs/PHASE1_ACCEPTANCE_REPORT.md`
- `docs/PHASE1_AUDIT_RESPONSE.md`
- `docs/PHASE1_PR2_IMPLEMENTATION_PLAN.md`
- `docs/RESULTS.md`
- `spur_mpm/experiments/reval_pre_fix_checkpoints.sh`

### `runs/smoke`

- `README.md`

## 恢复方式

本清单不含任何已被删除的**唯一**副本：可删除项（`action=delete`）都是可再生的
smoke/pilot 产物，重建命令见各自 `config.yaml` 与实验脚本。E002/E009 的 checkpoint
只在 PR #1 B4 重评完成后才可能降级，且届时会先写入归档索引再动盘。
