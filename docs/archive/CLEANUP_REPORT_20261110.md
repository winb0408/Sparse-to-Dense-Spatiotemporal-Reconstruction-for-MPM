# 清理报告（CLEANUP_REPORT_20261110）

对应任务：PR #2 §T0 / §9、PR 正文 T6。基线 HEAD `3e2885a0788b424c7c6132b2f70daa628de1d577`
（分支 `feature/phase1-dual-prior-20261110`，base `PUR-MPM` @ `ed060f0`）。

**结论先说：本轮没有删除任何东西。** 盘点把 31 个 run 目录分成 A/B/C/D 四类，
筛出 **1 个**可删项（`runs/pilot_S1_fold0`，19.8 MiB，零引用、随机初始化、已被 E002
队列取代），它**仍然在盘上**——等审阅者在这份报告上签字、把
`docs/archive/cleanup_candidates.json` 作为 reviewed allowlist 交给
`--apply --manifest`，才轮到它。其余 30 个目录里，28 个有引用链或属于 A/B/C 类，
一个都不许动。

## 1. 盘点口径

工具：`spur_mpm/scripts/audit_runs.py`（**默认 dry-run，不写盘**）。

```bash
# 盘点（只读）
python -m spur_mpm.scripts.audit_runs --out-dir docs/archive
# 预演删除（仍不写盘）
python -m spur_mpm.scripts.audit_runs --allowlist docs/archive/cleanup_candidates.json
# 真正删除（必须先完成上两步并人工确认 allowlist）
python -m spur_mpm.scripts.audit_runs --apply \
    --allowlist docs/archive/cleanup_candidates.json \
    --manifest docs/archive/cleanup_candidates.json
```

判据不是名字，而是**引用链**：扫描全部被 git 跟踪的文本文件（`.md/.py/.sh/.yaml/
.json/.csv/.txt`，跳过 `runs/` 内部），任何对 `runs/<run_id>` 的提及都会把该项的
`action` 从 `delete` 降级为 `review`，**父目录被提及也算**——实际就是这么发生的
（`README.md` 提的是 `runs/smoke`，不是 `runs/smoke/smoke_SS1`）。

每项记录的字段：`path, category, category_meaning, reason, run_id, referenced_by,
sha256, sha256_kind, file_sha256, bytes, file_count, tracked_files, action,
archive_destination`。`sha256` 是**目录摘要**：对 `(相对路径, 字节数)` 排序后取
sha256（大目录 4.2 GiB，逐文件全量哈希在盘点阶段没有信息增量）；小文件（< 64 MiB）
另在 `file_sha256` 里逐文件记录。`tracked_files` 区分了“已 force-add 进 git 的只读
快照”和“只在本地”的文件——`.gitignore` 里 `runs/` 整目录被忽略，**不代表**里面的
文件都没被跟踪。

## 2. 盘点结果

| 类别 | 条目 | 字节 | 处置 |
|---|---|---|---|
| A 正确且支持结论 | 2（`E021_ladder`, `E023_phase1`） | 4.24 GiB | KEEP |
| B 正确但负结果 | 6（`E001_A0/A1_fold*_test`） | 0.9 MiB | KEEP |
| C 协议无效 | 21（`E002_*`, `E009_*`） | 0.62 GiB | KEEP-EVIDENCE |
| D smoke/临时 | 2（`pilot_S1_fold0`, `smoke`） | 39.6 MiB | 1 DELETE / 1 REVIEW |
| 合计 | 31 | 4.72 GiB | 删除 **0** |

仓库规模：`git ls-files` 919 项，其中 `runs/` 下 671 项（force-add 的只读快照）；
`runs/` 磁盘 4.8 GiB、859 个文件；`data_local/` 3.0 GiB（原始数据、缓存、索引、标定，
类别 E，**不纳入任何自动删除**）。

## 3. 逐项决策

### 3.1 保留（A/B，负结果与主结果）

| 路径 | 类别 | 为什么不能删 |
|---|---|---|
| `runs/E021_ladder` | A | 52 个 cell 的主表、曲线、定性图、结构报告；其中 S2g“成本涨、收益没到”是本轮唯一净结论，X0/U/Gate 无独立贡献是**负结果**，正是 §9.1 B 类要求保护的证据 |
| `runs/E023_phase1` | A | 数据审计、mask 交叉核验、重打分日志；`docs/RESULTS.md` 与 `docs/PHASE1_AUDIT_RESPONSE.md` 逐条引用 |
| `runs/E001_A0/A1_fold*_test` | B | 旧队列上的零参数基线，结论是“A0 与 A1 在恒等 PSF 下等价”，是 D6 的数学恒等式的实证 |

### 3.2 保留待用（C，协议无效但仍有用途）

`runs/E002_{S0,S1}_fold{0,1,2}[_test]`、`runs/E009_{A0,S1}_fold{0,1,2}[_test]` 共 21 项：
它们是 **Z 边界修复之前**的划分与权重，数字已从 `docs/RESULTS.md` 撤下，不再作为任何
结论来源（类别 C）。但它们同时是 PR #1 条目 **B4**“旧 checkpoint 在新代码上重评”
唯一能用的权重——`spur_mpm/experiments/reval_pre_fix_checkpoints.sh` 直接读
`runs/E002_S*_fold*/best.pt` 与 `runs/E009_S1_fold*/best.pt`。删了就没法回答
“只改边界、不改结论”这个问题。**因此在 B4 结案前它们的 `action` 是 `keep-evidence`
而不是 `delete`，这是刻意的。**

### 3.3 唯一可删项（D）

| 路径 | 大小 | 引用 | 依据 | 恢复方式 |
|---|---|---|---|---|
| `runs/pilot_S1_fold0` | 19.8 MiB（7 文件） | **0** | E002 队列建立前的一次性试训，`best.pt` 是当时的随机初始化收敛点，任何报告、脚本、图、config 都不引用它 | 用它的 `config.yaml` 重跑：`python -m spur_mpm.train --config runs/pilot_S1_fold0/config.yaml --arm S1 --fold 0 --seed 0` |

### 3.4 明确不删但标 REVIEW（D）

| 路径 | 大小 | 为什么不自动删 |
|---|---|---|
| `runs/smoke` | 19.8 MiB | 名字像 smoke，但内含 `runs/smoke/audit/`（`manifest.csv/json`、`split.json`、`audit_report.md`）——那是 E000 数据审计的产物，且 `README.md:57,69` 引用 `runs/smoke`。名字不是证据，引用链才是。要删只能删 `runs/smoke/smoke_SS1`（随机权重的冒烟训练），且必须先改 README 与 `docs/ARCHIVE` 的说法，属于另一次经过审阅的改动 |

## 4. 清理前后对照

本轮**未执行删除**，故前后一致：

| | 清理前 | 清理后（本轮） |
|---|---|---|
| `runs/` 条目 | 31 | 31 |
| `runs/` 体积 | 4.72 GiB | 4.72 GiB |
| `git ls-files` | 919 | 919 |
| `git ls-files runs` | 671 | 671 |
| `runs/` 文件数 | 859 | 859 |

`git diff --stat`（本轮新增的归档相关文件）：

```
 docs/archive/ARCHIVE_MANIFEST_20261110.md   | 生成（盘点清单）
 docs/archive/cleanup_candidates.json        | 生成（机器可读，含 sha256）
 docs/archive/CLEANUP_REPORT_20261110.md     | 本文件
 spur_mpm/scripts/audit_runs.py              | 新增（盘点/预演/受限删除）
 spur_mpm/tests/test_archive_safety.py       | 新增 21 条安全测试
```

## 5. 删不掉的规则（写进代码，不是写进文档）

`apply_plan()` 在碰盘之前逐条拒绝，任一触发即 `SafetyRefusal` 中止整批：

1. 路径必须是仓库相对路径，出现 `..` 直接拒绝（`runs/../data_local/x` 拒绝）；
2. 解析后的真实路径必须仍在仓库根内（符号链接跳出拒绝）；
3. 出现 `*?[]` 通配拒绝——不允许“删一堆”；
4. 必须在 `runs/` 下（缓存目录例外，且只到 `review`）；
5. 命中 protectlist 拒绝：`data_local`、`runs/E021_ladder`、`runs/E023_phase1`、
   `docs/RESULTS.md`、`docs/PHASE1_AUDIT_RESPONSE.md`、`docs/archive`、`spur_mpm`；
6. 必须出现在**已保存的** manifest 里，且该项 `action == "delete"`；
7. 提交删除时**重新检查** `referenced_by`，口径与盘点时不一致就中止；
8. `--apply` 必须同时给 `--allowlist` 与 `--manifest`，缺一个就 `SystemExit`；
9. 默认 dry-run：只打印将要删除的绝对路径，不写盘。

这九条各有测试，见 `spur_mpm/tests/test_archive_safety.py`（21 条）：类别判定、
父目录引用降级、目录摘要可重现、dry-run 不落盘、`..`/绝对路径/通配/符号链接越界、
protectlist、未列入 manifest、`action != delete`、apply 时引用复核、
allowlist 里混入 `keep` 项时被跳过。

## 6. 未解决 / 需要人决定的事

1. **`runs/pilot_S1_fold0` 删不删**：技术上零风险，但它也是“E002 之前我们试过什么”
   的记录。建议保留到 PR 合并后由项目负责人决定；要删就一条命令（见 §3.3），
   前提是先把 `cleanup_candidates.json` 作为 allowlist 复核。
2. **`runs/smoke/smoke_SS1`**：同上，需要先改 `README.md` 的引用。
3. **大 checkpoint 的长期归档位置**：PR #2 §9.2.5 要求用“用户指定的长期存储”，
   目前没有指定，所以本轮**不做任何外部迁移**，E002/E009 的 9 个 `best.pt` 原样保留。
4. **`data_local/`** 3.0 GiB 属类别 E，本工具完全不碰；它的备份策略不在本报告范围。
