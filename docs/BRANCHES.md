# 分支说明（2026-11-19 盘点）

> 这份文件回答一个问题：**现在到底哪条分支是主干、哪条在评审、哪条已经死了**。
> 结论按 2026-11-19 的远端实际状态写，不按计划写。

仓库：`git@github.com:winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM.git`

## 1. 主干是 `PUR-MPM`，不是 `main`

GitHub 的**默认分支是 `main`**，但 `main` 只有一条 2026-04-09 的初始化提交
（"Initialize threePM as an independent repository"）。本项目所有真实工作都在 **`PUR-MPM`**
上，包括已合并的 PR #1。

**建议**：把仓库默认分支从 `main` 改成 `PUR-MPM`（GitHub → Settings → Branches）。
不改也能工作，但每个新访客、每次 `git clone`、每个默认的 "compare" 按钮都会先落到一个空分支上，
这是"仓库看着乱"的第一个来源。**此项未执行**——它改的是仓库设置，需要负责人点头。

## 2. 评审中的两条分支（栈式 PR）

| PR | head 分支 | base 分支 | 相对 base | 状态 |
|---|---|---|---|---|
| [#2](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/pull/2) | `feature/phase1-dual-prior-20261110` | **`PUR-MPM`** | 26 提交 | Draft |
| [#3](https://github.com/winb0408/Sparse-to-Dense-Spatiotemporal-Reconstruction-for-MPM/pull/3) | `feature/phase1-adaptive-prior-pr3-20261112` | `feature/phase1-dual-prior-20261110` | 59 提交 | Draft |

PR3 建在 PR2 之上（`git merge-base --is-ancestor` 可验证），合计距离 `PUR-MPM` 85 个提交。
**正常路径**：PR2 被接受 → 把 PR3 rebase 到 `PUR-MPM` 并 retarget → 再审 PR3。
在 PR2 合并前把 PR3 的 base 直接改成 `PUR-MPM`，会让 PR3 的 diff 变成 85 个提交、
把 PR2 的全部改动重复显示一遍——这是"看起来乱"的第二个来源，而且会让评审无法只审 PR3。

**不要**在没有明确许可的情况下合并 PR2 或 PR3。

## 3. 历史分支（保留，不删除）

2026-11-19 负责人决定：**不删任何远端分支**，只在此说明用途。下面按"是否已被 `PUR-MPM` 包含"列出。

| 分支 | 最后提交 | 是否已含于 `PUR-MPM` | 用途 |
|---|---|---|---|
| `review/phase1-supervised-fixes-20261103` | 2026-11-03 | **是**（PR #1 已合并） | Phase 1 监督流水线修复 + 固定 GT 的 Y/Z/YZ 可视化校验；已并入主干，仅作历史指针 |
| `review-v4-20261019` | 2026-10-27 | 否（22 个提交不在主干） | v4 评审包：进度、主张与证据、图件、数字 |
| `PSRAS` | 2026-10-19 | 否（21 个提交不在主干） | v4 创新点的论文级图件集（`eval_viz/paper_figures/`） |
| `dev` | 2026-09-08 | 否（4 个提交不在主干） | 早期归档：稀疏评估协议、先验融合、报告 |

后三条**含主干里没有的提交**，删除会丢历史，这也是本次不删的理由之一。

## 4. 本地与远端的一处偏差

本地 `feature/phase1-dual-prior-20261110` 比 `origin` 多一个提交 `cbdc99c`
（"E024: why the Wiener arm equals the interpolation floor…"）。该内容已经以 `e8384ed`
的形式进入 PR3 分支，且 `git diff cbdc99c HEAD -- docs/PHASE1_PR2_E024_RESULTS.md` 为空，
即**内容完全重复**。因此：

- 不需要把它推到 `origin`（推了就是同一段文字两个提交）；
- 本地这条分支若要回到与远端一致，可以 `git reset --hard origin/feature/phase1-dual-prior-20261110`，
  不会丢任何内容。**本次未执行**，因为它是本地分支的重写，留给你决定。

## 5. 一条命令看清全貌

```bash
# 远端各分支与主干的距离
for b in $(git branch -r --format='%(refname:short)' | sed 's#^origin/##' | grep -v HEAD); do
  printf '%-46s ahead=%s behind=%s\n' "$b" \
    "$(git rev-list --count origin/PUR-MPM..origin/$b)" \
    "$(git rev-list --count origin/$b..origin/PUR-MPM)"
done
```
