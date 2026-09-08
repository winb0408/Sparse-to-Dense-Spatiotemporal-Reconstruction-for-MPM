#!/usr/bin/env python3
"""
Quick snapshot / auto-updater for MDC25 K=2 training progress.

Examples:
  python scripts/mdc25_progress_snapshot.py
  python scripts/mdc25_progress_snapshot.py --write-log
  python scripts/mdc25_progress_snapshot.py --watch --interval 120 --write-log
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import time
from pathlib import Path


RUNS = [
    "sparse3d_2d_xy_K2_wiener_mdc25",
    "sparse3d_2p5d_xy_K2_mask_wiener_mdc25",
    "sparse3d_3d_xy_K2_wiener_mdc25",
    "sparse3d_2d_z_K2_wiener_mdc25",
    "sparse3d_2p5d_z_K2_mask_wiener_mdc25",
    "sparse3d_3d_z_K2_wiener_mdc25",
    "sparse3d_2d_xyz_K2_wiener_mdc25",
    "sparse3d_2p5d_xyz_K2_mask_wiener_mdc25",
    "sparse3d_3d_xyz_K2_wiener_mdc25",
]


def get_active_train_cmds() -> list[str]:
    out = subprocess.check_output(["ps", "-eo", "pid,etime,cmd"], text=True)
    lines = []
    for ln in out.splitlines():
        if "python train.py" in ln and "_mdc25" in ln:
            lines.append(ln.strip())
    return lines


def get_queue_cmds() -> list[str]:
    out = subprocess.check_output(["ps", "-eo", "pid,etime,cmd"], text=True)
    lines = []
    for ln in out.splitlines():
        if "run_mdc25_k2_optimized.sh" in ln and "python - <<'PY'" not in ln:
            lines.append(ln.strip())
    return lines


def parse_latest_epoch(log_path: Path) -> tuple[int | None, int | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"Epoch\s+(\d+)/(\d+)", text)
    if not matches:
        return None, None
    cur, total = matches[-1]
    return int(cur), int(total)


def parse_running_run(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"\[RUN GPU\d+\]\s+([^\n\r]+)", text)
    if not matches:
        return None
    return matches[-1].strip()


def collect_status(repo: Path, terminal_log: Path) -> dict:
    runs_root = repo / "runs"

    done = 0
    rows = []
    for run in RUNS:
        d = runs_root / run
        best = (d / "best.pth").exists()
        last = (d / "last.pth").exists()
        epochs = len(list(d.glob("epoch_*.pth"))) if d.exists() else 0
        if best:
            done += 1
        rows.append((run, best, last, epochs))

    active_train_cmds = get_active_train_cmds()
    queue_cmds = get_queue_cmds()
    running_run = parse_running_run(terminal_log)
    ep_now, ep_total = parse_latest_epoch(terminal_log)

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "now": now,
        "rows": rows,
        "done": done,
        "total": len(RUNS),
        "running_run": running_run,
        "ep_now": ep_now,
        "ep_total": ep_total,
        "active_train_cmds": active_train_cmds,
        "queue_cmds": queue_cmds,
    }


def print_console(status: dict) -> None:
    active_train_cmds = status["active_train_cmds"]
    running_run = status["running_run"]
    ep_now = status["ep_now"]
    ep_total = status["ep_total"]
    rows = status["rows"]

    print("=== MDC25 K=2 Progress Snapshot ===")
    print(f"Time     : {status['now']}")
    print(f"Completed: {status['done']}/{status['total']}")
    if running_run:
        if ep_now and ep_total:
            print(f"Running  : {running_run} (Epoch {ep_now}/{ep_total})")
        else:
            print(f"Running  : {running_run}")
    else:
        print("Running  : none detected from queue log")
    print(f"Active train.py processes: {len(active_train_cmds)}")
    for ln in active_train_cmds[:3]:
        print(f"  - {ln}")

    print("\nPer-run status:")
    for run, best, last, epochs in rows:
        status = "DONE" if best else "PENDING"
        print(
            f"- {run}: {status} | best={str(best).lower()} | "
            f"last={str(last).lower()} | epoch_ckpts={epochs}"
        )


def render_auto_block(status: dict) -> str:
    running = "none"
    if status["running_run"]:
        running = status["running_run"]
        if status["ep_now"] and status["ep_total"]:
            running = f"{running} (Epoch {status['ep_now']}/{status['ep_total']})"

    lines = [
        "## 自动更新进度",
        "",
        f"- 更新时间：{status['now']}",
        f"- 已完成：{status['done']}/{status['total']}",
        f"- 当前运行：`{running}`",
        f"- 活跃训练进程数：{len(status['active_train_cmds'])}",
        f"- 队列脚本进程数：{len(status['queue_cmds'])}",
        "",
        "### 每个 Run 状态",
        "",
    ]
    for run, best, last, epochs in status["rows"]:
        st = "DONE" if best else "PENDING"
        lines.append(
            f"- `{run}`: {st} | best={str(best).lower()} | "
            f"last={str(last).lower()} | epoch_ckpts={epochs}"
        )
    lines.append("")
    lines.append("> 该区块由脚本自动维护，请勿手动编辑。")
    lines.append("")
    return "\n".join(lines)


def write_log_block(log_path: Path, status: dict) -> None:
    start_marker = "<!-- AUTO_PROGRESS_START -->"
    end_marker = "<!-- AUTO_PROGRESS_END -->"
    block = (
        f"{start_marker}\n"
        f"{render_auto_block(status)}\n"
        f"{end_marker}\n"
    )
    if log_path.exists():
        content = log_path.read_text(errors="ignore")
    else:
        content = "# MDC25 K=2 训练进度日志\n\n"
    if start_marker in content and end_marker in content:
        pattern = re.compile(
            rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}\n?",
            flags=re.S,
        )
        content = pattern.sub(block, content)
    else:
        if not content.endswith("\n"):
            content += "\n"
        content += "\n" + block
    log_path.write_text(content)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MDC25 progress snapshot/updater")
    p.add_argument(
        "--terminal-log",
        type=str,
        default="/home/wjb/.cursor/projects/data2-wjb-threePM/terminals/276777.txt",
        help="Queue terminal log file path.",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default="/data2/wjb/threePM/runs/MDC25_K2_TRAINING_LOG.md",
        help="Markdown log file to update.",
    )
    p.add_argument("--write-log", action="store_true", help="Write status into log file.")
    p.add_argument("--watch", action="store_true", help="Keep updating periodically.")
    p.add_argument("--interval", type=int, default=120, help="Watch interval in seconds.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path("/data2/wjb/threePM")
    terminal_log = Path(args.terminal_log)
    log_path = Path(args.log_file)

    while True:
        status = collect_status(repo, terminal_log)
        print_console(status)
        if args.write_log:
            write_log_block(log_path, status)
        if not args.watch:
            break
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
