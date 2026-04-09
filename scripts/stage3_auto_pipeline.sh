#!/usr/bin/env bash
set -euo pipefail

source /data2/wjb/anaconda3/etc/profile.d/conda.sh
conda activate threePM

ROOT="/data2/wjb/threePM"
TOPIC_DIR="$ROOT/projects/three-photon-spatiotemporal-sparse-sampling-fast-high-quality-reconstruction"
DATA_ROOT="$ROOT/ThreePM_openDatasets/CIDC25_paired_A1F0"
FIG_DIR="$TOPIC_DIR/figures_stage3"
LOG_DIR="$TOPIC_DIR/stage3_logs"
mkdir -p "$FIG_DIR" "$LOG_DIR"

RUN_NAME="stage3_2p5d_w3_e180_full"
EVAL_DIR="$ROOT/eval_results/${RUN_NAME}_eval"
SUMMARY_CSV="$EVAL_DIR/eval/summary_${RUN_NAME}.csv"
METRICS_CSV="$EVAL_DIR/eval/metrics_${RUN_NAME}.csv"

{
  echo "[Stage3] start: $(date)"
  echo "[Stage3] env: $(python -c 'import torch; print(torch.__version__)')"

  python scripts/check_eval_protocol.py --split_dir "$DATA_ROOT/Test" --stack_policy first

  CUDA_VISIBLE_DEVICES=2 python train.py \
    --task spatial_2p5d --model swin_unet_2d \
    --data_root "$DATA_ROOT" \
    --temporal_window 3 \
    --patch_size 256 --patches_per_image 8 \
    --batch_size 4 --epochs 180 --num_workers 4 \
    --split_val_from_train 0.1 \
    --run_name "$RUN_NAME"

  CUDA_VISIBLE_DEVICES=2 python eval.py \
    --task spatial_2p5d --model swin_unet_2d \
    --data_root "$DATA_ROOT" \
    --checkpoint "$ROOT/runs/$RUN_NAME/best.pth" \
    --output_dir "$EVAL_DIR" \
    --eval_split_dir "$DATA_ROOT/Test" \
    --temporal_window 3 \
    --eval_full_image

  python plot.py --mode stats \
    --metrics_csv "$METRICS_CSV" \
    --output_dir "$FIG_DIR/$RUN_NAME" --fmt png

  python plot.py --mode training \
    --tensorboard_dir "$ROOT/runs/$RUN_NAME" \
    --output_dir "$FIG_DIR/$RUN_NAME" --fmt png

  python - << 'PY'
import csv, json, subprocess
from pathlib import Path
base=Path('/data2/wjb/threePM/projects/three-photon-spatiotemporal-sparse-sampling-fast-high-quality-reconstruction')
run='stage3_2p5d_w3_e180_full'
summary=Path(f'/data2/wjb/threePM/eval_results/{run}_eval/eval/summary_{run}.csv')
report=base/'STAGE3_REPORT.md'
rows={}
with open(summary) as f:
    for r in csv.DictReader(f):
        rows[r['metric']]=r

def g(k):
    try:return float(rows[k]['mean'])
    except:return None
with open(report,'w') as f:
    f.write('# Stage-3 Report\n\n')
    f.write('- Run: stage3_2p5d_w3_e180_full\n')
    f.write('- Data: CIDC25_paired_A1F0 (full)\n')
    f.write('- Model: spatial_2p5d (w=3), 180 epochs\n\n')
    f.write('## Core Metrics\n')
    for k in ['psnr','ssim','ms_ssim','mae','lpips','delta_psnr','delta_ssim']:
        f.write(f'- {k}: {g(k)}\\n')
    f.write('\n## Decision Draft\n')
    f.write('- If delta_psnr and delta_ssim remain positive with stable quality, Stage-3 is considered complete.\n')

prompt=report.read_text()+"\n\n请严格审查：是否通过Stage-3里程碑（Yes/No+理由），若No请给出必须执行的下一阶段最小实验清单。"
payload={"model":"glm-5","messages":[{"role":"system","content":"你是严格科研审稿人。"},{"role":"user","content":prompt[:12000]}],"temperature":0.2,"max_tokens":1400}
out=subprocess.check_output(['curl','-s','https://open.bigmodel.cn/api/paas/v4/chat/completions','-H','Content-Type: application/json','-H','Authorization: Bearer 9486da3f2dda4de2832053de63641a1f.RNLNDRxXeUcqu9dN','-d',json.dumps(payload,ensure_ascii=False)]).decode('utf-8','ignore')
obj=json.loads(out)
review=obj['choices'][0]['message'].get('content') or obj['choices'][0]['message'].get('reasoning_content','')
(base/'STAGE3_REVIEW_GLM5.md').write_text(review)

# Auto next stage trigger + paper draft
next_actions = base/'STAGE3_NEXT_ACTIONS.md'
if ('Yes' in review) or ('通过' in review and '不通过' not in review):
    next_actions.write_text('Stage-3 reviewer result: PASS.\\nProceed to paper consolidation.\\n')
    paper=base/'PAPER_DRAFT_STAGE3.md'
    paper.write_text('# Paper Draft (Stage-3)\\n\\n## Title\\nThree-photon spatiotemporal sparse sampling reconstruction\\n\\n## Abstract\\n[Auto draft placeholder based on Stage-2/3 reports]\\n\\n## Main Results\\n- See STAGE2_EXTENDED_REPORT.md\\n- See STAGE3_REPORT.md\\n\\n## Reviewer Notes\\n- See STAGE3_REVIEW_GLM5.md\\n')
else:
    next_actions.write_text('Stage-3 reviewer result: NOT PASS.\\nAuto-continue to next-stage experiments required by reviewer.\\n')
PY

  echo "[Stage3] done: $(date)"
} | tee "$LOG_DIR/stage3_pipeline.log"
