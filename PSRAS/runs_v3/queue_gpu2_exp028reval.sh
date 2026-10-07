#!/bin/bash
# EXP-028 reval chain (GPU2): for each finished U15W run -- 15-point dual-split
# reval + noise_stress 5-point x 3 stacks (EXP-027 protocol).
export CUDA_VISIBLE_DEVICES=2
export TMPDIR=/data2/wjb/tmp
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
cd /data2/wjb/threePM/PSRAS
LOG=runs_v3/queue_gpu2_exp028reval.log
PTS="y:2:0 y:4:0 y:8:0 yz2:2:2 yz2:2:4 yz2:2:8 yz2:4:2 yz2:4:4 yz2:8:2 xyz2:2:2 xyz2:2:4 xyz2:2:8 xyz2:4:2 xyz2:4:4 xyz2:8:2"
for run in UNIV4_U15W_s0 UNIV4_U15W_s1 UNIV4_U15W_s2; do
  [ -f "runs_v3/$run/final_val.json" ] || continue
  for pt in $PTS; do
    IFS=: read mode K kz <<< "$pt"
    if [ "$mode" = "y" ]; then tag="y_K${K}"; else tag="${mode}_K${K}x${kz}"; fi
    for split in val test; do
      out="runs_v3/$run/$( [ $split = val ] && echo val_metrics_corrected_${tag}.json || echo metrics_test_${tag}.json )"
      [ -s "$out" ] && continue
      $PY scripts/reval.py "runs_v3/$run" --split $split --eval_mode $mode \
        $( [ "$mode" = "y" ] && echo "--eval_K $K" || echo "--eval_K $K --eval_kz $kz" ) \
        > "runs_v3/$run/reval_${split}_${tag}.log" 2>&1
      [ -s "$out" ] && echo "[$(date +%m%d-%H:%M)] $run $split $tag ok" >> $LOG \
                  || echo "[$(date +%m%d-%H:%M)] $run $split $tag FAILED" >> $LOG
    done
  done
  for pt in y:2:0 y:4:0 y:8:0 yz2:4:4 xyz2:4:4; do
    IFS=: read mode K kz <<< "$pt"
    if [ "$mode" = "y" ]; then tag="y_K${K}"; else tag="${mode}_K${K}x${kz}"; fi
    for F in F1 F2 F3; do
      out="runs_v3/$run/metrics_noise_stress_${F}_${tag}.json"
      [ -s "$out" ] && continue
      $PY scripts/reval.py "runs_v3/$run" --noise_stack $F --eval_mode $mode \
        $( [ "$mode" = "y" ] && echo "--eval_K $K" || echo "--eval_K $K --eval_kz $kz" ) \
        > "runs_v3/$run/reval_ns_${F}_${tag}.log" 2>&1
      [ -s "$out" ] && echo "[$(date +%m%d-%H:%M)] $run NS $F $tag ok" >> $LOG \
                  || echo "[$(date +%m%d-%H:%M)] $run NS $F $tag FAILED" >> $LOG
    done
  done
  echo "[$(date +%m%d-%H:%M)] $run reval complete" >> $LOG
done
echo "[$(date +%m%d-%H:%M)] GPU2 EXP-028 reval chain DONE" >> $LOG
