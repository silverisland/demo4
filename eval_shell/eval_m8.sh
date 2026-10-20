#!/bin/bash
#
# eval_m8.sh —— M8 测试集评估：一个或多个训练产物目录（多个 = 三种子集成）→ 测试 parquet → 指标。
# 与 train_m8.sh 同一套约定（同一个 python、LD_LIBRARY_PATH）。推理很快，单卡即可。
#   默认（舰队快评）：10:00 行 + 整点行 → 南网 D+1（seen/unseen）/ nrmse → result_test.parquet + reports/
#   --full（逐站完整评测）：全部起报行 → ods_short / ods_ultra_short / nanwang_short_D+1 / nanwang_ultra_short
#          → metrics_ens/{站}-metrics.xlsx + {站}-summary.json（与 M1–M7 同构）+ reports/station_metrics.csv；
#          推理默认走线上接口 pipelines.pipeline_wrapper_inference.unified_inference（--engine native 走仓库内路径）
#
# 用法：
#   bash eval_m8.sh --runs runs/m8/round2a/a1_hybrid_s0 --test datasets/test
#   bash eval_m8.sh --runs runs/m8/final/e1_final_s0,runs/m8/final/e1_final_s1,runs/m8/final/e1_final_s2 \
#        --test "datasets/test/*.parquet" --out runs/m8/final/ens_eval
#   bash eval_m8.sh --runs runs/m8/round2a/a1_hybrid_s0 --test datasets/test --decision_yaml forecast_backbones/m8/config/variants/d2_pstar_point.yaml
#   bash eval_m8.sh --runs runs/m8/refit/e1_final_s0,runs/m8/refit/e1_final_s1,runs/m8/refit/e1_final_s2 \
#        --test "datasets/test/*.parquet" --out runs/m8/refit/full_eval --full
set -u
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$_HERE/forecast_backbones" ]; then BASE_DIR="$_HERE"; else BASE_DIR="$(cd "$_HERE/.." && pwd)"; fi

RUNS=""
TEST=""
OUT=""
POINT=""
DECISION_YAML=""
GPU="0"
BATCH_SIZE=""
TAG=""
ALLOW_CPU=0
FULL=0
ENGINE="unified"
VENV_PY="/home/ma-user/ts3/bin/python"
PRE_CUDA_LD="/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu"

usage() {
  cat <<'USAGE'
用法: bash eval_m8.sh --runs <dir[,dir...]> --test <目录|文件|通配符> [选项]
  --runs <dirs>            train_m8 的产物目录，逗号分隔多个 = 集成（分位平均后再决策）
  --test <path>            测试 parquet 的目录 / 文件 / 通配符（通配符加引号）
  --out <dir>              输出目录（默认 单成员 <run>/eval_test，多成员 <父目录>/ensemble_eval）
  --point <rule>           覆盖决策点规则：median / p_star / p_star_l1（清空 by_step）
  --decision_yaml <yaml>   合并该 yaml 的 decision 段（如 config/variants/d2_pstar_point.yaml）
  --gpu <id>               用哪张卡（默认 0）
  --batch_size <n>         推理 batch（默认配置 eval.batch_size）
  --tag <name>             写进 summary.csv 的名字
  --full                   逐站完整评测（四个指标 + {站}-metrics.xlsx / {站}-summary.json），见文件头
  --engine <unified|native|auto>   --full 的推理引擎（默认 unified = 线上接口 pipelines.pipeline_wrapper_inference）
  --venv-python <path>     python 解释器（默认 /home/ma-user/ts3/bin/python）
  --allow-cpu              没有 GPU 时用 CPU 跑
USAGE
}
while [ $# -gt 0 ]; do
  case "$1" in
    --runs)           RUNS="$2";          shift 2 ;;
    --test)           TEST="$2";          shift 2 ;;
    --out)            OUT="$2";           shift 2 ;;
    --point)          POINT="$2";         shift 2 ;;
    --decision_yaml)  DECISION_YAML="$2"; shift 2 ;;
    --gpu|--gpus)     GPU="$2";           shift 2 ;;
    --batch_size)     BATCH_SIZE="$2";    shift 2 ;;
    --tag)            TAG="$2";           shift 2 ;;
    --venv-python)    VENV_PY="$2";       shift 2 ;;
    --allow-cpu)      ALLOW_CPU=1;        shift   ;;
    --full)           FULL=1;             shift   ;;
    --engine)         ENGINE="$2";        shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done
[ -z "$RUNS" ] || [ -z "$TEST" ] && { echo "❌ --runs 和 --test 必给" >&2; usage; exit 1; }
[ -x "$VENV_PY" ] || { echo "❌ python 不存在: ${VENV_PY}" >&2; exit 1; }
cd "$BASE_DIR" || exit 1
export LD_LIBRARY_PATH="${PRE_CUDA_LD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if "$VENV_PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  export CUDA_VISIBLE_DEVICES="$GPU"; DEVICE="cuda"
elif [ "$ALLOW_CPU" -eq 1 ]; then DEVICE="cpu"; echo "⚠️  没检测到 GPU，按 --allow-cpu 用 CPU 跑"
else echo "❌ 没检测到 GPU；确要用 CPU 请加 --allow-cpu" >&2; exit 1; fi

IFS=',' read -r -a RUN_LIST <<< "$RUNS"
for r in "${RUN_LIST[@]}"; do
  [ -f "$r/model.safetensors" ] && [ -f "$r/model_config.yaml" ] || { echo "❌ ${r} 缺 model.safetensors / model_config.yaml" >&2; exit 1; }
done
args=(--runs "${RUN_LIST[@]}" --test "$TEST" --device "$DEVICE")
[ -n "$OUT" ] && args+=(--out "$OUT")
[ -n "$POINT" ] && args+=(--point "$POINT")
[ -n "$DECISION_YAML" ] && args+=(--decision-yaml "$DECISION_YAML")
[ -n "$BATCH_SIZE" ] && args+=(--batch-size "$BATCH_SIZE")
[ -n "$TAG" ] && args+=(--tag "$TAG")
MODULE="forecast_backbones.m8.evaluate"
if [ "$FULL" -eq 1 ]; then MODULE="forecast_backbones.m8.station_report"; args+=(--engine "$ENGINE"); fi
echo "=============================================================="
echo "M8 测试集评估$([ "$FULL" -eq 1 ] && echo "（逐站完整，engine=$ENGINE）")   成员 ${#RUN_LIST[@]} 个   device=${DEVICE}${POINT:+   point=$POINT}${DECISION_YAML:+   decision=$DECISION_YAML}"
echo "RUNS : ${RUNS}"
echo "TEST : ${TEST}"
echo "=============================================================="
"$VENV_PY" -m "$MODULE" "${args[@]}"
RC=$?
[ $RC -eq 0 ] && echo "✅ 评估完成" || echo "❌ 评估失败 (rc=$RC)"
exit $RC
