#!/bin/bash
#
# train_m8_variants.sh —— 批量训练 M8 的多个变体：每个变体一张卡，卡数不够时排队；跑完汇总成一张表。
#
# 用法：
#   bash train_m8_variants.sh --splits_dir /data/xxx/splits                      # 跑 config/variants/ 下全部变体，全部可见卡
#   bash train_m8_variants.sh --splits_dir ... --gpus 0,1,2,3 --variants a1_hybrid,a2_slstm,a3_attention
#   bash train_m8_variants.sh --splits_dir ... --max_steps 50 --gpus 0 --batch_tag smoke        # 全部变体冒烟
#   bash train_m8_variants.sh --splits_dir ... --variants a1_hybrid,c1_no_cpm --seeds 0,1,2      # 每个变体 3 颗种子（tag 带 _s<seed>）
#   bash train_m8_variants.sh --splits_dir data/splits/m8_refit --variants e1_final --seeds 0,1,2 --refit --max_steps 5000 --batch_tag refit
#                                                                                   # 全量重训 3 颗种子（--refit 透传给 train_m8.sh）
#   bash train_m8_variants.sh --train_parquet "/data/train/*.parquet" --variants j1_final2,k1_hint01,k3_resid --max_steps 6000 --batch_tag full
#                                                                                   # 全量训练：现成的训练 parquet 直接吃，不切分、不剔行、无验证
# 产物：runs/m8/<batch_tag>/<变体>/…，汇总 runs/m8/<batch_tag>/variants_summary.csv
set -u
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$_HERE/forecast_backbones" ]; then BASE_DIR="$_HERE"; else BASE_DIR="$(cd "$_HERE/.." && pwd)"; fi
SPLITS_DIR="${BASE_DIR}/data/splits/m1_m4_v1"
TRAIN_PARQUET=""       # 给了就是全量训练（透传 --train_parquet，不用 --splits_dir）
VARIANT_DIR="${BASE_DIR}/forecast_backbones/m8/config/variants"
VARIANTS=""            # 逗号分隔的变体名（不带 .yaml）；留空 = 目录下全部
GPUS="auto"
MAX_STEPS=""
SEEDS=""             # 逗号分隔；留空 = 用配置里的 run.seeds[0]
BATCH_TAG="variants_$(date +%Y%m%d_%H%M%S)"
RUNS_BASE="${BASE_DIR}/runs"
VENV_PY="/home/ma-user/ts3/bin/python"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --splits_dir) SPLITS_DIR="$2"; shift 2 ;;
    --train_parquet) TRAIN_PARQUET="$2"; shift 2 ;;
    --variants)   VARIANTS="$2";   shift 2 ;;
    --variant_dir) VARIANT_DIR="$2"; shift 2 ;;
    --gpus)       GPUS="$2";       shift 2 ;;
    --max_steps)  MAX_STEPS="$2";  shift 2 ;;
    --seeds)      SEEDS="$2";      shift 2 ;;
    --batch_tag)  BATCH_TAG="$2";  shift 2 ;;
    --runs_base)  RUNS_BASE="$2";  shift 2 ;;
    --venv-python) VENV_PY="$2";   shift 2 ;;
    *) EXTRA+=("$1"); shift ;;                      # 其余参数原样传给 train_m8.sh（如 --precision、--batch_size）
  esac
done
if [ "$GPUS" = "auto" ]; then
  GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -)"
fi
[ -n "$GPUS" ] || { echo "❌ 没检测到 GPU；用 --gpus 指定" >&2; exit 1; }
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [ -z "$VARIANTS" ]; then
  FILES=( $(ls "$VARIANT_DIR"/*.yaml | sort) )
else
  FILES=(); IFS=',' read -r -a NAMES <<< "$VARIANTS"
  for n in "${NAMES[@]}"; do f="$VARIANT_DIR/${n%.yaml}.yaml"; [ -f "$f" ] || { echo "❌ 变体不存在: $f" >&2; exit 1; }; FILES+=("$f"); done
fi
OUT_BASE="$RUNS_BASE/m8/$BATCH_TAG"; mkdir -p "${OUT_BASE}"
# 任务 = 变体 × 种子；种子留空时一个变体一个任务
JOBS=(); JOB_SEEDS=()
if [ -n "$SEEDS" ]; then IFS=',' read -r -a SEED_LIST <<< "$SEEDS"; else SEED_LIST=(""); fi
for f in "${FILES[@]}"; do for sd in "${SEED_LIST[@]}"; do JOBS+=("$f"); JOB_SEEDS+=("$sd"); done; done
echo "变体 ${#FILES[@]} 个 × 种子 ${#SEED_LIST[@]} = 任务 ${#JOBS[@]} 个，GPU ${#GPU_LIST[@]} 张：${GPUS}；产物 ${OUT_BASE}"
# 每张卡一条队列：第 i 个任务分给第 i mod G 张卡，同一张卡上的任务顺序执行
G=${#GPU_LIST[@]}
pids=()
for gi in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$gi]}"
  (
    for ((i=gi; i<${#JOBS[@]}; i+=G)); do
      f="${JOBS[$i]}"; sd="${JOB_SEEDS[$i]}"; name="$(basename "$f" .yaml)${sd:+_s$sd}"
      echo "[gpu $gpu] ▶ $name"
      if [ -n "$TRAIN_PARQUET" ]; then src=(--train_parquet "$TRAIN_PARQUET"); else src=(--splits_dir "$SPLITS_DIR"); fi
      args=("${src[@]}" --config "$f" --gpus "$gpu" --runs_base "$OUT_BASE/.runs" --run_id "$name" --tag "$name"
            --port $((29700 + gi)) --venv-python "$VENV_PY" --overwrite)
      [ -n "$sd" ] && args+=(--seed "$sd")
      [ -n "$MAX_STEPS" ] && args+=(--max_steps "$MAX_STEPS")
      bash "$_HERE/train_m8.sh" "${args[@]}" "${EXTRA[@]}" > "$OUT_BASE/${name}.log" 2>&1 \
        && echo "[gpu $gpu] ✅ $name" || echo "[gpu $gpu] ❌ ${name}（看 $OUT_BASE/${name}.log）"
      # 把产物目录提到 OUT_BASE/<name>
      [ -d "$OUT_BASE/.runs/m8/$name" ] && { rm -rf "$OUT_BASE/$name"; mv "$OUT_BASE/.runs/m8/$name" "$OUT_BASE/$name"; }
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
rm -rf "$OUT_BASE/.runs"
cd "$BASE_DIR" && "$VENV_PY" -m forecast_backbones.m8.collect_runs "${OUT_BASE}" --out "$OUT_BASE/variants_summary.csv"
