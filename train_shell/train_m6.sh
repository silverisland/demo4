#!/bin/bash
#
# train_m6.sh —— M6 路线 B（LoRA 微调，默认配置不做 conformal）正式训练脚本，默认用全部可见 GPU 做数据并行。
#
# 与 train_shell.sh（M1–M4）的对应关系：
#   - 同一个 python 解释器与同一条 LD_LIBRARY_PATH（否则 torch 报 Error 803、cuda 不可用，训练会悄悄落到 CPU）。
#   - 同一份切分目录（pipelines.time_splits 落的 train/valid.parquet + split_manifest.json）。
#   - 同样用 accelerate launch 起多进程；入口是 M6 自带的 forecast_backbones.m6.train（不走 train_script.py）。
#   - 配置同样只改运行时副本，不改仓库里的 yaml。
#
# 多卡：每张卡一个进程、各持一份模型、各吃不同的训练批，每步共 卡数 × batch_size 行。
#       验证打分、存最优权重与报表只在主进程（第一张卡）做，这期间其余卡等待。
#       训完主进程还要把验证集全部行推理一遍写 result_val_lora.parquet，只用第一张卡。
#
# 用法：
#   bash train_m6.sh --splits_dir /data/xxx/splits                  # 全部可见 GPU，按配置训（默认 4000 步，每 250 步验证）
#   bash train_m6.sh --gpus 0,1,2,3 --run_id lora_4gpu              # 指定用哪几张卡
#   bash train_m6.sh --gpus 0 --max_steps 2 --run_id smoke          # 单卡冒烟
#   bash train_m6.sh --max_steps 0                                  # 不限步数，按配置的 epochs_max 训
#
set -u

# 仓库根目录：脚本放在根目录或 train_shell/ 下都能找到
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$_HERE/forecast_backbones" ]; then BASE_DIR="$_HERE"; else BASE_DIR="$(cd "$_HERE/.." && pwd)"; fi

SPLITS_DIR="${BASE_DIR}/data/splits/m1_m4_v1"
CONFIG="${BASE_DIR}/forecast_backbones/m6/config/lora_no_conformal.yaml"
RUNS_BASE="${BASE_DIR}/runs"
RUN_ID="lora_$(date +%Y%m%d_%H%M%S)"
GPUS="auto"                         # auto = nvidia-smi 看到的全部卡
MAX_STEPS=""                        # 留空 = 配置里的 finetune.schedule.max_steps；0 = 不限（按 epochs_max）
EVAL_EVERY=""                       # 留空 = 配置里的 finetune.schedule.eval_every_steps
SAMPLING=""                         # 留空 = 配置里的 finetune.schedule.sampling
PRECISION="fp32"                    # V100 没有原生 bf16；A100/H100 上可传 --precision bf16
BATCH_SIZE=""                       # 留空 = 配置里的 finetune.schedule.batch_size（每张卡）
CHECKPOINT=""                       # 留空 = 配置里的 timesfm3.checkpoint
PORT=29600                          # 与 train_shell.sh 从 29500 起的端口错开
CPU_THREADS=""                      # 留空 = CPU 核数 / 进程数
OVERWRITE=0
ALLOW_CPU=0
VENV_PY="/home/ma-user/ts3/bin/python"
PRE_CUDA_LD="/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu"

usage() {
  cat <<'EOF'
用法:
  bash train_m6.sh [选项]

选项:
  --splits_dir <dir>       pipelines.time_splits 落的切分目录（默认 <仓库>/data/splits/m1_m4_v1）
  --config <yaml>          M6 配置（默认 forecast_backbones/m6/config/lora_no_conformal.yaml）
  --run_id <name>          产物目录名：<runs_base>/m6/<run_id>（默认 lora_<日期_时间>）
  --runs_base <dir>        产物根目录（默认 <仓库>/runs）
  --gpus <csv|auto>        用哪些卡，例如 0,1,2,3（默认 auto = 全部可见卡）；每张卡一个进程
  --gpu <id>               同 --gpus <id>（单卡）
  --max_steps <n>          训练步数上限（默认用配置，4000）；0 = 不限，按配置的 epochs_max 训
  --eval_every <n>         每多少步验证一次（默认用配置，250）
  --sampling <random|station_day>  训练行采样方式（默认用配置，random）
  --precision <fp32|bf16>  训练精度（默认 fp32；V100 不支持原生 bf16）
  --batch_size <n>         每张卡的 batch_size（覆盖 finetune.schedule.batch_size）
  --checkpoint <dir>       覆盖 timesfm3.checkpoint（TimesFM-3 权重目录）
  --port <n>               多进程通信端口（默认 29600）
  --cpu_threads <n>        每个进程的 CPU 线程数（默认 CPU 核数 / 进程数）
  --overwrite              产物目录已存在时删掉重训（默认拒绝覆盖）
  --venv-python <path>     python 解释器（默认 /home/ma-user/ts3/bin/python）
  --allow-cpu              检测不到 GPU 时用 CPU 单进程跑（很慢，仅排查用）
  -h|--help                显示本帮助
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --splits_dir)   SPLITS_DIR="$2";  shift 2 ;;
    --config)       CONFIG="$2";      shift 2 ;;
    --run_id)       RUN_ID="$2";      shift 2 ;;
    --runs_base)    RUNS_BASE="$2";   shift 2 ;;
    --gpus|--gpu)   GPUS="$2";        shift 2 ;;
    --max_steps)    MAX_STEPS="$2";   shift 2 ;;
    --eval_every)   EVAL_EVERY="$2";  shift 2 ;;
    --sampling)     SAMPLING="$2";    shift 2 ;;
    --precision)    PRECISION="$2";   shift 2 ;;
    --batch_size)   BATCH_SIZE="$2";  shift 2 ;;
    --checkpoint)   CHECKPOINT="$2";  shift 2 ;;
    --port)         PORT="$2";        shift 2 ;;
    --cpu_threads)  CPU_THREADS="$2"; shift 2 ;;
    --overwrite)    OVERWRITE=1;      shift   ;;
    --venv-python)  VENV_PY="$2";     shift 2 ;;
    --allow-cpu)    ALLOW_CPU=1;      shift   ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

# ============================================================
# 前置校验
# ============================================================
[ -f "$VENV_PY" ] || { echo "❌ python 解释器不存在: $VENV_PY" >&2; exit 1; }
[ -f "$CONFIG" ]  || { echo "❌ 找不到 M6 配置: $CONFIG" >&2; exit 1; }
for f in train.parquet valid.parquet split_manifest.json; do
  [ -f "$SPLITS_DIR/$f" ] || { echo "❌ 切分目录缺 $f: $SPLITS_DIR" >&2; exit 1; }
done
case "$PRECISION" in fp32|bf16) ;; *) echo "❌ --precision 只能是 fp32 或 bf16" >&2; exit 1 ;; esac
case "$SAMPLING" in ""|random|station_day) ;; *) echo "❌ --sampling 只能是 random 或 station_day" >&2; exit 1 ;; esac
for v in "$MAX_STEPS" "$EVAL_EVERY" "$BATCH_SIZE" "$PORT" "$CPU_THREADS"; do
  [ -z "$v" ] && continue
  case "$v" in *[!0-9]*) echo "❌ 数值参数须是非负整数: $v" >&2; exit 1 ;; esac
done

# 要用的卡
if [ "$GPUS" = "auto" ]; then
  GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -)"
fi
if [ -z "$GPUS" ]; then
  if [ "$ALLOW_CPU" -eq 1 ]; then
    echo "⚠️  没检测到 GPU，按 --allow-cpu 用 CPU 单进程跑，会很慢"
  else
    echo "❌ 没检测到 GPU（nvidia-smi 无输出）；用 --gpus 指定，或确要用 CPU 请加 --allow-cpu" >&2
    exit 1
  fi
fi
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NP="${#GPU_LIST[@]}"; [ "$NP" -ge 1 ] || NP=1
if [ -z "$CPU_THREADS" ]; then
  NCPU="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo "$NP")"
  CPU_THREADS=$(( NCPU / NP )); [ "$CPU_THREADS" -ge 1 ] || CPU_THREADS=1
fi

export LD_LIBRARY_PATH="${PRE_CUDA_LD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -n "$GPUS" ] && export CUDA_VISIBLE_DEVICES="$GPUS"
cd "$BASE_DIR" || exit 1

RUN_DIR="$RUNS_BASE/m6/$RUN_ID"
LOG_FILE="$BASE_DIR/logger/m6_${RUN_ID}_train.log"
TMP_CFG="$RUNS_BASE/.m6_cfg_${RUN_ID}.yaml"
mkdir -p "$RUNS_BASE" "$BASE_DIR/logger"

if [ -d "$RUN_DIR" ] && [ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]; then
  if [ "$OVERWRITE" -eq 1 ]; then
    rm -rf "$RUN_DIR"
  else
    echo "❌ 产物目录已存在且非空: ${RUN_DIR}（换 --run_id，或加 --overwrite 删掉重训）" >&2
    exit 1
  fi
fi

# torch 看到的卡数（在套好 LD_LIBRARY_PATH / CUDA_VISIBLE_DEVICES 之后查）
CUDA_INFO="$("$VENV_PY" -c "import torch; n = torch.cuda.device_count() if torch.cuda.is_available() else 0; print(n, torch.cuda.get_device_name(0) if n else 'cpu')" 2>&1 | tail -1)"
N_CUDA="${CUDA_INFO%% *}"
case "$N_CUDA" in ""|*[!0-9]*) N_CUDA=0 ;; esac
if [ -n "$GPUS" ] && [ "$N_CUDA" -lt "$NP" ]; then
  echo "❌ 要用 $NP 张卡（CUDA_VISIBLE_DEVICES=${GPUS}），torch 只看到 ${N_CUDA} 张（${CUDA_INFO}）。" \
       "检查 --gpus、LD_LIBRARY_PATH=${PRE_CUDA_LD}、--venv-python" >&2
  exit 1
fi

# 运行时配置副本：只改精度 / batch_size / checkpoint / 验证间隔 / 采样，不动仓库里的 yaml
"$VENV_PY" - "$CONFIG" "$TMP_CFG" "$PRECISION" "$BATCH_SIZE" "$CHECKPOINT" "$EVAL_EVERY" "$SAMPLING" <<'PYEOF'
import sys, yaml
src, dst, precision, batch_size, checkpoint, eval_every, sampling = sys.argv[1:8]
cfg = yaml.safe_load(open(src, encoding="utf-8")) or {}
sch = cfg.setdefault("finetune", {}).setdefault("schedule", {})
sch["precision"] = precision
if batch_size:
    sch["batch_size"] = int(batch_size)
if eval_every:
    sch["eval_every_steps"] = int(eval_every)
if sampling:
    sch["sampling"] = sampling
if checkpoint:
    cfg.setdefault("timesfm3", {})["checkpoint"] = checkpoint
yaml.safe_dump(cfg, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
import os
ck = cfg.get("timesfm3", {}).get("checkpoint")
meta = cfg.get("dataset", {}).get("station_meta")
print(f"CHECKPOINT={ck}"); print(f"STATION_META={meta}")
sys.exit(0 if (ck and os.path.exists(ck)) and (meta and os.path.exists(meta)) else 3)
PYEOF
RC=$?
if [ $RC -eq 3 ]; then
  echo "❌ 配置里的 timesfm3.checkpoint 或 dataset.station_meta 路径不存在（见上面两行）；权重路径可用 --checkpoint 覆盖" >&2
  rm -f "$TMP_CFG"; exit 1
elif [ $RC -ne 0 ]; then
  echo "❌ 生成运行时配置失败" >&2; rm -f "$TMP_CFG"; exit 1
fi

echo "=============================================================="
echo "M6 LoRA 训练"
echo "BASE_DIR   : $BASE_DIR"
echo "VENV_PY    : $VENV_PY"
echo "SPLITS_DIR : $SPLITS_DIR"
echo "CONFIG     : $CONFIG  (运行时副本: $TMP_CFG)"
echo "RUN_DIR    : $RUN_DIR"
if [ -n "$GPUS" ]; then
  echo "设备       : $NP 张卡 (CUDA_VISIBLE_DEVICES=$GPUS, $CUDA_INFO)   精度: $PRECISION   每进程 CPU 线程: $CPU_THREADS"
else
  echo "设备       : CPU 单进程   精度: fp32"
fi
case "$MAX_STEPS" in
  "") echo "max_steps  : 按配置" ;;
  0)  echo "max_steps  : 不限（按配置的 epochs_max）" ;;
  *)  echo "max_steps  : $MAX_STEPS" ;;
esac
echo "日志       : $LOG_FILE"
echo "=============================================================="

mkdir -p "$RUN_DIR"

cmd=(
  "$VENV_PY" -m accelerate.commands.launch
  --num_processes="$NP" --num_machines=1 --dynamo_backend=no --mixed_precision=no
  --main_process_port="$PORT" --num_cpu_threads_per_process="$CPU_THREADS"
)
[ "$NP" -gt 1 ] && cmd+=(--multi_gpu)
[ -z "$GPUS" ] && cmd+=(--cpu)
cmd+=(
  -m forecast_backbones.m6.train
  --config "$TMP_CFG"
  --splits-dir "$SPLITS_DIR"
  --out "$RUN_DIR"
)
[ -n "$MAX_STEPS" ] && cmd+=(--max-steps "$MAX_STEPS")

echo "CMD: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
rm -f "$TMP_CFG"

if [ $rc -ne 0 ]; then
  echo "❌ M6 训练失败 (exit=$rc)，日志: $LOG_FILE" >&2
  exit 1
fi
for f in model.safetensors model_config.yaml train_log.json; do
  [ -f "$RUN_DIR/$f" ] || { echo "❌ 训练结束但缺少 $RUN_DIR/$f" >&2; exit 1; }
done
echo "✅ M6 完成: $RUN_DIR/model.safetensors（推理配置 $RUN_DIR/model_config.yaml）"
echo "   测试集评测: CKPT_DIR=$RUN_DIR bash eval_shell/eval_m6.sh"
echo "   对比表: $RUN_DIR/reports/ladder_lora.csv、$RUN_DIR/reports/station_ladder.csv"
echo "   验证曲线: $RUN_DIR/train_log.json（history 每条一次验证）"
