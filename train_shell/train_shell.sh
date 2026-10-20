#!/bin/bash
#
# run_100train.sh —— 串行训练 M1 / M2_new / M3 / M4，然后（可选）跑融合模块 mF / mF_mean。
#
# 数据：
#   使用多站 100m 数据（train/valid/test.parquet，schema: general_ghi(_predict)/
#   general_temp(_predict)/observe_power/observe_power_future/observe_power_chronos_pred）。
#   三个文件按 --dataset_path = <目录> ，脚本内部拼接 <dir>/train.parquet 等。
#
# 步骤：
#   1. 对每个模型（m1 / m2_new / m3 / m4）跑 train_script.py —— 固定 RUN_ID。
#   2. 产物目录：$RUNS_BASE/{model}/{RUN_ID}/model_result/{model.safetensors, model_config.yaml}
#   3. （可选，RUN_FUSION=1）融合模块按
#        {base}/{name}/{run_id}/model_result/...  加载上述四个专家权重，再训练融合层。
#
# 用法示例：
#   bash train_shell/run_100train.sh                                    # 8 卡并行训 m1~m4（每模型 2 卡）
#   bash train_shell/run_100train.sh --gpus 0,1,2,3                     # 4 模型各绑 1 卡（0/1/2/3）
#   bash train_shell/run_100train.sh --gpus 0                          # 全部模型只用 GPU 0（1 卡冒烟）
#   bash train_shell/run_100train.sh --epochs 1 --gpus 0               # 1 epoch 冒烟，只用 1 卡
#   bash train_shell/run_100train.sh --fusion                           # 训练 + mF / mF_mean 融合
#   bash train_shell/run_100train.sh --models m1,m4 --dataset_path /data/100pretrain/100m_whole_dataset
#
# ★ 本机 CUDA 要点（重要）：
#   这台 8×V100S-32GB 机器默认 shell 环境下 torch 报 "Error 803"（cuda_available=False），
#   但这不是驱动坏，而是缺 LD_LIBRARY_PATH。训练前必须：
#     export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu
#   脚本内已对所有 accelerate 启动自动套用该环境（见 PRE_CUDA_LD 变量），
#   因此直接 `bash train_shell/run_100train.sh` 即可，无需手动 export。
#
# ★ 多卡策略：
#   --gpus <csv>  给 "每个模型" 分配 GPU 列表。模型个数 N，你给的总卡数 M，会被均分。
#   - M >= N：前 N 个模型各拿到 M 中 1 张卡，其余卡预留（融合/后续用）
#   - M > N*2：每模型可拿多卡（如 --gpus 0,1,2,3,4,5,6,7 → 4 模型 × 2 卡 DDP）
#   - --gpus auto：按 nvidia-smi 自动检测，N 个模型平分所有可见卡
#
set -u

# ============================================================
# 默认参数（可被命令行覆盖）
# ============================================================
# 仓库根目录：脚本位于 <根>/train_shell/run_100train.sh
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_PATH="${BASE_DIR}/datasets/splits"
RUNS_BASE="${BASE_DIR}/runs"
RUN_ID="pretrain_100m_v1"          # 四个专家共用同一个 run_id（融合层按专家名+run_id 定位）
MODELS="m1,m2_new,m3,m4"           # 参与训练的专家（逗号分隔）
FUSION_MODELS="mF,mF_mean"         # 训练完专家后要跑的融合模型（需 RUN_FUSION=1）
GPUS_SPEC="auto"                   # 向各模型分配 GPU 的 csv；"auto" = nvidia-smi 自动检测并均分
NUM_PROCESSES=""                   # 留空 = 由脚本按 “每模型卡数” 自动算（见下方 --num_processes 说明）
GPU=0                              # 兼容旧参数：--num_processes 1 时使用的单卡（CUDA_VISIBLE_DEVICES）
RUN_FUSION=0                       # 是否在专家训练后执行融合模块
EPOCHS=""                          # 留空 = 用各模型 config 里的 max_epochs/num_epochs
VENV_PY="/home/ma-user/ts3/bin/python"
# CUDA 可用化：torch 2.6.0+cu124 需显式加载 12.1 运行时库，否则 Error 803。
PRE_CUDA_LD="/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu"

# ============================================================
# 解析参数
# ============================================================
usage() {
  cat <<'EOF'
用法:
  bash train_shell/run_100train.sh [选项]

选项:
  --models <csv>           要训练的专家模型，默认 "m1,m2_new,m3,m4"
  --dataset_path <dir>     train/valid/test.parquet 所在目录，默认 /data/100pretrain/100m_whole_dataset
  --run_id <name>          四个专家共用的 run_id（默认 pretrain_100m_v1）
  --runs_base <dir>        模型产物根目录（默认 <仓库>/runs）
  --gpus <csv|auto>        要使用的 GPU 列表（默认 auto = nvidia-smi 自动检测并均分）。
                           例：--gpus 0,1,2,3 → 每个模型绑定 1 卡（0/1/2/3）；
                               --gpus 0,1,2,3,4,5,6,7 → 4 模型 × 各有 2 卡（DDP）；
                               --gpus 0 → 全部模型只用 GPU 0（1 卡冒烟）
  --num_processes <n>      accelerate 每模型进程数；留空 = 自动 = 每模型卡数。
                           显式给 1 时，配合 --gpus 单个卡号即可单卡冒烟。
  --gpu <id>               (兼容) 指定 --num_processes 1 时使用的单张卡
  --epochs <n>             覆盖训练轮数（小 epoch 测试用）；留空用各模型 config 默认值。
                           m1=max_epochs(m1 config) / m2_new=num_epochs / m3=epochs / m4=train_epochs
  --fusion                 训练完成后跑 mF / mF_mean 融合
  --fusion-models <csv>    融合模块列表（默认 "mF,mF_mean"，需 --fusion）
  --venv-python <path>     python 解释器（默认 /home/ma-user/ts3/bin/python）
  -h|--help                显示本帮助
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --models)            MODELS="$2";          shift 2 ;;
    --dataset_path)      DATASET_PATH="$2";    shift 2 ;;
    --run_id)            RUN_ID="$2";          shift 2 ;;
    --runs_base)         RUNS_BASE="$2";       shift 2 ;;
    --gpus)              GPUS_SPEC="$2";       shift 2 ;;
    --num_processes)     NUM_PROCESSES="$2";   shift 2 ;;
    --gpu)               GPU="$2";             shift 2 ;;
    --epochs)            EPOCHS="$2";          shift 2 ;;
    --fusion)            RUN_FUSION=1;         shift   ;;
    --fusion-models)     FUSION_MODELS="$2";   shift 2 ;;
    --venv-python)       VENV_PY="$2";         shift 2 ;;
    -h|--help)           usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a FUSION_LIST <<< "$FUSION_MODELS"

# ============================================================
# 多卡分配：决定每个模型用哪几张卡、每模型几个进程
# ============================================================
# - GPUS_SPEC=list  → 显式列表；GPUS_SPEC=auto → nvidia-smi 检测
# - N_MODELS 个模型，可用卡 C 张
#   * C >= N_MODELS：前 N_MODELS 个模型各 1 卡，其余卡预留（如 8 卡 4 模型 → 0/1/2/3，4-7 空闲）
#   * C <  N_MODELS：循环分配（每模型至少 1 卡）
# - NUM_PROCESSES 留空 → 自动 = 每模型分到的卡数（C==N_MODELS 时即 1）
nproc_probe() { false; }
detect_gpus() {
  if [ -n "$GPUS_SPEC" ] && [ "$GPUS_SPEC" != "auto" ]; then
    IFS=',' read -r -a _ALL_GPUS <<< "$GPUS_SPEC"
  else
    # nvidia-smi 查询可见设备（按 PCI bus id 排序，与 CUDA_VISIBLE_DEVICES 顺序一致）
    _ALL_GPUS=()
    while read -r line; do
      [ -n "$line" ] && _ALL_GPUS+=("$line")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
  fi
  if [ "${#_ALL_GPUS[@]}" -eq 0 ]; then
    echo "❌ 未检测到任何 GPU。若不使用 GPU（CPu fallback），请改用 CPU 专用脚本。" >&2
    return 1
  fi
}
detect_gpus || exit 1
TOTAL_GPUS="${#_ALL_GPUS[@]}"
N_MODELS="${#MODEL_LIST[@]}"
if [ "$N_MODELS" -le 0 ]; then echo "❌ MODELS 为空" >&2; exit 1; fi

# 每个模型分到的卡数（PER_MODEL_CARDS）：
#   - 默认：把总卡按模型数均分（尽可能让 8 卡 4 模型 = 每模型 2 卡全用上）
#   - 不可整除时：每模型 1 卡（剩余卡预留）
#   - 单模型：默认用全部卡（DDP）
if [ "$N_MODELS" -eq 1 ]; then
  PER_MODEL_CARDS="$TOTAL_GPUS"
elif [ $((TOTAL_GPUS % N_MODELS)) -eq 0 ] && [ $((TOTAL_GPUS / N_MODELS)) -ge 1 ]; then
  PER_MODEL_CARDS=$((TOTAL_GPUS / N_MODELS))
else
  PER_MODEL_CARDS=1
fi

# 若显式给了 NUM_PROCESSES，用它覆盖（每模型进程数 = 该模型可见卡数，须一致）
if [ -z "$NUM_PROCESSES" ]; then
  NUM_PROCESSES="$PER_MODEL_CARDS"
fi

# 为每个模型分配连续的 GPU 块（块大小 = PER_MODEL_CARDS），便于 CUDA_VISIBLE_DEVICES
declare -a MODEL_GPUS=()
for ((i=0; i<N_MODELS; i++)); do
  start=$((i * PER_MODEL_CARDS))
  block=""
  for ((j=0; j<PER_MODEL_CARDS; j++)); do
    idx=$((start + j))
    [ "$idx" -ge "$TOTAL_GPUS" ] && idx=$((i % TOTAL_GPUS))   # 越界兜底（理论上不会）
    if [ -n "$block" ]; then block="$block,${_ALL_GPUS[$idx]}"; else block="${_ALL_GPUS[$idx]}"; fi
  done
  MODEL_GPUS[i]="$block"
done

echo "── 资源 ──────────────────────────"
echo "可见 GPU 数 : $TOTAL_GPUS  (${_ALL_GPUS[*]})"
echo "模型数      : $N_MODELS (${MODEL_LIST[*]})"
echo "每模型卡数  : $PER_MODEL_CARDS ; accelerate 进程数/模型 = $NUM_PROCESSES"
for ((i=0; i<N_MODELS; i++)); do
  echo "  ${MODEL_LIST[i]} -> CUDA_VISIBLE_DEVICES=${MODEL_GPUS[i]}"
done
echo "──────────────────────────────────"

# ============================================================
# 前置校验
# ============================================================
: "${VENV_PY:?必须提供 VENV_PY}"   # (占位)

for m in "${MODEL_LIST[@]}"; do
  case "$m" in
    m1|m2_new|m3|m4) ;;
    *) echo "❌ 不支持的模型: $m（可选 m1/m2_new/m3/m4）" >&2; exit 1 ;;
  esac
done

if [ ! -d "$DATASET_PATH" ]; then
  echo "❌ 数据集目录不存在: $DATASET_PATH" >&2; exit 1
fi
# 两种数据形态任一满足即可：
#   1) 多站目录:  $DATASET_PATH/{train,valid,test}/  各含若干 station=*.parquet
#   2) 单文件:    $DATASET_PATH/{train,valid,test}.parquet
_dir_ok() { [ -d "$DATASET_PATH/$1" ] && [ -n "$(ls -A "$DATASET_PATH/$1" 2>/dev/null)" ]; }
_file_ok() { [ -f "$DATASET_PATH/$1.parquet" ]; }
for split in train valid test; do
  if _dir_ok "$split" || _file_ok "$split"; then
    :
  else
    echo "❌ 缺少数据集($split): 需要目录 $DATASET_PATH/$split/ 或文件 $DATASET_PATH/$split.parquet" >&2
    exit 1
  fi
done
if [ ! -x "$VENV_PY" ]; then  # -x 可能因路径为软链失败，兜底检查文件存在
  [ -f "$VENV_PY" ] || { echo "❌ python 解释器不存在: $VENV_PY" >&2; exit 1; }
fi

# m1 使用仓库内的正式配置 forecast_backbones/m1/config/configs.yaml（已含最优超参）。
# 说明：head_emb_size 完全以 configs.yaml 里 model_hyperparameter.head_emb_size 的值为准，
#       脚本只补顶层 device 键（见 train_expert）。
M1_CONFIG="$BASE_DIR/forecast_backbones/m1/config/configs.yaml"
[ -f "$M1_CONFIG" ] || { echo "❌ 找不到 M1 配置: $M1_CONFIG" >&2; exit 1; }

mkdir -p "$RUNS_BASE"
mkdir -p "$BASE_DIR/logger"

cd "$BASE_DIR" || exit 1

echo "=============================================================="
echo "100m 多站训练流水线"
echo "BASE_DIR      : $BASE_DIR"
echo "VENV_PY       : $VENV_PY"
echo "DATASET_PATH  : $DATASET_PATH"
echo "RUNS_BASE     : $RUNS_BASE"
echo "RUN_ID        : $RUN_ID"
echo "模型          : ${MODELS}"
echo "GPU 分配      : ${MODEL_GPUS[*]}  (每模型 $PER_MODEL_CARDS 卡)"
echo "num_processes : $NUM_PROCESSES/模型    LD_LIBRARY_PATH: ${PRE_CUDA_LD%%:*}"
echo "FUSION(RUN_FUSION=$RUN_FUSION): ${FUSION_MODELS:-无}"
[ -n "$EPOCHS" ] && echo "EPOCHS 覆盖   : $EPOCHS"
echo "=============================================================="

# ============================================================
# 主函数：训练单个专家
#   $1 = model_name ; $2 = config_path
# ============================================================
train_expert() {
  local model_name="$1"
  local config_path="$2"
  local run_dir="$RUNS_BASE/$model_name/$RUN_ID"
  local log_file="$BASE_DIR/logger/${model_name}_${RUN_ID}_train.log"
  local tmp_cfg=""          # 运行时 config 副本（用于 sed 打补丁）
  local gpu_id="${MODEL_GPUS[$IDX]}"   # 本模型使用哪些卡（csv）

  echo ""
  echo ">>> [train] $model_name  (config: $config_path)  GPU=$gpu_id"

  # ---- GPU + CUDA 环境：torch 2.6.0+cu124 需要显式 12.1 运行时库，否则 Error 803 ----
  export LD_LIBRARY_PATH="${PRE_CUDA_LD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  # ---- 生成运行时 config 副本，用于按模型 patch 必要的键 ----
  tmp_cfg="${RUNS_BASE}/.run100_cfg_${model_name}.yaml"
  cp "$config_path" "$tmp_cfg"

  # 1) 补 / 改 device 键（train_script.py 的 build_train_args 按模型读不同位置）：
  #    m1          -> 顶层 device（configs.yaml 没有，须补）
  #    m2_new/m4   -> 顶层 device / optimization.device（config 本身已有）
  #    m3          -> hyperparameter.device（已有）
  case "$model_name" in
    m1|m2_new)
      if ! grep -qE "^device:" "$tmp_cfg"; then
        printf 'device: cuda\n' >> "$tmp_cfg"
      fi
      ;;
  esac

  # 2) --epochs 覆盖：只在运行时副本里改对应模型的 epoch 键（不改仓库原 config）
  if [ -n "$EPOCHS" ]; then
    case "$model_name" in
      m1)        KEY_LINE="model_trainer:"; VAL_INDENT="  " ;;
      m2_new)    KEY_LINE="num_epochs:";    VAL_INDENT="" ;;
      m3)        KEY_LINE="hyperparameter:"; VAL_INDENT="  " ;;
      m4)        KEY_LINE="optimization:"; VAL_INDENT="  " ;;
    esac
    case "$model_name" in
      m1)   # model_trainer: max_epochs: N (顶层/缩进皆可)
        sed -i -E "s/^([[:space:]]*)max_epochs:[[:space:]]*[0-9]+/\1max_epochs: $EPOCHS/" "$tmp_cfg"
        ;;
      m2_new)   # 顶层 num_epochs: N
        sed -i -E "s/^[[:space:]]*num_epochs:[[:space:]]*[0-9]+/num_epochs: $EPOCHS/" "$tmp_cfg"
        ;;
      m3)   # hyperparameter: epochs: N —— 键在 config 中唯一，直接整文件替换即可
        if ! sed -i -E "s/^([[:space:]]*)epochs:[[:space:]]*[0-9]+/\1epochs: $EPOCHS/" "$tmp_cfg" \
             | grep -q "epochs: $EPOCHS"; then
          :  # sed -i 无输出；改后验证失败才提示
        fi
        grep -qE "^[[:space:]]*epochs:[[:space:]]*$EPOCHS([[:space:]]*#.*)?$" "$tmp_cfg" || \
          echo "  ⚠️ [m3 epochs] 未能确认 epochs 已被改为 $EPOCHS"
        ;;
      m4)   # optimization: train_epochs: N —— 键在 config 中唯一，直接整文件替换
        sed -i -E "s/^([[:space:]]*)train_epochs:[[:space:]]*[0-9]+/\1train_epochs: $EPOCHS/" "$tmp_cfg"
        grep -qE "^[[:space:]]*train_epochs:[[:space:]]*$EPOCHS([[:space:]]*#.*)?$" "$tmp_cfg" || \
          echo "  ⚠️ [m4 epochs] 未能确认 train_epochs 已被改为 $EPOCHS"
        ;;
    esac
    echo "    [epochs override] $model_name 设 $EPOCHS epoch（运行时副本: $tmp_cfg）"
  fi

  rm -rf "$run_dir"
  mkdir -p "$run_dir"

  # accelerate 进程数：--num_processes 1（单卡）或按模型卡数（多卡 DDP）
  local np="$NUM_PROCESSES"
  [ "${np}" -lt 1 ] && np=1

  # 每个模型独占一个 main_process_port，避免并行 launch 抢 29500 端口
  local port=$((29500 + IDX * 10))

  local cmd=(
    env LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    CUDA_VISIBLE_DEVICES="$gpu_id"
    "$VENV_PY" -m accelerate.commands.launch
    --num_processes="$np" --num_machines=1 --dynamo_backend=no
    --mixed_precision=no
    --main_process_port="$port"
    "$BASE_DIR/train_script.py"
    --model_name "$model_name"
    --model_config_path "$tmp_cfg"
    --dataset_path "$DATASET_PATH"
    --run_id "$RUN_ID"
    --run_path "$run_dir"
  )

  echo "    CMD: ${cmd[*]}"
  "${cmd[@]}" > "$log_file" 2>&1
  local rc=$?

  # 运行结束清理运行时副本（保留便于排查，需保留则注释下一行）
  rm -f "$tmp_cfg"

  if [ $rc -ne 0 ]; then
    echo "  ❌ $model_name 训练失败 (exit=$rc)，日志: $log_file" >&2
    tail -20 "$log_file" >&2
    return 1
  fi
  if [ ! -f "$run_dir/model_result/model.safetensors" ]; then
    echo "  ❌ $model_name 训练结束但缺少 model.safetensors" >&2
    return 1
  fi
  echo "  ✅ $model_name 完成: $run_dir/model_result/model.safetensors"
  return 0
}

# ============================================================
# 主流水线 1：训练专家 —— 多模型并行（每个 train_expert 后台运行，wait 汇总）
# ============================================================
PASS=0; FAIL=0
declare -a EXPERT_PIDS=()
IDX=0
for MODEL in "${MODEL_LIST[@]}"; do
  case "$MODEL" in
    m1) CONFIG_PATH="$M1_CONFIG" ;;              # 用 m1/config/configs.yaml（脚本只补 device）
    m2_new) CONFIG_PATH="$BASE_DIR/forecast_backbones/m2_new/config/configs.yaml" ;;
    m3)     CONFIG_PATH="$BASE_DIR/forecast_backbones/m3/config/configs.yaml" ;;
    m4)     CONFIG_PATH="$BASE_DIR/forecast_backbones/m4/config/configs.yaml" ;;
  esac
  train_expert "$MODEL" "$CONFIG_PATH" &
  EXPERT_PIDS+=("$!")
  IDX=$((IDX + 1))
done

# 等待所有并行专家完成
for pid in "${EXPERT_PIDS[@]}"; do
  wait "$pid"
  if [ $? -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
done
echo ""
echo "========== 专家训练结束：通过 $PASS，失败 $FAIL =========="
[ $FAIL -gt 0 ] && echo "（存在失败，融合阶段按需可只跑成功的专家？建议先修复失败模型再融合）"

# ============================================================
# 主流水线 2：融合（可选）
# ============================================================
if [ "$RUN_FUSION" -eq 1 ]; then
  # 校验四个专家产物齐全
  MISSING=0
  for m in m1 m2_new m3 m4; do
    if [ ! -f "$RUNS_BASE/$m/$RUN_ID/model_result/model.safetensors" ]; then
      echo "  ⚠️  融合前置失败：缺少 $RUNS_BASE/$m/$RUN_ID/model_result/model.safetensors" >&2
      MISSING=$((MISSING + 1))
    fi
  done
  if [ $MISSING -gt 0 ]; then
    echo "❌ 专家产物不齐（缺 $MISSING 个），跳过融合。" >&2
    exit 1
  fi

  # 融合模块各自走 train_script.py --model_name mF / mF_mean
  FUSION_GPU="${MODEL_GPUS[0]:-0}"   # 复用第一块 GPU 分配（或 0）
  for FM in "${FUSION_LIST[@]}"; do
    echo ""
    echo ">>> [fusion] $FM  (GPU=$FUSION_GPU)"
    local_run_dir="$RUNS_BASE/$FM/$RUN_ID"
    mkdir -p "$local_run_dir"
    local_log="$BASE_DIR/logger/${FM}_${RUN_ID}_train.log"

    local cmd=(
      env LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
      CUDA_VISIBLE_DEVICES="$FUSION_GPU"
      "$VENV_PY" -m accelerate.commands.launch
      --num_processes=1 --num_machines=1
      --mixed_precision=no --dynamo_backend=no
      "$BASE_DIR/train_script.py"
      --model_name "$FM"
      --dataset_path "$DATASET_PATH"
      --run_id "$RUN_ID"
      --run_path "$local_run_dir"
    )
    echo "    CMD: ${cmd[*]}"
    "${cmd[@]}" > "$local_log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "$local_run_dir/model_result/model.safetensors" ]; then
      echo "  ✅ $FM 融合完成"
    else
      echo "  ❌ $FM 融合失败 (exit=$rc)，日志: $local_log" >&2
      tail -20 "$local_log" >&2
    fi
  done
fi

echo ""
echo "=============================================================="
echo "全部结束。产物目录: $RUNS_BASE"
echo "日志目录: $BASE_DIR/logger/"
echo "=============================================================="