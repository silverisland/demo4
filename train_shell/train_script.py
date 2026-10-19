"""
Transformer_trainer_script.py

—— 按照配置一键启动训练，训练完成之后进行测试集推理。
—— 后续需要加入参数列表，提供调用时修改参数的能力。

"""
import os, argparse, sys
import torch
from pathlib import Path

import pandas as pd
from datetime import datetime

BASE_PATH = str(Path(__file__).resolve().parent)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 共有文件
from pipelines.transformer_dataset import UnifiedDataset
# from pipelines.distribute_utils import PartialState, print_main, resolve_run_id, run_main_process_step
from pipelines.model_registry import build_model, MODEL_REGISTRY

def build_train_args(MODEL_NAME, model_config):

    training_args, device = None, None
    if MODEL_NAME == "m1":
        from forecast_backbones.m1.transformer_trainer import M1_build_training_args
        training_args = M1_build_training_args(model_config, result_dir = RUN_PATH)
        device = model_config["device"]
    elif MODEL_NAME == "m2_new":
        from forecast_backbones.m2_new.transformer_trainer import M2_build_training_args
        training_args = M2_build_training_args(model_config, result_dir = RUN_PATH)
        device = model_config["device"]
    elif MODEL_NAME == "m2":
        from forecast_backbones.m2.transformer_trainer import M2_build_training_args
        training_args = M2_build_training_args(model_config, result_dir = RUN_PATH)
        device = model_config["device"]
    elif MODEL_NAME == "m3":
        from forecast_backbones.m3.transformer_trainer import M3_build_training_args
        training_args = M3_build_training_args(model_config, result_dir = RUN_PATH)
        device = model_config["hyperparameter"]["device"]
    elif MODEL_NAME == "m4":
        from forecast_backbones.m4.transformer_trainer import M4_build_training_args
        training_args = M4_build_training_args(model_config, result_dir = RUN_PATH)
        device = model_config["optimization"]["device"]
    elif MODEL_NAME == "m4_pinball":
        from forecast_backbones.m4_pinball.transformer_trainer import M4Pinball_build_training_args
        training_args = M4Pinball_build_training_args(model_config, result_dir=RUN_PATH)
        device = str(training_args.device)
    elif MODEL_NAME == "m6":
        from forecast_backbones.m6.transformer_trainer import M6_build_training_args
        training_args = M6_build_training_args(model_config, result_dir = RUN_PATH)
        device = (model_config.get("timesfm3") or {}).get("device", "auto")
    elif MODEL_NAME == "m8":
        from forecast_backbones.m8.transformer_trainer import M8_build_training_args
        training_args = M8_build_training_args(model_config, result_dir = RUN_PATH)
        device = str(training_args.device)
    elif MODEL_NAME == "mW":
        from weather_fusion.mW.transformer_trainer import mW_build_training_args
        training_args = mW_build_training_args(model_config, result_dir = RUN_PATH)
        device = model_config["device"]
    elif MODEL_NAME == "mF":
        from hetero_fusion.mF.transformer_trainer import mF_build_training_args
        training_args = mF_build_training_args(model_config, result_dir = RUN_PATH)
        device = model_config["training"]["device"]
    elif MODEL_NAME == "mF_mean":
        from hetero_fusion.mF_mean.transformer_trainer import mF_build_training_args
        training_args = mF_build_training_args(model_config, result_dir = RUN_PATH)
        device = model_config["training"]["device"]


    return training_args, device

if __name__ == '__main__':
    # distributed_state = PartialState()

    # ======================= Parse command line arguments ========================
    parser = argparse.ArgumentParser(description="End-to-end training script for HFTrainer")
    #  模型以及模型配置文件，
    parser.add_argument("--model_name", type=str, default="m1", help="Model name to use for training (default: m1)")
    parser.add_argument("--model_config_path", type=str, default=None, help="Path to model config file. If not set, auto-resolved based on model_name.")

    # 模型训练配置文件
    parser.add_argument("--training_config_path", type=str, default=None, help="Path to model config file. If not set, auto-resolved based on model_name.")

    # 两种数据集读取方式，
    # 1、给定数据路径 + 特定名称数据集；
    parser.add_argument("--dataset_path", type=str, default="./dataset/YLJ_0812_90_train_10_test", help="Path to dataset directory. If not set, defaults to BASE_PATH/dataset.")
    #  2、给定三个数据文件路径，此时，三个数据文件路径必须同时提供。
    parser.add_argument("--train_path", type=str, default=None, help="Full path to the training parquet file.")
    parser.add_argument("--valid_path", type=str, default=None, help="Full path to the validation parquet file.",)
    parser.add_argument("--test_path", type=str, default=None, help="Full path to the test parquet file.")

    # 输出文件以及观测项
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run ID for this training run. If not set, auto-generated from current timestamp.")
    parser.add_argument("--run_path", type=str, default=None,
                        help="Output path for this run. If not set, defaults to BASE_PATH/runs/model_name/run_id.")
    args = parser.parse_args()

    # ======================= Resolve model config ========================
    MODEL_NAME = args.model_name

    if args.model_config_path is not None:
        MODEL_CONFIG_PATH = Path(args.model_config_path) # 优先读取输入配置
    elif MODEL_NAME == "m4_pinball":
        from forecast_backbones.m4_pinball.config import DEFAULT_CONFIG
        MODEL_CONFIG_PATH = DEFAULT_CONFIG
    else:
        # 没有读取保留默认配置
        if (Path(BASE_PATH) / "forecast_backbones" / MODEL_NAME / "config" / "configs.yaml").is_file():
            MODEL_CONFIG_PATH = (Path(BASE_PATH) / "forecast_backbones" / MODEL_NAME / "config" / "configs.yaml")
        elif (Path(BASE_PATH) / "hetero_fusion" / MODEL_NAME / "config" / "configs.yaml").is_file():
            MODEL_CONFIG_PATH = (Path(BASE_PATH) / "hetero_fusion" / MODEL_NAME / "config" / "configs.yaml")
        elif (Path(BASE_PATH) / "weather_fusion" / MODEL_NAME / "config" / "configs.yaml").is_file():
            MODEL_CONFIG_PATH = (Path(BASE_PATH) / "weather_fusion" / MODEL_NAME / "config" / "configs.yaml")
        else:
            raise FileNotFoundError( f"Model config file should not left as blank")
    # ======================= Resolve dataset directory ========================
    # dataset_path 未提供时，使用默认目录 BASE_PATH/dataset。
    if args.dataset_path is not None:
        DATASET_PATH = Path(args.dataset_path)
    else:
        DATASET_PATH = Path(BASE_PATH) / "dataset"

    # ======================= Resolve data file paths ========================
    explicit_data_paths = {
        "train_path": args.train_path,
        "valid_path": args.valid_path,
        "test_path":  args.test_path,
    }

    provided_path_names = [name for name, value in explicit_data_paths.items() if value is not None]

    if len(provided_path_names) == 3:
        # 三个完整路径全部提供，直接使用。
        train_path, valid_path, test_path = Path(args.train_path), Path(args.valid_path), Path(args.test_path)
        data_path_source = "explicit file paths"

    elif len(provided_path_names) == 0:
        # 三个文件路径未都提供，使用 dataset_path 拼接固定文件名。
        train_path = DATASET_PATH / "train.parquet"
        valid_path = DATASET_PATH / "valid.parquet"
        test_path  = DATASET_PATH / "test.parquet"
        data_path_source = "dataset directory"
    else:
        missing_path_names = [ name for name, value in explicit_data_paths.items() if value is None]

        parser.error(
            "train_path、valid_path 和 test_path 必须同时提供。"
            f"当前已提供: {provided_path_names}；"
            f"当前缺少: {missing_path_names}"
        )

    for split_name, file_path in (("train", train_path),("valid", valid_path),("test", test_path)):
        if not file_path.is_file():
            raise FileNotFoundError(f"{split_name} dataset file does not exist: {file_path}")

    # ======================= Resolve run path ========================
    if args.run_id is None:
        RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        RUN_ID = args.run_id

    if args.run_path is not None:
        RUN_PATH = Path(args.run_path)
    else:
        RUN_PATH = Path(BASE_PATH) / "runs" / MODEL_NAME / RUN_ID
    RUN_PATH.mkdir(parents=True, exist_ok=True),

    # ======================= Validate paths ========================
    print(f"[Paths] MODEL_CONFIG_PATH = {MODEL_CONFIG_PATH}")
    print(f"[Paths] DATASET_PATH      = {DATASET_PATH}")
    print(f"[Paths] TRAIN_PATH        = {train_path}")
    print(f"[Paths] VALID_PATH        = {valid_path}")
    print(f"[Paths] TEST_PATH         = {test_path}")
    print(f"[Paths] DATA_PATH_SOURCE  = {data_path_source}")
    print(f"[Paths] RESULT_PATH       = {RUN_PATH}")

    # ======================= Load all configs for HF Trainer ========================
    # model_config 设定并初始化 model
    import yaml
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    if MODEL_NAME == "m4_pinball":
        from forecast_backbones.m4_pinball.train import train
        train(model_config, str(train_path), str(valid_path), RUN_PATH, test_source=str(test_path))
        raise SystemExit(0)
    # ======================= M6：自带训练入口（校准段 + 按块早停 + 训后路由），不拆进 train_model ========================
    # 数据从 DATASET_PATH（pipelines.time_splits 落的 train/valid/cal.parquet + split_manifest.json）读，
    # 与 M1–M4 共用同一份切分；--train_path 显式给文件时取其所在目录。
    if MODEL_NAME == "m6":
        from forecast_backbones.m6.train import train_m6
        splits_dir = DATASET_PATH if data_path_source == "dataset directory" else train_path.parent
        print(f"[m6] train_m6(config={MODEL_CONFIG_PATH}, splits_dir={splits_dir}, out_dir={RUN_PATH})")
        train_m6(MODEL_CONFIG_PATH, splits_dir=splits_dir, out_dir=RUN_PATH)
        raise SystemExit(0)
    # ======================= M8：同 M6，自带训练入口（验证集早停 + 存最优 + 报表），不拆进 train_model ========================
    if MODEL_NAME == "m8":
        from forecast_backbones.m8.train import train_m8
        splits_dir = DATASET_PATH if data_path_source == "dataset directory" else train_path.parent
        print(f"[m8] train_m8(config={MODEL_CONFIG_PATH}, splits_dir={splits_dir}, out_dir={RUN_PATH})")
        train_m8(MODEL_CONFIG_PATH, splits_dir=splits_dir, out_dir=RUN_PATH)
        raise SystemExit(0)

    # Dedicated model entry points above do not depend on the legacy trainer utilities.
    from pipelines.transformer_trainer import train_model
    model = build_model(model_name = MODEL_NAME, model_config = model_config)

    # ======================= Build the training args and Training ========================
    training_args, device = build_train_args(MODEL_NAME, model_config)

    # ── 显式设置全局随机种子，确保每次训练结果可复现 ──
    # set_training_seed(training_args.seed, full_determinism=True)

    # 自动降级：无 GPU 时强制使用 CPU
    requested_device = str(training_args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[Auto-Downgrade] device={requested_device}, but CUDA not available → forcing CPU")
        training_args.device = "cpu"

    # ======================= Load datasets ========================
    # 列名从模型配置的 dataset 段读（feature_cols / label_cols / entity_cols / ts_col），没写的沿用历史默认；
    # 数据列的真源是数据本身，不同数据版本（chronos vs observe_power_chronos_pred 等）改配置即可，不改脚本。
    dataset_cfg = model_config.get("dataset") or {}
    ts_col       = dataset_cfg.get("ts_col", "timestamp_win")
    entity_cols  = dataset_cfg.get("entity_cols", [ts_col, "station"])
    feature_cols = dataset_cfg.get("feature_cols", ['general_ghi','general_ghi_predict','general_temp','general_temp_predict',
                                                    'observe_power','observe_power_chronos_pred'])
    label_cols   = dataset_cfg.get("label_cols", [dataset_cfg.get("label_col", "observe_power_future")])
    print(f"[Dataset] ts_col={ts_col} entity_cols={entity_cols} feature_cols={feature_cols} label_cols={label_cols}")
    df_agg_train = pd.read_parquet(train_path)
    df_agg_val   = pd.read_parquet(valid_path)
    train_dataset = UnifiedDataset(
        df_agg_train,
        timestamp_cols=[ts_col],
        entity_cols=list(entity_cols),
        feature_cols=list(feature_cols),
        label_cols=list(label_cols),
        for_hf_trainer=True,
    )
    valid_dataset = UnifiedDataset(
        df_agg_val,
        timestamp_cols=[ts_col],
        entity_cols=list(entity_cols),
        feature_cols=list(feature_cols),
        label_cols=list(label_cols),
        for_hf_trainer=True,
    )

    print(f"[Dataset] train = {len(train_dataset)}")
    print(f"[Dataset] valid = {len(valid_dataset)}")

    # collate_fn 显式定义
    from functools import partial
    custom_collate_fn = partial(
        UnifiedDataset.collate_fn,
        timestamp_output='epoch_s',
        drop_str_type=True,
        for_hf_trainer=True,
    )

    trainer = train_model(training_config=training_args,
                        model = model,
                        train_dataset = train_dataset,
                        eval_dataset = valid_dataset,
                        collate_fn = custom_collate_fn)

    model_save_dir = Path(RUN_PATH) / "model_result"
    trainer.save_model(model_save_dir)
    # 等待所有进程完成 save_model，避免与下面的配置写入竞争
    trainer.accelerator.wait_for_everyone()
    # 只在主进程保存模型配置，避免多进程重复写入同一文件
    if trainer.accelerator.is_main_process:
        with open(model_save_dir / "model_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(trainer.model.config, f, allow_unicode=True, sort_keys=False)


    # ======================= Build and train on every process ========================
    # CHECKPOINT_DIR = RUN_PATH / "model_save"
    # trainer.accelerator.wait_for_everyone()
    # run_main_process_step(
    #         distributed_state,
    #         "save model checkpoint",
    #         lambda: save_model_checkpoint(trainer.model, checkpoint_dir=str(CHECKPOINT_DIR), model_name=MODEL_NAME, model_config=config, accelerator=trainer.accelerator,),
    #     )

    # ======================= Load Model for the evaluation (no need for the end to end training script) ========================

    # CHECKPOINT_PATH = RUN_PATH / "model_save"
    # model, config = load_model_checkpoint(
    #     checkpoint_dir=str(CHECKPOINT_PATH),
    #     model_builder=lambda model_config: build_model(MODEL_NAME, model_config),
    # )

    # if device == "cuda" and torch.cuda.is_available():
    #     model.to(device)
    # else:
    #     model.to("cpu")
    # print("load the model with checkpoints.")

    # ======================= Evaluation (post-training) =======================
    # test_dataset = UnifiedDataset(pd.read_parquet(test_path))
    # print_main(distributed_state,f"[Dataset] test  = {len(test_dataset)}")

    # print_main(distributed_state, "\n" + "=" * 60)
    # print_main(distributed_state, f"Starting evaluation: model_version = {MODEL_NAME}/{RUN_ID}")
    # print_main(distributed_state, "=" * 60)

    # CAPACITY = 465
    # result = test_model(trainer= trainer,
    #                     test_dataset=test_dataset,
    #                     MODEL_NAME = MODEL_NAME,
    #                     RUN_ID = RUN_ID,
    #                     RUN_PATH = RUN_PATH,
    #                     CAPACITY = CAPACITY,
    #                     distributed_state = distributed_state)
    # trainer.accelerator.wait_for_everyone()

    # if distributed_state.is_main_process and result is not None:
    #     print(f"[Evaluation] result = {result}", flush=True)

    # 这里必须要加入在所有脚本的最后。
    # ── 退出训练，清除accelerate 子进程 ──
    # 优雅退出，减少内存消耗
    # 优先走 Trainer 内部的 Accelerator
    trainer.accelerator.wait_for_everyone()
    if trainer is not None and hasattr(trainer, "accelerator"):
        trainer.accelerator.end_training()
    # else:
    # 兜底：纯 PyTorch 分布式清理
    import torch
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
