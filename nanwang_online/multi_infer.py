import copy
import datetime
import logging
import os
import sys
import time
from typing import Optional
import argparse
import yaml
import warnings
import duckdb
import pandas as pd
import numpy as np 
from pathlib import Path

algo_home_path = Path("/home/nfdw/nuoya")  # 算法的home目录
# algo_home_path = Path("/home/ma-user/work/test_ljl/nuoya_xmj_0928/nanwang_offline_test")
jtdata_home_path = Path("/jtdata/products")   # jtdata的home目录
# jtdata_home_path = Path("/home/ma-user/work/test_ljl/nuoya_xmj_0928/nanwang_offline_test/jtdata/products")   # jtdata的home目录

sys.path.append(str(algo_home_path))
sys.path.append(str(algo_home_path / "algo/nuoya_algo"))
sys.path.append(str(algo_home_path / "algo/nuoya_algo/ensemble_model"))
sys.path.append(str(algo_home_path / "algo/nuoya_algo/individual_models_deploy") )
sys.path.append( str(algo_home_path / "algo/nuoya_algo/individual_models_deploy/ultra_province"))

from ods_transform.ds_power import get_ds_observe_power, fill_power_partial_miss
from weather_aggregation_rules_v2_1 import RULES
from feature_aggregation import aggregate_features
from etl_fix_forecast2 import fix_observe_power_predict
from etl_fix_forecast_v2 import fix_observe_power_predict_linear

from provinced1.reg_model_v1.infer.reg_model_infer import RegInfer_reg_modelV2D1
from ultra_province import Model as UltraProvinceModel 

from safetensors.torch import load_file

from forecast_backbones.m1.transformer_models import TransformersAdaptedM1Model
from forecast_backbones.m5.transformer_models import TransformersAdaptedTimesFM3Model
from pipelines.pipeline_wrapper_inference import unified_inference
from pipelines.rolling_wrapper_inference import rolling_inference


from detect_data_abnormal import check_ds_format_abnormal
from ods_transform.dataset_inferface import DataSetInterface
from inference import multi_station_inference, ds_to_ods, MultiStationInference
from ods_transform.trace_store import save_ds_table, load_ds_table

warnings.filterwarnings("ignore")
multi_logger = logging.getLogger('PredictTool.model1')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

INFO_PATH = str(jtdata_home_path / "data/info.csv")
CON = duckdb.connect(database=":memory:", config={"threads":"12", "max_memory":"50GB"})

class MultiInfer:
    def __init__(
            self,
            report_time: str,
            data_path: dict = None,
            predict_type: str = 'short',
            checkpoints_dir: str = "",
            config: dict | str = None,
            train_data_saved_dir: str = "",
            infer_data_save_dir: str = "",
            save_path: str = "",
            timeout: Optional[int] = None,
            log_path: str = "",
    ):
        # 日志
        log_path = str(algo_home_path / "logs/")
        log_filename = f"multi_infer_{predict_type}_{report_time.replace(' ', '_').replace(':', '')}.log"
        # checkpoint
        self.checkpoints_dir = str(algo_home_path / "algo/checkpoints/checkpoints_moev3_1_100m_4dr2")  # 站级的 ckpt
        self.province_checkpoints_dir = str(algo_home_path / "algo/checkpoints/ckpt_province")  # 省级的 ckpt
        # ods和ds保存路径
        self.train_data_saved_dir = str(algo_home_path / "data/train")
        self.infer_data_save_dir = str(algo_home_path / "data/infer")
        self.infer_data_save_dir_no_cache = str(algo_home_path / "data/infer_no_cache" )
        # 结果保存
        save_path = str(jtdata_home_path / "data/result/nuoya")
        # 输入气象源
        self.data_path = {
            "power": "",
            "observe_weather": "",
            "SOLARGIS": "",
            "solar_southern": "",
            "GDFS_JTHFGNED_MID_JTNWQISX": "",
            "EC_JTKRD7RO": "",
        }
        # 需要预测的区域
        self.areas = ["province_guangxi_solar"]
        # nuoya配置文件
        self.config = str(algo_home_path / "algo/nuoya_algo/ensemble_model/chunk_config.yaml")

        self.report_time = report_time
        self.logger = multi_logger
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, log_filename)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.info(f"日志文件：{log_file}")

        self.save_path = os.path.join(save_path, predict_type)

        # 历史/预报长度
        self.history_len = 672
        self.predict_type = predict_type
        if self.predict_type == 'short':
            self.predict_seq_lens = 480
            self.logger.info(f"设置预测长度：{self.predict_seq_lens}")

        elif self.predict_type == 'ultra_short':
            self.predict_seq_lens = 192
            self.logger.info(f"设置预测长度：{self.predict_seq_lens}")
        else:
            self.predict_seq_lens = 480

        self.logger.info(f"预测类型：{self.predict_type}， 设置预测长度：{self.predict_seq_lens}，"
                         f"使用气象源：{self.data_path.keys().__str__()}")

        # fix: 生成 start_time, end_time 的代码封装为函数
        self._refresh_time_range()

        with open(self.config, 'r', encoding="utf-8") as f:
            self.config_dict = yaml.safe_load(f)

        self.only_power = False

        ## 09.23新增：初始化 MultiStationInference 对象来替代 multi_station_inference() 的旧的推理方式
        # 站级模型对象
        config_station = copy.deepcopy(self.config_dict)
        config_station["data_check"]["future_len"] = self.predict_seq_lens
        self.station_inference_obj = MultiStationInference(config=config_station, checkpoints_dir=self.checkpoints_dir)
        # 省级模型对象
        config_province = copy.deepcopy(self.config_dict)
        config_province["data_check"]["future_len"] = self.predict_seq_lens
        self.prov_inference_obj = MultiStationInference(config=config_province, checkpoints_dir=self.province_checkpoints_dir)
        # tsfm模型对象
        config_tsfm = copy.deepcopy(self.config_dict)
        config_tsfm["backstop_strategy"]["tsfm_only"] = True
        config_tsfm["pre_trained_model"]["future_len"] = self.predict_seq_lens
        config_tsfm["data_check"]["future_len"] = self.predict_seq_lens
        self.tsfm_inference_obj = MultiStationInference(config=config_tsfm, checkpoints_dir=self.checkpoints_dir)

        ###### DEVELOPE INSERT: 新pipeline的模型初始化位置  #############
        if predict_type == 'short':
            # forecast backbone: 省级短期模型
            short_province_config = str(algo_home_path / "algo/nuoya_algo/individual_models_deploy/provinced1/config_0917_0916_0615_0825_150.yaml" )
            self.prov_short_config = yaml.safe_load(open(short_province_config, 'r', encoding="utf-8"))
            self.short_province_model = RegInfer_reg_modelV2D1(self.prov_short_config)

            # M1_0821_release: 站级短期模型
            m1_0821_release_config = str(algo_home_path / "algo/checkpoints/m1_0903release_epoch1/model_config.yaml")
            m1_0821_ckpt_path = str(algo_home_path / "algo/checkpoints/m1_0903release_epoch1/model.safetensors")
            self.station_short_config_m1 = yaml.safe_load(open(m1_0821_release_config, 'r', encoding="utf-8"))
            self.station_short_model_m1 = TransformersAdaptedM1Model(self.station_short_config_m1)
            self.station_short_model_m1.load_state_dict(load_file(m1_0821_ckpt_path), strict=False)

            # M5 的省、站级的短期模型
            m5_release_config = str(algo_home_path / "algo/nuoya_algo/forecast_backbones/m5/model_config.yaml")
            self.short_config_m5 = yaml.safe_load(open(m5_release_config, 'r', encoding="utf-8"))
            self.short_model_m5 = TransformersAdaptedTimesFM3Model(self.short_config_m5)

        elif predict_type == 'ultra_short':
            # M1_0821_release: 站级超短期模型
            m1_0821_release_config = str(algo_home_path / "algo/checkpoints/m1_0903release_epoch1/model_config.yaml")
            m1_0821_ckpt_path = str(algo_home_path / "algo/checkpoints/m1_0903release_epoch1/model.safetensors")
            self.station_ultra_short_config_m1 = yaml.safe_load(open(m1_0821_release_config, 'r', encoding="utf-8"))
            self.station_ultra_short_model_m1 = TransformersAdaptedM1Model(self.station_ultra_short_config_m1)
            self.station_ultra_short_model_m1.load_state_dict(load_file(m1_0821_ckpt_path), strict=False)

            # forecast backbone: 省级超短期模型
            ultra_prov_config = str(algo_home_path / "algo/nuoya_algo/individual_models_deploy/ultra_province/config.yaml")
            checkpoint = str(algo_home_path / "algo/checkpoints/ckpt_v1")
            self.ultra_province_model = UltraProvinceModel(ultra_prov_config, checkpoint)

            # M5 省、站级的超短期模型
            m5_release_config = str(algo_home_path / "algo/nuoya_algo/forecast_backbones/m5/model_config.yaml")
            self.ultra_short_config_m5 = yaml.safe_load(open(m5_release_config, 'r', encoding="utf-8"))
            self.ultra_short_model_m5 = TransformersAdaptedTimesFM3Model(self.ultra_short_config_m5)

        ################################################################

    def _refresh_time_range(self):
        """刷新时间"""
        # 起报时间
        self.forecast_time = datetime.datetime.strptime(self.report_time, "%Y-%m-%d %H:%M:%S")
        # 起报时间下一个时间点上开始预测的第一个点
        self.forecast_s_time = self.forecast_time + datetime.timedelta(minutes=15)

        # 构造预测结果保持点时间段
        if self.predict_type == 'short':
            self.result_s_time = (self.forecast_time + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0)
            self.result_e_time = (self.forecast_time + datetime.timedelta(days=4)).replace(hour=23, minute=45, second=0)
        else:
            self.result_s_time = self.forecast_s_time
            self.result_e_time = self.forecast_s_time + datetime.timedelta(hours=5) - datetime.timedelta(minutes=15)
        

    def _clean_cache(self):
        """
        因为ods cache 缓存存在历史数据污染，每次调用前后删除缓存；
        如果未来有性能问题，则不用清理缓存
        """
        for a in self.areas:
            for suffix in ("short", "ultra_short"):
                path = os.path.join(self.infer_data_save_dir_no_cache, "ods", f"{a}_{suffix}_ods_infer_cache.parquet")  # 清理指定 no_cache 缓存路径
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        self.logger.info(f"[{suffix}]缓存已删除：{path}")
                    else:
                        self.logger.info(f"[{suffix}]缓存路径不存在：{path}")
                except Exception as e:
                    self.logger.warning(f"[{suffix}]删除缓存失败：{path}, err_msg: {e}")

    def get_feature_store(self, cur_report_time=""):
        if cur_report_time == "":
            cur_report_time = self.report_time
        start_t = time.time()
        self.logger.info(
            f"multi_infer: start to get feature..., params: "
            f"cur_report_time: {cur_report_time}, "
            f"data_path: {self.data_path}, "
            f"save_path: {self.save_path}, "
            f"predict_type: {self.predict_type}, "
            f"predict_seq_lens: {self.predict_seq_lens}"
            f"checkpoints_dir: {self.checkpoints_dir}, "
            f"config: {self.config},"
            f"only_power: {self.only_power}"
        )

        stations_df = pd.read_csv(INFO_PATH)
        stations_df = stations_df[["plant_pointname", "LONGITUDE", "LATITUDE", "GCCAPACITY", "province_pointname"]]
        stations_df.rename(
            columns={'plant_pointname': 'plantId', 'LONGITUDE': 'lng', 'LATITUDE': 'lat', 'GCCAPACITY': 'cap_power_on'},
            inplace=True)

        error_area = []  # 统计预测异常的区域
        for area in self.areas:
            # 区分solar和wind
            area_type = "_solar" if area.endswith("_solar") else "_wind"
            province = area.replace(area_type, "")
            station_name_suffix = "plant_guangfu" if area.endswith("_solar") else "plant_fengdian"
            station_df = stations_df[stations_df["province_pointname"] == province]
            station_df = station_df[station_df["plantId"].str.startswith(station_name_suffix)]
            area_df = station_df[["plantId", "lng", "lat", "cap_power_on"]].copy(deep=True)

            self.logger.info(f"=======start predict area: {area}, area_df shape: {area_df.shape}=======")
            df, success_flag = DataSetInterface(
                save_history_data_dir=self.train_data_saved_dir,
                save_infer_data_dir=self.infer_data_save_dir,
                logger=self.logger,
            ).prepare_infer_data(
                forecast_time=cur_report_time,
                predict_type=self.predict_type,
                data_path=self.data_path,
                area_name=area,
                plants_df=area_df,
                predict_seq_lens=self.predict_seq_lens,
                history_len=self.history_len,
                only_power=self.only_power,
            )
            # 插入 ETL 数据保存，如果后续有新表再这下面平铺添加，比如TSFM推理会产生的新列形成的表
            save_ds_table(
                df,
                base_path=self.train_data_saved_dir,
                table_name='ds_trajectory_input',  # 表名
                area=area,
                predict_type=self.predict_type,
                report_time=cur_report_time,
                dtype="float16",
                logger=self.logger
            )

    def handle(self):
        start_t = time.time()
        self._refresh_time_range()  # 09.23 fix 时间范围导致的裁剪bug
        self.logger.info(
            f"multi_infer: start to handle..., params: "
            f"report_time: {self.report_time}, "
            f"data_path: {self.data_path}, "
            f"save_path: {self.save_path}, "
            f"predict_type: {self.predict_type}, "
            f"predict_seq_lens: {self.predict_seq_lens}"
            f"checkpoints_dir: {self.checkpoints_dir}, "
            f"config: {self.config},"
            f"only_power: {self.only_power}"
        )

        stations_df = pd.read_csv(INFO_PATH)
        stations_df = stations_df[["plant_pointname", "LONGITUDE", "LATITUDE", "GCCAPACITY", "province_pointname"]]
        stations_df.rename(
            columns={'plant_pointname': 'plantId', 'LONGITUDE': 'lng', 'LATITUDE': 'lat', 'GCCAPACITY': 'cap_power_on'},
            inplace=True)

        error_area = []  # 统计预测异常的区域
        for area in self.areas:
            # 区分solar和wind
            area_type = "_solar" if area.endswith("_solar") else "_wind"
            province = area.replace(area_type, "")
            station_name_suffix = "plant_guangfu" if area.endswith("_solar") else "plant_fengdian"
            station_df = stations_df[stations_df["province_pointname"] == province]
            station_df = station_df[station_df["plantId"].str.startswith(station_name_suffix)]
            area_df = station_df[["plantId", "lng", "lat", "cap_power_on"]].copy(deep=True)

            self.logger.info(f"=======start predict area: {area}, area_df shape: {area_df.shape}=======")
            df, success_flag = DataSetInterface(
                save_history_data_dir=self.train_data_saved_dir,
                save_infer_data_dir=self.infer_data_save_dir,
                logger=self.logger,
            ).prepare_infer_data(
                forecast_time=self.report_time,
                predict_type=self.predict_type,
                data_path=self.data_path,
                area_name=area,
                plants_df=area_df,
                predict_seq_lens=self.predict_seq_lens,
                history_len=self.history_len,
                only_power=self.only_power,
            )
            # debug: 修复 observe_power 字段使用nuoya治理方案
            ds_observe_power = get_ds_observe_power(
                    con=CON,
                    ods_table=str(jtdata_home_path / 'govern_data/power'),
                    area="guangxi",
                    report_time=self.report_time,
                    logger=self.logger,
                )
            df['timestamp_win'] = pd.to_datetime(df['timestamp_win'])
            ds_observe_power = df[['timestamp_win', 'station']].merge(ds_observe_power, on=['timestamp_win', 'station'], how='outer')  # 确保 entity 无缺失
            ds_observe_power['observe_power'] = fill_power_partial_miss(ds_observe_power['observe_power'], pos_idx=666)  # 确保算法repair的执行环境

            # 进行算法的填补
            if self.predict_type == 'short':
                m5_model = self.short_model_m5
            elif self.predict_type == 'ultra_short':
                m5_model = self.ultra_short_model_m5
            ds_observe_power_repair = fix_observe_power_tail_miss(
                df_observe_power=ds_observe_power[['timestamp_win','station','observe_power']], 
                pipeline_backbone=m5_model, 
                fix_len=600
                )

            ds_observe_power_repair['timestamp_win'] = pd.to_datetime(ds_observe_power_repair['timestamp_win'])
            df['timestamp_win'] = pd.to_datetime(df['timestamp_win'])
            df_repaired = ds_observe_power_repair[['timestamp_win','station','observe_power_repaired']].merge(df, on=['timestamp_win','station'], how='left')
            
            ############ end of debug #############
            # 插入 ETL 数据保存，如果后续有新表再这下面平铺添加，比如TSFM推理会产生的新列形成的表
            save_ds_table(
                df_repaired,
                base_path=self.train_data_saved_dir,
                table_name='ds_trajectory_input',  # 表名
                area=area,
                predict_type=self.predict_type,
                report_time=self.report_time,
                dtype="float16",
                logger=self.logger
            )
            # =========== 无 ods cache 保存 ds_trajectory_input_no_ods_cache  =============
            self.logger.info(f"开始清理ods cache，并保存 no cache ds_trajectory, ods cache路径：{self.infer_data_save_dir_no_cache}")
            self._clean_cache()  # 先清理ods cache
            df_no_ods_cache, _ = DataSetInterface(
                save_history_data_dir=self.train_data_saved_dir,
                save_infer_data_dir=self.infer_data_save_dir_no_cache,
                logger=self.logger,
            ).prepare_infer_data(
                forecast_time=self.report_time,
                predict_type=self.predict_type,
                data_path=self.data_path,
                area_name=area,
                plants_df=area_df,
                predict_seq_lens=self.predict_seq_lens,
                history_len=self.history_len,
                only_power=self.only_power,
            )
            # 插入 ETL 数据保存，如果后续有新表再这下面平铺添加，比如TSFM推理会产生的新列形成的表
            save_ds_table(
                df_no_ods_cache,
                base_path=self.train_data_saved_dir,
                table_name='ds_trajectory_input_no_ods_cache',  # 表名
                area=area,
                predict_type=self.predict_type,
                report_time=self.report_time,
                dtype="float16",
                logger=self.logger
            )

            # 读入轨迹数据, bugfix: 使用原始的 df 后续代码会报错
            import glob
            rtime = datetime.datetime.strptime(self.report_time, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M")
            tmp_file_pattern = self.train_data_saved_dir + f"/ds_trajectory_input/area={area}/predict_type={self.predict_type}/date={self.report_time.split()[0]}/{rtime}_*.parquet"
            tmp_parquet_file = sorted(glob.glob(tmp_file_pattern))[-1] # TODO: 如果有修改上游数据，回测使用最新数据
            df = pd.read_parquet(tmp_parquet_file).drop(columns=['update_time']) 
            df['observe_power'] = df['observe_power_repaired']  # bugfix-09.30: 使用算法修补的 observe_power_repaired 列进行替换
            success_flag = True 
            self.logger.info(f"[FIX] observe_power修正成功： shape: {df.shape} ")
            # ===============================================

            # 从 cap_power_on 生成 cap_power_on_val 列，作为单个元素
            df['cap_power_on_val'] = df['cap_power_on'].apply(lambda x:x[-1] if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0 else 1) # 使用1 来兜底

            if not success_flag:
                self.logger.info(
                    f"multi_infer: finish generate infer area failed, area name: {area}, cost time: {time.time() - start_t}")
                error_area.append(area)
                continue

            self.logger.info(
                f"multi_infer: finish generate infer data..., shape: {df.shape}, cost time: {time.time() - start_t}")

            self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RULES')
            self.logger.info(RULES)
            df = aggregate_features(df, RULES)
            # bugfix: 09.29 兼容性提升的方案，使用 general_GHI, general_TEMP, general_GHI_predict, general_TEMP_predict 来赋值给 MoE V3.1
            for i in range(9):
                df[f'ssrd_pos_{i + 1}'] = df['general_GHI']
                df[f't2m_pos_{i + 1}'] = df['general_TEMP']
                df[f'ssrd_pos_{i + 1}_predict'] = df['general_GHI_predict']
                df[f't2m_pos_{i + 1}_predict'] = df['general_TEMP_predict']

            df['GHI_SOLARGIS'] = df['general_GHI']
            df['TEMP_SOLARGIS'] = df['general_TEMP']
            df['GHI_SOLARGIS_predict'] = df['general_GHI_predict']
            df['TEMP_SOLARGIS_predict'] = df['general_TEMP_predict']
            self.logger.info(df.iloc[0])

            # 先检查数据，然后把站点分2类，异常类走tsfm，正常类走MoE V3.1
            if self.only_power:
                df_abnormal_part = df.copy(deep=True)
                df_normal_part = pd.DataFrame()
                self.logger.info(f"multi_infer: only_power模式，跳过数据检查，全部数据送入tsfm: {df_abnormal_part.shape}")
            else:
                self.logger.info(f"multi_infer: 数据检查开始： {time.time() - start_t} s")
                self.logger.info(f"multi_infer: 数据检查开始 shape: {df.shape}")

                # 传入检查长度, 只检查 RULES 配置里我们输出的列，降低敏感性
                # bugfix: 09.29 兼容性提升的方案，使用 general_GHI, general_TEMP, general_GHI_predict, general_TEMP_predict 来进行异常的检测，放松检测防止单个气象源的缺失让模型走了兜底 
                col_his, col_pred = ['general_GHI', 'general_TEMP'], ['general_GHI_predict', 'general_TEMP_predict'] 
                df_result = check_ds_format_abnormal(df[['timestamp_win', 'station'] + col_his + col_pred], 
                                                     pred_col_len=self.predict_seq_lens)
                self.logger.info(f"multi_infer: 数据检查结束： {time.time() - start_t} s")

                # 异常数据
                df_abnormal = df_result[df_result['is_abnormal'] == True]
                df_abnormal_part = df.merge(df_abnormal, on=['timestamp_win', 'station'], how='inner').drop(
                    columns=['is_abnormal'])
                self.logger.info(
                    f"发现异常数据 shape: {df_abnormal_part.shape} ， 站点：{df_abnormal_part['station'].tolist()}")

                # 正常数据
                df_normal = df_result[df_result['is_abnormal'] == False]
                df_normal_part = df.merge(df_normal, on=['timestamp_win', 'station'], how='inner').drop(
                    columns=['is_abnormal'])
                self.logger.info(f"正常数据 shape: {df_normal_part.shape} ")

            # 提取 province 行（station == area），只存在于 normal 中
            df_province = pd.DataFrame()
            if not df_normal_part.empty and area in df_normal_part['station'].values:
                df_province = df_normal_part[df_normal_part['station'] == area].copy(deep=True)
                df_normal_part.drop(df_normal_part[df_normal_part['station'] == area].index, inplace=True)
                self.logger.info(f"提取 province 行: {area}, shape={df_province.shape}")
            else:
                self.logger.warning(f"province 行未找到: {area}")

            df_res_list = []
            # 站级正常数据-MoE v3.1 入口
            if df_normal_part.shape[0] > 0 and self.only_power is False:
                self.logger.info(f"站级推理开始，checkpoint: {self.checkpoints_dir}")
                res_tmp_df_normal = self.station_inference_obj.inference(ds_dataframe=df_normal_part)  # 09.23 改造附加
                res_tmp_df_normal['timestamp_win'] = pd.to_datetime(res_tmp_df_normal['timestamp_win']) # 09.30 debug: 缺少行的问题
                result_df_normal_part = df_normal_part[['timestamp_win','station']].merge(res_tmp_df_normal, on=['timestamp_win', 'station'], how='left') # 09.30 debug: 缺少行的问题
                result_df_normal_part['timestamp_win'] = pd.to_datetime(result_df_normal_part['timestamp_win']) # 09.30 debug: 缺少行的问题
                df_res_list.append(result_df_normal_part)
                self.logger.info(f"======站级推理完成======")

                ############ DEVELOPE INSERT: 站级预测 forecast backbone 的插入 ########### 
                # 26.09.03 暂时对齐开发态需要的列名称 
                df_normal_part_station = df_normal_part.copy(deep=True)
                df_normal_part_station['timestamp_win'] = pd.to_datetime(df_normal_part_station['timestamp_win'])

                # # * 使用旧版代码进行TSFM推理
                tsfm_res_station = self.tsfm_inference_obj.inference(ds_dataframe=df_normal_part_station)  # 09.29 改造更新
                # * end of TSFM 推理

                tsfm_res_station['timestamp_win'] = pd.to_datetime(tsfm_res_station['timestamp_win'])
                df_station_input = df_normal_part_station.merge(tsfm_res_station, on=['timestamp_win', 'station'], how='left')

                if self.predict_type == "short": # 站级短期推理分支

                    try:
                        rolling_config = {
                            'past_covariate':['general_GHI', 'general_TEMP'],
                            'future_covariate':['general_GHI_predict', 'general_TEMP_predict'],
                            'past_target': ['observe_power'],
                            'future_target': ['tsfm_predict'],
                            'future_len': 480,
                        }
                        ds_out_station_short_m1 = rolling_inference(
                            self.station_short_model_m1,
                            df_station_input,
                            config=rolling_config,
                            timestamp_cols=("timestamp_win",),
                            entity_cols=("timestamp_win", "station"),
                            feature_cols=[
                                    "general_GHI",
                                    "general_GHI_predict",
                                    "general_TEMP",
                                    "general_TEMP_predict",
                                    "observe_power",
                                    "tsfm_predict",
                                ],
                        )
                        self.logger.info(f"======M1 0903 D+1 站级推理成功 ======")
                    except Exception as e:
                        # 使用 MoE V3.1 的预测结果兜底
                        ds_out_station_short_m1 = result_df_normal_part[['timestamp_win', 'station', 'ensemble']].rename(columns={'ensemble': 'TransformersAdaptedM1Model'})
                        self.logger.info(f"======M1 0903 D+1 站级推理失败,使用 MoE V3.1兜底======")



                    try:
                        rolling_config = {
                            'past_covariate':['general_GHI', 'general_TEMP'],
                            'future_covariate':['general_GHI_predict', 'general_TEMP_predict'],
                            'past_target': ['observe_power'],
                            'future_target': ['tsfm_predict'],
                            'future_len': 480,
                        }
                        ds_out_station_short_m5 = rolling_inference(
                            self.short_model_m5,
                            df_station_input,
                            config=rolling_config,
                            timestamp_cols=("timestamp_win",),
                            entity_cols=("timestamp_win", "station"),
                            feature_cols=[
                                "general_GHI",
                                "general_GHI_predict",
                                "general_TEMP",
                                "general_TEMP_predict",
                                "observe_power",
                            ],
                        )
                    except Exception as e:
                            # 使用 MoE V3.1 的预测结果兜底
                        ds_out_station_short_m5 = result_df_normal_part[['timestamp_win', 'station', 'ensemble']].rename(columns={'ensemble': 'TransformersAdaptedTimesFM3Model'})
                        self.logger.info(f"======M5 0910 D+1 站级推理失败,使用 MoE V3.1兜底======, err_msg: {e}")

                    self.logger.info("新 pipeline M5 短期推理完成")
                    
                elif self.predict_type == "ultra_short": # 站级超短期推理分支

                    try:
                        ds_out_station_ultra_short_m1 = unified_inference(
                            self.station_ultra_short_model_m1,
                            df_station_input,
                            timestamp_cols=("timestamp_win",),
                            entity_cols=("timestamp_win", "station"),
                            feature_cols=[
                                "general_GHI",
                                "general_GHI_predict",
                                "general_TEMP",
                                "general_TEMP_predict",
                                "observe_power",
                                "tsfm_predict"
                            ],
                        )
                        self.logger.info(f"====== new pipeline M1 ultra short 站级推理完成======")
                    except Exception as e:
                        # 使用 MoE V3.1 的预测结果兜底
                        ds_out_station_ultra_short_m1 = result_df_normal_part[['timestamp_win', 'station', 'ensemble']].rename(columns={'ensemble': 'TransformersAdaptedM1Model'})
                        self.logger.info(f"======M5 0915 ultra short 站级推理失败,使用 MoE V3.1兜底======, err_msg: {e}")

                    try:
                        ds_out_station_ultra_short_m5 = unified_inference(
                            self.ultra_short_model_m5,
                            df_station_input,
                            timestamp_cols=("timestamp_win",),
                            entity_cols=("timestamp_win", "station"),
                            feature_cols=[
                                "general_GHI",
                                "general_GHI_predict",
                                "general_TEMP",
                                "general_TEMP_predict",
                                "observe_power",
                            ],
                        )
                        self.logger.info(f"======M5 0915 ultra short 站级推理完成======")
                    except Exception as e:
                        # 使用 MoE V3.1 的预测结果兜底
                        ds_out_station_ultra_short_m5 = result_df_normal_part[['timestamp_win', 'station', 'ensemble']].rename(columns={'ensemble': 'TransformersAdaptedTimesFM3Model'})
                        self.logger.info(f"======M5 0915 ultra short 站级推理失败,使用 MoE V3.1兜底======, err_msg: {e}")
                ###########################################################################

            # 省级正常数据-MoE v3.1 入口
            if df_province.shape[0] == 1:
                df_province = pd.concat([df_province] * 2, ignore_index=True)
                df_province.loc[1, "station"] = "unknow_province"
            if df_province.shape[0] > 0:
                self.logger.info(f"省级推理开始, checkpoint: {self.province_checkpoints_dir}")
                result_df_province = self.prov_inference_obj.inference(ds_dataframe=df_province)  # 09.23 改造更新
                df_res_list.append(result_df_province)
                self.logger.info(f"======省级推理完成======")

                ############ DEVELOPE INSERT: 省级预测 forecast backbone 的插入 ########### 
                if self.predict_type == "short": # 省级短期推理分支

                    try:
                        df_in_prov_short = df.merge(df_normal, on=['timestamp_win', 'station'], how='inner').drop(
                                                            columns=['is_abnormal']) 
                        ds_res_short_province = self.short_province_model.infer_nanwangd1(
                                df_in_prov_short,
                                station_col=self.prov_short_config["data_config"]["station"],
                                pred_name=self.prov_short_config["data_config"]["pred_name"],
                                timestamp_col=self.prov_short_config["data_config"]["timestamp"],
                                area="province_guangxi_solar",
                                station_name_suffix="plant_guangfu",
                                to_parquet=True
                                )
                        ds_res_short_province['yili_short_prov_D1_pred'] = \
                            ds_res_short_province[self.prov_short_config["data_config"]["pred_name"]].apply(
                                lambda x: list(np.append(x, [np.nan]*(self.predict_seq_lens-len(x)) ))
                                )
                        ds_res_short_province['timestamp_win'] = pd.to_datetime(ds_res_short_province['timestamp_win'])
                        self.logger.info(f"======YiLi 省级D+1推理完成======")
                    except:
                        # 使用 MoE V3.1 的预测结果兜底
                        ds_res_short_province = result_df_province[['timestamp_win', 'station', 'ensemble']].rename(columns={'ensemble': 'yili_short_prov_D1_pred'})
                        self.logger.info(f"======YiLi 省级D+1推理失败,使用 MoE V3.1兜底======")

                    try:
                        df_in_prov_short_m5 = df_province.merge(result_df_province, on=['timestamp_win', 'station'], how='left')  # 省级的DS层推理数据
                        rolling_config = {
                            'past_covariate':['general_GHI', 'general_TEMP'],
                            'future_covariate':['general_GHI_predict', 'general_TEMP_predict'],
                            'past_target': ['observe_power'],
                            'future_target': ['ensemble'],
                            'future_len': 480,
                        }
                        ds_res_short_province_m5 = rolling_inference(
                            self.short_model_m5,
                            df_in_prov_short_m5,
                            config=rolling_config,
                            timestamp_cols=("timestamp_win",),
                            entity_cols=("timestamp_win", "station"),
                            feature_cols=[
                                "general_GHI",
                                "general_GHI_predict",
                                "general_TEMP",
                                "general_TEMP_predict",
                                "observe_power",
                            ],
                        )
                        self.logger.info(f"======M5 省级D+1推理完成======")
                    except Exception as e:
                        # 使用 MoE V3.1 的预测结果兜底
                        ds_res_short_province_m5 = result_df_province[['timestamp_win', 'station', 'ensemble']].rename(columns={'ensemble': 'TransformersAdaptedTimesFM3Model'})
                        self.logger.info(f"======M5 省级D+1推理失败,使用 MoE V3.1兜底======, err_msg: {e}")
                    
                elif self.predict_type == "ultra_short": # 省级超短期推理分支
                    try:
                        ds_res_ultra_province = self.ultra_province_model.inference(df)
                        ds_res_ultra_province['jiashu_ultra_prov_16p_pred'] = ds_res_ultra_province['observe_power_predict']
                        # 将16个点的序列拓展到192个点的序列
                        ds_res_ultra_province['jiashu_ultra_prov_16p_pred'] = ds_res_ultra_province['jiashu_ultra_prov_16p_pred'].apply(
                                    lambda x: list(np.append(x, [np.nan]*(self.predict_seq_lens-len(x)) ))
                                    )
                        ds_res_ultra_province.drop(columns=['observe_power_predict'], inplace=True)
                        ds_res_ultra_province['timestamp_win'] = pd.to_datetime(ds_res_ultra_province['timestamp_win'])
                        self.logger.info(f"======Hanjiashu 省级 Ultra 推理完成======")

                    except:
                        # 使用 MoE V3.1 的预测结果兜底
                        ds_res_ultra_province = result_df_province[['timestamp_win', 'station', 'ensemble']].rename(columns={'ensemble': 'jiashu_ultra_prov_16p_pred'})
                        self.logger.info(f"======Hanjiashu 省级 Ultra 推理失败,使用 MoE V3.1 兜底======")

                    try:
                        df_in_prov_ultra_short_m5 = df_province.merge(result_df_province, on=['timestamp_win', 'station'], how='left')  # 省级的DS层推理数据
                        ds_res_ultra_short_province_m5 = unified_inference(
                            self.ultra_short_model_m5,
                            df_in_prov_ultra_short_m5,
                            timestamp_cols=("timestamp_win",),
                            entity_cols=("timestamp_win", "station"),
                            feature_cols=[
                                "general_GHI",
                                "general_GHI_predict",
                                "general_TEMP",
                                "general_TEMP_predict",
                                "observe_power",
                            ],
                        )
                        self.logger.info(f"======M5 省级超短期推理完成======")
                    except Exception as e:
                        # 使用 MoE V3.1 的预测结果兜底
                        ds_res_ultra_short_province_m5 = result_df_province[['timestamp_win', 'station', 'ensemble']].rename(columns={'ensemble': 'TransformersAdaptedTimesFM3Model'})
                        self.logger.info(f"======M5 省级超短期推理失败,使用 MoE V3.1兜底======, err_msg: {e}")

                
                ############################################################################

            # 异常数据-tsfm入口
            if df_abnormal_part.shape[0] == 1:  # 只有一行数据会报错，复制一行，但station不能相同 
                df_abnormal_part = pd.concat([df_abnormal_part, df_abnormal_part], ignore_index=True)
                df_abnormal_part.loc[1, "station"] = "unknow"
            if df_abnormal_part.shape[0] > 0:
                result_df_abnormal_part = self.tsfm_inference_obj.inference(ds_dataframe=df_abnormal_part)
                df_res_list.append(result_df_abnormal_part)

            # 合并结果（支持多个结果依次 merge）
            if len(df_res_list) > 1:
                ds_result = pd.concat(df_res_list, ignore_index=True)
            elif len(df_res_list) == 1:
                ds_result = df_res_list[0]
            else:
                ds_result = pd.DataFrame()
                self.logger.warning("MoE V3.1和tsf均未推理出结果！")

            ds_result.rename(columns={"ensemble": "observe_power_predict"}, inplace=True)
            ds_result = ds_result[~ds_result['station'].isin(['unknow', 'unknow_province'])]
            # 将 ds_result 分为3组：正常的省级预测，正常的站级预测以及异常样本的预测
            ds_result_province = ds_result[ds_result['station'] == area].merge(
                df_normal, on=['timestamp_win','station'], how='inner').drop(columns=['is_abnormal'])
            ds_result_station = ds_result[~(ds_result['station'] == area)].merge(
                df_normal, on=['timestamp_win','station'], how='inner').drop(columns=['is_abnormal'])
            ds_result_abnormal = ds_result.merge(
                df_abnormal, on=['timestamp_win','station'], how='inner').drop(columns=['is_abnormal'])
            ds_result_province['timestamp_win'] = pd.to_datetime(ds_result_province['timestamp_win'])
            ds_result_station['timestamp_win'] = pd.to_datetime(ds_result_station['timestamp_win'])
            ds_result_abnormal['timestamp_win'] = pd.to_datetime(ds_result_abnormal['timestamp_win'])

            ###### DEVELOPE INSERT: DS层的结果的 多个 forecast backbone 的结果列 的 feat_aggregation ######

            #-- 在该代码块里提供两个 DS层的 DataFrame 可以使用: ds_result_province, ds_result_station
            #-- ds_result_province: 省级的DS 层的推理结果的 DataFrame，包含 timestamp_win, station, observe_power_predict 3个列
            #-- observe_power_predict 列，超短期预测结果长度为192，短期预测结果长度为480

            #-- ds_result_station：站级的DS 层的推理结果的 DataFrame, 包含 timestamp_win, station, observe_power_predict 3个列
            #-- observe_power_predict 列，超短期预测结果长度为192，短期预测结果长度为480
            #########
            if self.predict_type == "short": # 短期推理分支

                # merge 每一个省级短期预测结果 DataFrame 的位置
                try:
                    ds_result_province = ds_result_province.merge(ds_res_short_province_m5, on=['timestamp_win','station'], how='left')
                    ds_result_province['timestamp_win'] = pd.to_datetime(ds_result_province['timestamp_win'])
                    ds_result_province = ds_result_province.merge(ds_res_short_province, on=['timestamp_win','station'], how='left') # yili省级短期模型
                except Exception as e: 
                    self.logger.warning(f"===== 省级短期merge出错！错误信息:{str(e)} =====")

                SHORT_PROV_PRED_RULES = [
                    {
                        "agg_output_col": "observe_power_predict_update",  # 保持固定
                        "agg_method": "mean",
                        "min_valid": 1,
                        "input_cols": {"fields": [  # 这里添加新的预测列
                            {"field": "observe_power_predict"},  #原有的MOE V3.1预测
                            {"field": "yili_short_prov_D1_pred"},  # yili省级短期模型
                            {"field": "TransformersAdaptedTimesFM3Model"},  # M5
                        ]},
                    },
                ]
                ds_result_province = aggregate_features(ds_result_province, SHORT_PROV_PRED_RULES)

                # merge 每一个站级短期预测结果 DataFrame 的位置
                try:
                    ds_result_station = ds_result_station.merge(ds_out_station_short_m5, on=['timestamp_win','station'], how='left')
                    ds_result_station['timestamp_win'] = pd.to_datetime(ds_result_station['timestamp_win'])
                    ds_result_station = ds_result_station.merge(ds_out_station_short_m1, on=['timestamp_win','station'], how='left')

                except Exception as e: 
                    self.logger.warning(f"===== 站级短期merge出错！错误信息:{str(e)} =====")

                SHORT_STATION_PRED_RULES = [
                    {
                        "agg_output_col": "observe_power_predict_update",  # 保持固定
                        "agg_method": "mean",
                        "min_valid": 1,
                        "input_cols": {"fields": [  # 这里添加新的预测列
                            {"field": "observe_power_predict"},  #原有的MOE V3.1预测
                            {"field": "TransformersAdaptedM1Model"},  # M1模块结果列
                            {"field": "TransformersAdaptedTimesFM3Model"},  # M5
                        ]},
                    },
                ]
                ds_result_station = aggregate_features(ds_result_station, SHORT_STATION_PRED_RULES)


            elif self.predict_type == "ultra_short": # 超短期推理分支
                # merge 每一个省级超短期预测结果 DataFrame 的位置
                try:
                    ds_result_province = ds_result_province.merge(ds_res_ultra_short_province_m5, 
                                                                on=['timestamp_win','station'], how='left')
                    ds_result_province['timestamp_win'] = pd.to_datetime(ds_result_province['timestamp_win'])

                    ds_result_province = ds_result_province.merge(ds_res_ultra_province,
                                                                  on=['timestamp_win','station'], how='left')  # merge jiashu模型省级超短期预测
                    
                except Exception as e: 
                    self.logger.warning(f"===== 省级超短期merge出错！错误信息:{str(e)} =====")

                ULTRA_PROV_PRED_RULES = [
                    {
                        "agg_output_col": "observe_power_predict_update",  # 保持固定
                        "agg_method": "mean",
                        "min_valid": 1,
                        "missing": "fill_nan",
                        "input_cols": {"fields": [  # 这里添加新的预测列
                            {"field": "observe_power_predict"},  #原有的MOE V3.1预测
                            {"field": "jiashu_ultra_prov_16p_pred"},  # 其他的预测模块结果列
                            {"field": "TransformersAdaptedTimesFM3Model"}  # M5 模型
                        ]},
                    },
                ]
                ds_result_province = aggregate_features(ds_result_province, ULTRA_PROV_PRED_RULES)

                # merge 超短期站级预测结果
                try:
                    ds_result_station = ds_result_station.merge(ds_out_station_ultra_short_m5, on=['timestamp_win','station'], how='left')
                    ds_result_station['timestamp_win'] = pd.to_datetime(ds_result_station['timestamp_win'])
                    ds_result_station = ds_result_station.merge(ds_out_station_ultra_short_m1, on=['timestamp_win','station'], how='left')
                except Exception as e: 
                    self.logger.warning(f"===== 站级超短期merge出错！错误信息:{str(e)} =====")

                ULTRA_SHORT_STATION_PRED_RULES = [
                    {
                        "agg_output_col": "observe_power_predict_update",  # 保持固定
                        "agg_method": "mean",
                        "min_valid": 1,
                        "input_cols": {"fields": [  # 这里添加新的预测列
                            {"field": "observe_power_predict"},  #原有的MOE V3.1预测
                            {"field": "TransformersAdaptedTimesFM3Model"},  # M5
                            {"field": "TransformersAdaptedM1Model"}, # M1 - new pipeline 
                        ]},
                    },
                ]
                ds_result_station = aggregate_features(ds_result_station, ULTRA_SHORT_STATION_PRED_RULES)

            ###############################################################################################
            # 保存 DS 结果表，来存储每个预测模块的DS层结果
            if self.predict_type == "short":
                # 保存省级的预测结果DS层表
                save_ds_table(
                    ds_result_province,
                    base_path=self.train_data_saved_dir,
                    table_name='ds_trajectory_pred_short_prov',  # 表名
                    area=area,
                    predict_type=self.predict_type,
                    report_time=self.report_time,
                    dtype="float16",
                    logger=self.logger
                )
                # 保存站级的预测结果DS层表
                save_ds_table(
                    ds_result_station,
                    base_path=self.train_data_saved_dir,
                    table_name='ds_trajectory_pred_short_station',  # 表名
                    area=area,
                    predict_type=self.predict_type,
                    report_time=self.report_time,
                    dtype="float16",
                    logger=self.logger
                )
                # 保存异常的样本的预测结果DS层表
                save_ds_table(
                    ds_result_abnormal,
                    base_path=self.train_data_saved_dir,
                    table_name='ds_trajectory_pred_short_abnormal',  # 表名
                    area=area,
                    predict_type=self.predict_type,
                    report_time=self.report_time,
                    dtype="float16",
                    logger=self.logger
                )
            elif self.predict_type == "ultra_short":
                # 保存省级的预测结果DS层表
                save_ds_table(
                    ds_result_province,
                    base_path=self.train_data_saved_dir,
                    table_name='ds_trajectory_pred_ultra_short_prov',  # 表名
                    area=area,
                    predict_type=self.predict_type,
                    report_time=self.report_time,
                    dtype="float16",
                    logger=self.logger
                )
                # 保存站级的预测结果DS层表
                save_ds_table(
                    ds_result_station,
                    base_path=self.train_data_saved_dir,
                    table_name='ds_trajectory_pred_ultra_short_station',  # 表名
                    area=area,
                    predict_type=self.predict_type,
                    report_time=self.report_time,
                    dtype="float16",
                    logger=self.logger
                )
                # 保存异常的样本的预测结果DS层表
                save_ds_table(
                    ds_result_abnormal,
                    base_path=self.train_data_saved_dir,
                    table_name='ds_trajectory_pred_ultra_short_abnormal',  # 表名
                    area=area,
                    predict_type=self.predict_type,
                    report_time=self.report_time,
                    dtype="float16",
                    logger=self.logger
                )

            ###############################################################################################

            ds_result_province = ds_result_province[['timestamp_win', 'station', 'observe_power_predict_update']]
            ds_result_province.rename(columns={"observe_power_predict_update": "observe_power_predict"}, inplace=True)

            ds_result_station = ds_result_station[['timestamp_win', 'station', 'observe_power_predict_update']]
            ds_result_station.rename(columns={"observe_power_predict_update": "observe_power_predict"}, inplace=True)

            ds_result_abnormal = ds_result_abnormal[['timestamp_win', 'station', 'observe_power_predict']]
            ds_result = pd.concat([ds_result_station, ds_result_province, ds_result_abnormal], ignore_index=True) # Fix: 修复了漏掉的异常的样本，在之前的代码加上 left join

            try:  # 保存DS数据
                save_cols = [
                    'timestamp_win',
                    'station',
                    'observe_power',
                    'observe_power_future',
                    'ssrd_pos_1',
                    'ssrd_pos_2',
                    'ssrd_pos_3',
                    'ssrd_pos_4',
                    'ssrd_pos_5',
                    'ssrd_pos_6',
                    'ssrd_pos_7',
                    'ssrd_pos_8',
                    'ssrd_pos_9',
                    't2m_pos_1',
                    't2m_pos_2',
                    't2m_pos_3',
                    't2m_pos_4',
                    't2m_pos_5',
                    't2m_pos_6',
                    't2m_pos_7',
                    't2m_pos_8',
                    't2m_pos_9',
                    'GHI_SOLARGIS',
                    'TEMP_SOLARGIS',
                    'ssrd_pos_1_predict',
                    'ssrd_pos_2_predict',
                    'ssrd_pos_3_predict',
                    'ssrd_pos_4_predict',
                    'ssrd_pos_5_predict',
                    'ssrd_pos_6_predict',
                    'ssrd_pos_7_predict',
                    'ssrd_pos_8_predict',
                    'ssrd_pos_9_predict',
                    't2m_pos_1_predict',
                    't2m_pos_2_predict',
                    't2m_pos_3_predict',
                    't2m_pos_4_predict',
                    't2m_pos_5_predict',
                    't2m_pos_6_predict',
                    't2m_pos_7_predict',
                    't2m_pos_8_predict',
                    't2m_pos_9_predict',
                    'GHI_SOLARGIS_predict',
                    'TEMP_SOLARGIS_predict',
                ]
                # 合入predict_power
                df["timestamp_win"] = pd.to_datetime(df["timestamp_win"])
                ds_result["timestamp_win"] = pd.to_datetime(ds_result["timestamp_win"])
                df = pd.merge(df, ds_result, how='inner', on=['timestamp_win', 'station'])
                df["timestamp_win"] = pd.to_datetime(df["timestamp_win"])  # bugfix-09.29: 修复 merge 后 timestamp_win 变为 object 的bug
                new_col = ['observe_power_predict']
                if self.predict_type == "ultra_short":
                    ef_config = {
                        'initial_threshold_max': 0.8,
                        'initial_threshold_min': -0.25,
                        'split_point': 0.25,
                        'mid_factor': 0.35,
                        'end_factor': 0.1,
                        'keep_first': 16
                    }
                    ef_province_config = {
                        'initial_threshold_max': 0.27,
                        'initial_threshold_min': -0.27,
                        'rate': 0.1,
                        'keep_first': 1
                    }
                    self.logger.info("start into Station EF V2 module; Province EF module")
                    try:
                        df_province_input = df[df['station'] == area]
                        df_station_input = df[~(df['station'] == area)]
                        self.logger.info(f"ef input province time: {list(set(df_province_input['timestamp_win'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()))}")
                        self.logger.info(f"ef input station time: {list(set(df_station_input['timestamp_win'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()))}")
                        ef_input_path = os.path.join(self.train_data_saved_dir, self.predict_type)
                        df_province_output = fix_observe_power_predict(df_province_input, ef_input_path,
                                                                       config=ef_province_config)

                        df_station_output = fix_observe_power_predict_linear(df_station_input, ef_input_path,
                                                                      config=ef_config)
                        new_col += ['observe_power_predict_error_fix_16p']
                        # 上下合并 station 与 province
                        df = pd.concat([df_station_output, df_province_output], ignore_index=True)
                    except Exception as e:
                        self.logger.warning(f"EF_V2修正失败，跳过EF修正，使用原始预测值：{e}")
                save_ds_data_cache(
                    df=df[save_cols + new_col], report_time=self.report_time, area=area, predict_type=self.predict_type,
                    base_save_path=self.train_data_saved_dir
                )
            except Exception as e:
                ds_result["ensemble"] = ds_result["observe_power_predict"]
                self.logger.error(f"multi_infer: save ds data failed, error msg: {e}")

            # save_ds_data_cache 之后处理 ensemble 列：observe_power_predict 保持不动，
            # 仅当 error_fix 列非空时用其覆盖 ensemble；最后同步回 ds_result 供 ODS 转换
            if 'observe_power_predict' in df.columns:
                df["ensemble"] = df["observe_power_predict"]
                if 'observe_power_predict_error_fix_16p' in df.columns:
                    fix_mask = df['observe_power_predict_error_fix_16p'].notna()
                    if fix_mask.any():
                        df.loc[fix_mask, "ensemble"] = df.loc[fix_mask, "observe_power_predict_error_fix_16p"]  # TODO: 处理逻辑待优化
                        self.logger.info(f"EF修正应用到 ensemble: {int(fix_mask.sum())} 行")
                df["timestamp_win"] = pd.to_datetime(df["timestamp_win"])
                ds_result = ds_result.drop(columns=['ensemble'], errors='ignore').merge(
                    df[['timestamp_win', 'station', 'ensemble']], on=['timestamp_win', 'station'], how='left')
            else:
                ds_result["ensemble"] = ds_result["observe_power_predict"]
            result_df = ds_to_ods(ds_result)
            value_cols = result_df.columns.difference(['dtime'])
            result_df[value_cols] = result_df[value_cols].clip(lower=0)  # 负值置零

            result_df = result_df[
                (result_df['dtime'] >= self.result_s_time) & (result_df['dtime'] <= self.result_e_time)]
            # 夜间结果置零
            result_df = zero_night_hours(result_df)

            if self.predict_type == "short":
                date_str = datetime.datetime.strptime(self.report_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d')
            else:
                date_str = datetime.datetime.strptime(self.report_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            save_full_path = os.path.join(self.save_path, f"hw_nuoya_{date_str}_{self.predict_type}_{area}.parquet")

            result_df.fillna(0).to_parquet(save_full_path)
            self.logger.info(f"结果保存成功：{save_full_path}, shape: {result_df.shape}")

            self.logger.info(f"multi_infer: finish handle... cost time: {time.time() - start_t}")


def zero_night_hours(
        df: pd.DataFrame,
        start_time: str = "20:00",
        end_time: str = "06:00",
        dtime_col: str = "dtime",
) -> pd.DataFrame:
    """
    将 DataFrame 中指定夜间时段的预测值置零。
    时间精度到分钟，支持 15 分钟频率的精确边界控制。

    Args:
        df: 输入 DataFrame，需包含时间列
        start_time: 夜间开始时刻，格式 HH:MM（含），默认 "20:00"
        end_time: 夜间结束时刻，格式 HH:MM（含），默认 "06:00"
        dtime_col: 时间列名，默认 'dtime'

    Returns:
        处理后的 DataFrame（已置零夜间预测值）
    """
    if dtime_col not in df.columns:
        raise ValueError(f"时间列 '{dtime_col}' 不存在，现有列: {list(df.columns)}")

    df = df.copy(deep=True)
    time_str = df[dtime_col].dt.strftime('%H:%M')
    if start_time <= end_time:
        mask = (time_str >= start_time) & (time_str <= end_time)
    else:
        mask = (time_str >= start_time) | (time_str <= end_time)

    pred_cols = [c for c in df.columns if c != dtime_col]
    zeros = int(mask.sum())
    if zeros > 0:
        df.loc[mask, pred_cols] = 0.0
        multi_logger.info(f"夜间置零: {zeros}/{len(df)} ({zeros / len(df) * 100:.1f}%)")
    else:
        multi_logger.info(f"夜间置零: 0/{len(df)} (0.0%) - 该时间范围内无夜间数据")

    multi_logger.info(f"夜间时段: {start_time} ~ {end_time}")
    multi_logger.info(f"影响列数: {len(pred_cols)} 个预测列")
    return df


def _sanitize_for_parquet(val):
    """递归将 ndarray / list / numpy 标量转为 Parquet 友好的原生 Python 类型"""
    if isinstance(val, np.ndarray):
        if np.issubdtype(val.dtype, np.number):
            return val.astype(np.float64).tolist()
        return [_sanitize_for_parquet(v) for v in val.tolist()]
    elif isinstance(val, list):
        return [_sanitize_for_parquet(v) for v in val]
    elif isinstance(val, np.generic):
        return val.item()
    return val

def save_ds_data_cache(df, report_time, area, predict_type, base_save_path):
    report_dt = datetime.datetime.strptime(report_time, '%Y-%m-%d %H:%M:%S')
    date_str = report_dt.strftime('%Y-%m-%d')
    time_str = report_dt.strftime('%H:%M')
    dir_path = os.path.join(
        base_save_path, predict_type,
        f"area={area}", f"date={date_str}", f"time={time_str}",
    )
    os.makedirs(dir_path, exist_ok=True)
    date_full_str = datetime.datetime.strptime(report_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')
    save_path = os.path.join(dir_path, f"hw_nuoya_ds_{date_full_str}_{predict_type}_{area}.parquet")

    # === 改进：自动检测并转换所有含 ndarray/list/numpy 标量的列 ===
    df_to_save = df.copy()
    for col in df_to_save.columns:
        sample_series = df_to_save[col].dropna()
        if not sample_series.empty:
            sample = sample_series.iloc[0]
            if isinstance(sample, (np.ndarray, list, np.generic)):
                df_to_save[col] = df_to_save[col].map(_sanitize_for_parquet)
    # =============================================================

    df_to_save.to_parquet(save_path, index=False)
    multi_logger.info(f"[{report_time}] ✓ DS数据保存成功: {save_path}, shape={df.shape}")

def fix_observe_power_tail_miss(df_observe_power, pipeline_backbone, fix_len: int=600):
    """nuoya 算法补全 observe_power 列最后几个点延迟的缺失"""
    def _extract(seq):
        arr = np.asarray(seq, dtype=float)
        nan_pos = np.nonzero(np.isnan(arr))[0]
        pos_idx = int(nan_pos[0]) if len(nan_pos) else len(arr)
        start = max(0, pos_idx - fix_len)
        fixlen = arr[start:pos_idx]
        return pos_idx, fixlen.tolist()

    out = df_observe_power.copy()
    extracted = df_observe_power["observe_power"].apply(_extract)
    out["pos_idx"] = extracted.map(lambda t:t[0])
    out["observe_power_fixlen"] = extracted.map(lambda t: t[1])

    # 单变量推理: 以 observe_power_fixlen 为输入序列
    infer_input = out[["timestamp_win", "station", "observe_power_fixlen"]].rename(columns={"observe_power_fixlen": "observe_power"})
    pred = unified_inference(
        pipeline_backbone,
        infer_input,
        timestamp_cols=("timestamp_win",),
        entity_cols=("timestamp_win", "station"),
        feature_cols=["observe_power"])
    pred_cols = [c for c in pred.columns if c not in ("timestamp_win", "station")]
    out["observe_power_fixlen_pred"] = pred[pred_cols[0]].tolist()

    # 修复：用预测结果从 pos_idx 开始填充 observe_power 的尾部缺失
    def _repair(row):
        arr = np.asarray(row["observe_power"], dtype=float)
        pos = int(row["pos_idx"])
        pred_seq = np.asarray(row["observe_power_fixlen_pred"], dtype=float)
        n_fill = min(len(arr) - pos, len(pred_seq))    # 尾部缺失数 vs 预测长度
        if n_fill > 0:
            arr[pos: pos + n_fill] = pred_seq[:n_fill]
        return arr.tolist()

    out["observe_power_repaired"] = out.apply(_repair, axis=1)
    return out 


if __name__ == '__main__':
    # start_time = datetime.datetime(2026, 8, 10, 0, 00)
    # end_time = datetime.datetime(2026, 8, 16, 23, 45)
    # # interval = datetime.timedelta(minutes=15)
    # interval = datetime.timedelta(days=1)
    # current_time = start_time

    # forecasting_type = "short"  # short, ultra_short

    # if forecasting_type == "ultra_short":
    #     while current_time <= end_time:
    #         t = current_time.strftime("%Y-%m-%d %H:%M:%S")
    #         MultiInfer(
    #             report_time=t,
    #             predict_type=forecasting_type,
    #         ).handle()
    #         current_time += interval

    #         # raise ValueError("STOP!")
    # elif forecasting_type == "short":
    #     while current_time <= end_time:
    #         t = current_time.strftime("%Y-%m-%d %H:%M:%S")
    #         MultiInfer(
    #             report_time=t,
    #             predict_type=forecasting_type,
    #         ).handle()
    #         current_time += interval

    parser = argparse.ArgumentParser(description="multi_infer.py")
    parser.add_argument("--report_time", type=str, required=True)
    parser.add_argument("--predict_type", type=str, required=True, choices=["ultra_short","short"])
    args = parser.parse_args()
    MultiInfer(report_time=args.report_time, predict_type=args.predict_type).handle()

    """
    示例调用代码:
    python multi_infer.py --report_time "2026-09-12 10:00:00" --predict_type short
    """