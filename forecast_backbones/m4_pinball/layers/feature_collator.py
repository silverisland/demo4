import torch.nn as nn
import torch
import pandas as pd
from datetime import datetime

class HFFeatureCollator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.weather_history = configs["dataset"]["conv_cols"]
        self.weather_future =  configs["dataset"]['conv_predict_cols']
        self.pv_history = configs["dataset"]["pv_col"]
        self.tsfm = configs["dataset"]['tsfm_col']
        self.target = configs["dataset"]["target_col"]

        self.register_buffer("_device_anchor", torch.empty(0))

    # def data_preprocess(self, batch_dict):
    #     for i in range(len(self.configs["dataset"]["feature_aggregations"])):
    #         agg_output_col = self.configs["dataset"]["feature_aggregations"][i]['agg_output_col']
    #         raw_input_cols = self.configs["dataset"]["feature_aggregations"][i]['raw_input_cols']
    #         x_raw_input = torch.stack([batch_dict[key] for key in raw_input_cols], dim=1).mean(dim = 1)
    #         batch_dict[agg_output_col] = x_raw_input
    #     return batch_dict

    def forward(self, batch_dict): # 注意：这里改名为 forward，因为它现在是计算图的一部分

        target_device = self._device_anchor.device

        # batch_dict = self.data_preprocess(batch_dict) # 这里注释掉是因为我们再外部已经做过一遍了，这里不用重新重复

        x1 = torch.stack([batch_dict[key] for key in self.weather_history], dim=1).to(target_device)    # 原有dataset： torch.Size([512, 4, 672]) 现在FeatureCollator： torch.Size([512, 4, 672])
        x2 = torch.stack([batch_dict[key] for key in self.weather_future], dim=1).to(target_device)     # 原有dataset： torch.Size([512, 4, 192]) 现在FeatureCollator： torch.Size([512, 4, 192])
        x3 = torch.stack([batch_dict[key] for key in [self.pv_history]], dim=1).to(target_device)       # 原有dataset： torch.Size([512, 672])    现在FeatureCollator： torch.Size([512, 1, 672])
        x4 = torch.stack([batch_dict[key] for key in [self.tsfm]], dim=1).to(target_device)             # 原有dataset： torch.Size([512, 192])    现在FeatureCollator： torch.Size([512, 1, 192])
        # y = torch.stack([batch_dict[key] for key in [self.target]], dim=1).to(target_device)            # 原有dataset： torch.Size([512, 192])    现在FeatureCollator： torch.Size([512, 1, 192])

        # y = y.squeeze(1)
        x3 = x3.squeeze(1)
        x4 = x4.squeeze(1)

        return x1, x2, x3, x4


import torch.nn as nn
import torch
import pandas as pd
from datetime import datetime

class FeatureCollator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.weather_history = configs.conv_cols
        self.weather_future =  configs.conv_predict_cols
        self.pv_history = configs.pv_col
        self.tsfm = configs.tsfm_col
        self.target = configs.target_col

        self.register_buffer("_device_anchor", torch.empty(0))

    def data_preprocess(self, batch_dict):
        for i in range(len(self.configs.feature_aggregations)):
            agg_output_col = self.configs.feature_aggregations[i]['agg_output_col']
            raw_input_cols = self.configs.feature_aggregations[i]['raw_input_cols']
            x_raw_input = torch.stack([batch_dict[key] for key in raw_input_cols], dim=1).mean(dim = 1)
            batch_dict[agg_output_col] = x_raw_input
        return batch_dict

    def forward(self, batch_dict): # 注意：这里改名为 forward，因为它现在是计算图的一部分

        target_device = self._device_anchor.device

        batch_dict = self.data_preprocess(batch_dict)

        x1 = torch.stack([batch_dict[key] for key in self.weather_history], dim=1).to(target_device)    # 原有dataset： torch.Size([512, 4, 672]) 现在FeatureCollator： torch.Size([512, 4, 672])
        x2 = torch.stack([batch_dict[key] for key in self.weather_future], dim=1).to(target_device)     # 原有dataset： torch.Size([512, 4, 192]) 现在FeatureCollator： torch.Size([512, 4, 192])
        x3 = torch.stack([batch_dict[key] for key in [self.pv_history]], dim=1).to(target_device)       # 原有dataset： torch.Size([512, 672])    现在FeatureCollator： torch.Size([512, 1, 672])
        x4 = torch.stack([batch_dict[key] for key in [self.tsfm]], dim=1).to(target_device)             # 原有dataset： torch.Size([512, 192])    现在FeatureCollator： torch.Size([512, 1, 192])
        y = torch.stack([batch_dict[key] for key in [self.target]], dim=1).to(target_device)            # 原有dataset： torch.Size([512, 192])    现在FeatureCollator： torch.Size([512, 1, 192])

        y = y.squeeze(1)
        x3 = x3.squeeze(1)
        x4 = x4.squeeze(1)

        return x1, x2, x3, x4, y
