"""Cross-source aggregation rules for the nine selected weather grid points."""

POINTS = (7, 8, 9, 12, 13, 14, 17, 18, 19)

RULES = [
    {
        "agg_output_col": "solar_southern_GHI", # 1
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [{"field": f"solar_southern_GHI_p{point}"} for point in POINTS]},
    },
    {
        "agg_output_col": "solar_southern_TEMP", # 1
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [{"field": f"solar_southern_t2m_p{point}"} for point in POINTS]},
    },
    {
        "agg_output_col": "GDFS_JTHFGNED_MID_JTNWQISX_GHI",
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [{"field": f"GDFS_JTHFGNED_MID_JTNWQISX_GHI_p{point}"} for point in POINTS]},
    },
    {
        "agg_output_col": "GDFS_JTHFGNED_MID_JTNWQISX_TEMP",
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [{"field": f"GDFS_JTHFGNED_MID_JTNWQISX_tem_hl_10_p{point}"} for point in POINTS]},
    },
    {
        "agg_output_col": "EC_JTKRD7RO_GHI",
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [{"field": f"EC_JTKRD7RO_GHI_p{point}"} for point in POINTS]},
    },
    {
        "agg_output_col": "EC_JTKRD7RO_TEMP",
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [{"field": f"EC_JTKRD7RO_t2m_p{point}"} for point in POINTS]},
    },
    {
        "agg_output_col": "general_GHI",  # 1
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [
            # {"field": "solar_southern_GHI"},
            {"field": "GDFS_JTHFGNED_MID_JTNWQISX_GHI"},
            {"field": "EC_JTKRD7RO_GHI"},
            {"field": "GHI_SOLARGIS"},
        ]},
    },
    {
        "agg_output_col": "general_TEMP",  # 1
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [
            # {"field": "solar_southern_TEMP"},
            {"field": "GDFS_JTHFGNED_MID_JTNWQISX_TEMP"},
            {"field": "EC_JTKRD7RO_TEMP"},
            {"field": "TEMP_SOLARGIS"},
        ]},
    },
    ### PREDICT
    {
        "agg_output_col": "solar_southern_GHI_predict",  # 1
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [{"field": f"solar_southern_GHI_p{point}_predict"} for point in POINTS]},
    },
    {
        "agg_output_col": "solar_southern_TEMP_predict",  # 1
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [{"field": f"solar_southern_t2m_p{point}_predict"} for point in POINTS]},
    },
    {
        "agg_output_col": "GDFS_JTHFGNED_MID_JTNWQISX_GHI_predict",
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [{"field": f"GDFS_JTHFGNED_MID_JTNWQISX_GHI_p{point}_predict"} for point in POINTS]},
    },
    {
        "agg_output_col": "GDFS_JTHFGNED_MID_JTNWQISX_TEMP_predict",
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [{"field": f"GDFS_JTHFGNED_MID_JTNWQISX_tem_hl_10_p{point}_predict"} for point in POINTS]},
    },
    {
        "agg_output_col": "EC_JTKRD7RO_GHI_predict",
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [{"field": f"EC_JTKRD7RO_GHI_p{point}_predict"} for point in POINTS]},
    },
    {
        "agg_output_col": "EC_JTKRD7RO_TEMP_predict",
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [{"field": f"EC_JTKRD7RO_t2m_p{point}_predict"} for point in POINTS]},
    },
    {
        "agg_output_col": "general_GHI_predict",  # 1
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [
            # {"field": "solar_southern_GHI_predict"},
            {"field": "GDFS_JTHFGNED_MID_JTNWQISX_GHI_predict"},
            {"field": "EC_JTKRD7RO_GHI_predict"},
            {"field": "GHI_SOLARGIS_predict"},
        ]},
    },
    {
        "agg_output_col": "general_TEMP_predict",  # 1
        "agg_method": "median",
        "min_valid": 2,
        "input_cols": {"fields": [
            # {"field": "solar_southern_TEMP_predict"},
            {"field": "GDFS_JTHFGNED_MID_JTNWQISX_TEMP_predict"},
            {"field": "EC_JTKRD7RO_TEMP_predict"},
            {"field": "TEMP_SOLARGIS_predict"},
        ]},
    },
    # ===== 临时使用genral气象替代AI气象 ===== 20260920 14:53关闭; 20260921 9:30打开
    # {
    #     "agg_output_col": "solar_southern_GHI_predict",
    #     "agg_method": "median",
    #     "min_valid": 1,
    #     "input_cols": {"fields": [
    #         {"field": "general_GHI_predict"},
    #     ]},
    # },
    # {
    #     "agg_output_col": "solar_southern_TEMP_predict",
    #     "agg_method": "median",
    #     "min_valid": 1,
    #     "input_cols": {"fields": [
    #         {"field": "general_TEMP_predict"},
    #     ]},
    # },
    # {
    #     "agg_output_col": "solar_southern_GHI",
    #     "agg_method": "median",
    #     "min_valid": 1,
    #     "input_cols": {"fields": [
    #         {"field": "general_GHI"},
    #     ]},
    # },
    # {
    #     "agg_output_col": "solar_southern_TEMP",
    #     "agg_method": "median",
    #     "min_valid": 1,
    #     "input_cols": {"fields": [
    #         {"field": "general_TEMP"},
    #     ]},
    # },
    # ============================================
]


if __name__ == "__main__":
    assert len(RULES) == 16