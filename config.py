import os

# 数据路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BK_DATA_DIR = os.path.join(BASE_DIR, 'data', 'bk')

# 板块配置文件路径
BK_CONFIG_PATH = os.path.join(BASE_DIR, 'data', 'bk_config.json') 