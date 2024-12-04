import os
import json
from config import BK_DATA_DIR, BK_CONFIG_PATH
from utils.Log import Log, datetime_to_timestamp
import pickle
import time
from collections import defaultdict
import numpy as np
import requests
import re
from datetime import datetime


def get_klines_data(gp_code: str) -> dict:
    """
    获取单个gp代号对应的全数据
    :param gp_code: '601360'
    :return: dic_klines[date_timestamp] = {
                '开盘价': float(klines[i][1]),
                '收盘价': float(klines[i][2]),
                '最高价': float(klines[i][3]),
                '最低价': float(klines[i][4]),
                '总手': float(klines[i][5]),
                '金额': float(klines[i][6]),
                '振幅': float(klines[i][7]),
                '涨跌幅': float(klines[i][8]),
                '涨跌额': float(klines[i][9]),
                "换手率": float(klines[i][10]),
                'date_timestamp': date_timestamp
            }
    """

    def parse_data(jsonobj_inside: dict) -> dict:
        klines = jsonobj_inside['data']['klines']

        dic_klines = {}
        for i in range(0, len(klines)):
            klines[i] = klines[i].split(',')
            # print(klines)
            date = klines[i][0]
            date_time = datetime.strptime(date, "%Y-%m-%d")
            date_timestamp = date_time.timestamp()  # 转换为时间戳
            dic_klines[date_timestamp] = {
                '开盘价': float(klines[i][1]),
                '收盘价': float(klines[i][2]),
                '最高价': float(klines[i][3]),
                '最低价': float(klines[i][4]),
                '总手': float(klines[i][5]),
                '金额': float(klines[i][6]),
                '振幅': float(klines[i][7]),
                '涨跌幅': float(klines[i][8]),
                '涨跌额': float(klines[i][9]),
                "换手率": float(klines[i][10]),
                'date_timestamp': date_timestamp
            }
        return dic_klines

    timestamp = int(time.time() * 1000)
    unknown = "35106668583059676032"
    url_1 = "http://21.push2his.eastmoney.com/api/qt/stock/kline/get"
    call_back = f"jQuery{unknown}_{timestamp}"
    url_2 = f"?cb={call_back}"
    url_3 = f"&secid=1.{gp_code}{'&ut=fa5fd1943c7b386f172d6893dbfba10b'}"
    url_4 = "&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6"
    url_5 = "&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61"
    url_6 = "&klt=101"
    url_7 = "&fqt=1"
    url_8 = "&end=20500101"
    url_9 = "&lmt=1000000"
    url_10 = f"&_={timestamp}"
    url = f"{url_1}{url_2}{url_3}{url_4}{url_5}{url_6}{url_7}{url_8}{url_9}{url_10}"
    try:
        try:
            res = requests.get(
                url=url,
                headers={
                    'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1'
                },
                timeout=8
            )
            jsonobj = json.loads(res.text.split("(")[1][:-2])
        except:
            Log.warning(f"获取 {gp_code} 股票信息失败，网络问题")
            return None

        data = jsonobj['data']
        code = data['code']
    except:
        url_1 = "http://92.push2his.eastmoney.com/api/qt/stock/kline/get"
        call_back = f"jQuery{unknown}_{timestamp}"
        url_2 = f"{'?cb='}{call_back}"
        url_3 = f"{'&secid=0.'}{gp_code}{'&ut=fa5fd1943c7b386f172d6893dbfba10b'}"
        url_4 = "&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6"
        url_5 = "&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61"
        url_6 = "&klt=101"
        url_7 = "&fqt=1"
        url_8 = "&end=20500101"
        url_9 = "&lmt=1000000"
        url_10 = f"&_={timestamp}"
        url = f"{url_1}{url_2}{url_3}{url_4}{url_5}{url_6}{url_7}{url_8}{url_9}{url_10}"
        try:
            res = requests.get(
                url=url,
                headers={
                    'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1'
                },
                timeout=8
            )
            jsonobj = json.loads(res.text.split(call_back + "(")[1][:-2])
        except:
            Log.warning(f"获取 {gp_code} 股票信息失败，网络问题")
            return None

    dic_klines_data = parse_data(jsonobj_inside=jsonobj)
    return dic_klines_data


class BKManager:
    @staticmethod
    def get_available_bks():
        """获取所有可用的板块信息，返回给前端"""
        # 读取板块配置
        with open(BK_CONFIG_PATH, 'r', encoding='utf-8') as f:
            bk_config = json.load(f)

        # 获取实际存在的板块目录
        available_bks = []
        if os.path.exists(BK_DATA_DIR):
            for bk_code in os.listdir(BK_DATA_DIR):
                if bk_code in bk_config:
                    available_bks.append({
                        'code': bk_code,
                        'name': bk_config[bk_code],
                        'path': os.path.join(BK_DATA_DIR, bk_code)
                    })
        return available_bks

    @staticmethod
    def scan(bk_index="BK0655", bk_name='网络安全') -> dict:
        """
        根据已知的BK代号和BK名，扫描该BK下所有GP的名字和代号
        :param bk_index: BK0655
        :param bk_name: 网络安全
        :return: {'gp_code':gp_name}
        """
        Log.info(f"板块扫描启动，开始遍历板块 {bk_name} 可能存在的所有股票")
        dic_current_bk = {}
        for page_now in range(1, 20):
            res = requests.get(
                url="https://push2.eastmoney.com/api/qt/clist/get",
                params={
                    'cb': f"jQuery阿巴阿巴_{int(time.time())}",
                    'fid': 'f62',
                    'po': '1',
                    'pz': '50',
                    'pn': page_now,
                    'np': '1',
                    'fltt': '2',
                    'invt': '2',
                    'ut': 'b2884a393a59ad64002292a3e90d46a5',
                    'fs': f"b:{bk_index}"
                }
            )
            matches = re.findall(r'\((.*?)\);', string=res.text)
            if matches:
                json_data = json.loads(matches[0])
                data = json_data.get('data')
                if data:
                    diff = data.get('diff')
                    if diff:
                        for item in diff:
                            code = item['f12']
                            name = item['f14']
                            dic_current_bk[str(code)] = name
                            Log.debug(f"扫描到：{name}")
                    else:
                        Log.trace(f"{bk_index}:{bk_name}在{page_now}页及之后无内容")
                        break
                else:
                    Log.trace(f"{bk_index}:{bk_name}在{page_now}页及之后无内容，获取完毕")
                    break
            else:
                Log.trace(f"{bk_index}:{bk_name}在{page_now}页及之后无内容")
                break
        return dic_current_bk

    @staticmethod
    def update_bk_data(bk_scan_result: dict, bk_index=None, save=True) -> dict:
        """
        根据BK扫描结果得出的所有gp代号，遍历获取所有数据，默认保存数据到本地
        :param bk_index: BK0655
        :param save: 是否将结果保存到本地
        :param bk_scan_result: result from bk_scan
        :return:
        {'gp_code':gp_klines_data}
        其中gp_klines_data包括{
                    '开盘价': float,
                    '收盘价': float,
                    '最高价': float,
                    '最低价': float,
                    '总手': float,
                    '金额': float,
                    '振幅': float,
                    '涨跌幅': float,
                    '涨跌额': float,
                    "换手率": float,
                    'date_timestamp': date_timestamp
                }
        """
        dic_result = {}
        for gp_code, gp_name in bk_scan_result.items():
            gp_klines_data = get_klines_data(gp_code)
            if gp_klines_data:
                dic_result[gp_code] = gp_klines_data
                Log.debug(f"{gp_name}获取完毕")
            else:
                continue
        # dic_result = bk.生成布林带数据(bk_result=dic_result)
        dic_result = BKManager.生成MACD数据(bk_result=dic_result)
        dic_result = BKManager.生成RSI数据(bk_result=dic_result)
        dic_result = BKManager.生成KDJ数据(bk_result=dic_result)
        dic_result = BKManager.生成动量数据(bk_result=dic_result)
        if save:
            BKManager.save_data_to_pkl(bk_result=dic_result, bk_index=bk_index)
        return dic_result

    @staticmethod
    def save_data_to_pkl(bk_result, bk_index):
        """
        将板块数据保存为pickle数据格式
        :param bk_result:
        :param bk_index:板块代号
        :return:
        """
        dir_path = os.path.join(BK_DATA_DIR, bk_index)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        # 使用pickle实测大小为txt的45%左右
        with open(f"{dir_path}/data.pkl", 'wb') as f:
            pickle.dump(bk_result, f)

    @staticmethod
    def load_data_from_file(bk_index: str) -> dict:
        """
        读取本地保存的板块pk文件
        :param bk_index: BK0655
        :return: {股票一:{第一天时间戳:{数据字典},第二天时间戳:{数据字典}}，股票二:{第一天时间戳:{数据字典},第二天时间戳:{数据字典}}}
        例如：
        873001：{
        1641484800.0:
        {'开盘价': 8.0, '收盘价': 10.01, '最高价': 10.01, '最低价': 8.0, '总手': 5.0, '金额': 4402.0, '振幅': 0.0, '涨跌幅': 0.0, '涨跌额': 0.0, '换手率': 0.0, 'date_timestamp': 1641484800.0},
        1641744000.0:
        {
        """
        start_time = time.time()
        dir_path = os.path.join(BK_DATA_DIR, bk_index)
        with open(f"{dir_path}/data.pkl", 'rb') as f:
            loaded_dict = pickle.load(f)
        Log.debug(f"读取板块{bk_index}数据库完毕，耗时{time.time() - start_time}秒")
        return loaded_dict

    @staticmethod
    def load_multi_data_from_file(bk_indexs: list[str]):
        """
        读取多个本地bk数据pk文件
        :param bk_indexs:
        :return:
        """
        final_data = {}
        for bk_index in bk_indexs:
            loaded_dict = BKManager.load_data_from_file(bk_index=bk_index)
            final_data = {**loaded_dict, **final_data}
        return final_data

    @staticmethod
    def produce_simData(bk_result: dict, start_date_time=None) -> dict:
        """
        将以个股代号为键的原数据，转换成以时间为键的字典，便于仿真
        :return:
        """
        start_timestamp = datetime_to_timestamp(start_date_time)  # 将日期如"2011-03-22"转换为时间戳
        dic_temp = defaultdict(dict)
        for gp_code, gp_data in bk_result.items():
            if len(str(gp_code)) != 6:
                continue
            for date_timestamp, gp_klines_data in gp_data.items():
                if date_timestamp >= start_timestamp:
                    dic_temp[date_timestamp][gp_code] = gp_klines_data
        dic_result = {}
        for date_timestamp in sorted(dic_temp):
            dic_result[date_timestamp] = dic_temp[date_timestamp]
        return dic_result

    @staticmethod
    def 生成MACD数据(bk_result, short_window=12, long_window=26, signal_window=9):
        """
        计算MACD指标并添加到数据包
        :param bk_result:
        :param short_window: 短期EMA窗口期
        :param long_window: 长期EMA窗口期
        :param signal_window: 信号线EMA窗口期
        :return: 包含MACD数据的结果
        """

        def calculate_ema(prices, window):
            """
            计算指数移动平均线
            :param prices: 收盘价列表
            :param window: EMA的窗口大小
            :return: EMA列表
            """
            ema = [np.mean(prices[:window])]  # 计算初始的简单移动平均
            multiplier = 2 / (window + 1)
            for price in prices[window:]:
                ema.append((price - ema[-1]) * multiplier + ema[-1])
            return [0] * (window - 1) + ema  # 前window个点没有EMA值，填0

        def calculate_macd(list_close_price, short_window, long_window, signal_window):
            """
            计算MACD及信号线
            :param list_close_price: 收盘价列表
            :param short_window: 短期EMA窗口
            :param long_window: 长期EMA窗口
            :param signal_window: 信号线EMA窗口
            :return: MACD线、信号线和DIF值
            """
            short_ema = calculate_ema(list_close_price, short_window)
            long_ema = calculate_ema(list_close_price, long_window)
            macd_line = [s - l for s, l in zip(short_ema, long_ema)]  # DIF = 短期EMA - 长期EMA
            signal_line = calculate_ema(macd_line, signal_window)  # 信号线是MACD线的EMA
            macd_histogram = [m - s for m, s in zip(macd_line, signal_line)]  # MACD柱状图 = DIF - 信号线
            return short_ema, long_ema, macd_line, signal_line, macd_histogram

        Log.info(f"开始计算MACD数据")
        start_time = time.time()

        # 以个股代号为键，一个包含所有收盘价的列表为值
        for gp_code, gp_data in bk_result.items():
            list_收盘价列表 = [value['收盘价'] for value in gp_data.values()]
            short_ema, long_ema, macd_line, signal_line, macd_histogram = calculate_macd(
                list_close_price=list_收盘价列表,
                short_window=short_window,
                long_window=long_window,
                signal_window=signal_window
            )
            ema_200 = calculate_ema(prices=list_收盘价列表, window=200)
            # bk.测试绘画MACD数据(收盘价列表=list_收盘价列表, macd_line=macd_line, signal_line=signal_line, macd_histogram=macd_histogram)
            index = 0
            for timestamp, 所有股票当天数据字典 in gp_data.items():
                bk_result[gp_code][timestamp]['ema_200'] = ema_200[index]
                bk_result[gp_code][timestamp]['short_ema'] = short_ema[index]
                bk_result[gp_code][timestamp]['long_ema'] = long_ema[index]
                bk_result[gp_code][timestamp]['macd_line'] = macd_line[index]
                bk_result[gp_code][timestamp]['signal_line'] = signal_line[index]
                bk_result[gp_code][timestamp]['macd_histogram'] = macd_histogram[index]
                index += 1

        Log.debug(f"MACD计算完毕，耗时{time.time() - start_time}秒")
        return bk_result

    @staticmethod
    def 生成WR数据(bk_result, window_size=14):
        """
        计算WR指标并添加到数据包
        :param bk_result: 数据包
        :param window_size: WR计算的窗口期
        :return: 包含WR数据的结果
        """

        def calculate_wr(high_prices, low_prices, close_prices, window_size):
            """
            计算WR指标
            :param high_prices: 最高价列表
            :param low_prices: 最低价列表
            :param close_prices: 收盘价列表
            :param window_size: WR的窗口大小
            :return: WR列表
            """
            list_wr = []
            for i in range(len(close_prices)):
                if i < window_size - 1:
                    list_wr.append(0)  # 在最开始window_size个点没有WR值，填0
                else:
                    highest_high = max(high_prices[i - window_size + 1: i + 1])
                    lowest_low = min(low_prices[i - window_size + 1: i + 1])
                    if highest_high - lowest_low == 0:
                        wr = -100  # 避免除以零，如果最高价等于最低价
                    else:
                        wr = (highest_high - close_prices[i]) / (highest_high - lowest_low) * -100
                    list_wr.append(wr)
            return list_wr

        Log.info(f"开始计算WR数据")
        start_time = time.time()

        for gp_code, gp_data in bk_result.items():
            list_收盘价列表 = [value['收盘价'] for value in gp_data.values()]
            list_最高价列表 = [value['最高价'] for value in gp_data.values()]
            list_最低价列表 = [value['最低价'] for value in gp_data.values()]

            list_wr = calculate_wr(
                high_prices=list_最高价列表,
                low_prices=list_最低价列表,
                close_prices=list_收盘价列表,
                window_size=window_size
            )

            index = 0
            for timestamp, 所有股票当天数据字典 in gp_data.items():
                bk_result[gp_code][timestamp]['wr'] = list_wr[index]
                index += 1

        Log.debug(f"WR计算完毕，耗时{time.time() - start_time}秒")
        return bk_result

    @staticmethod
    def 生成KDJ数据(bk_result, window_size=9):
        """
        计算KDJ指标并添加到数据包
        :param bk_result: 数据包
        :param window_size: KDJ计算的窗口期
        :return: 包含KDJ数据的结果
        """

        def calculate_kdj(high_prices, low_prices, close_prices, window_size, k_period=3, d_period=3):
            """
            计算KDJ指标
            :param high_prices: 最高价列表
            :param low_prices: 最低价列表
            :param close_prices: 收盘价列表
            :param window_size: KDJ的窗口大小
            :param k_period: K值平滑周期，默认3天
            :param d_period: D值平滑周期，默认3天
            :return: K值、D值和J值列表
            """
            list_k = []
            list_d = []
            list_j = []

            rsv_list = []

            # 计算RSV（未成熟随机值）
            for i in range(len(close_prices)):
                if i < window_size - 1:
                    rsv_list.append(0)
                else:
                    highest_high = max(high_prices[i - window_size + 1: i + 1])
                    lowest_low = min(low_prices[i - window_size + 1: i + 1])
                    if highest_high - lowest_low == 0:
                        rsv = 50  # 避免除以零，设置默认RSV值
                    else:
                        rsv = (close_prices[i] - lowest_low) / (highest_high - lowest_low) * 100
                    rsv_list.append(rsv)

            # 初始化K和D的值
            k_value = 50  # 默认K值为50
            d_value = 50  # 默认D值为50

            for rsv in rsv_list:
                k_value = (2 / 3) * k_value + (1 / 3) * rsv  # K值平滑
                d_value = (2 / 3) * d_value + (1 / 3) * k_value  # D值平滑
                j_value = 3 * k_value - 2 * d_value  # J值计算

                list_k.append(k_value)
                list_d.append(d_value)
                list_j.append(j_value)

            return list_k, list_d, list_j

        Log.info(f"开始计算KDJ数据")
        start_time = time.time()

        for gp_code, gp_data in bk_result.items():
            list_收盘价列表 = [value['收盘价'] for value in gp_data.values()]
            list_最高价列表 = [value['最高价'] for value in gp_data.values()]
            list_最低价列表 = [value['最低价'] for value in gp_data.values()]

            list_k, list_d, list_j = calculate_kdj(
                high_prices=list_最高价列表,
                low_prices=list_最低价列表,
                close_prices=list_收盘价列表,
                window_size=window_size
            )

            index = 0
            for timestamp, 所有股票当天数据字典 in gp_data.items():
                bk_result[gp_code][timestamp]['kdj_k'] = list_k[index]
                bk_result[gp_code][timestamp]['kdj_d'] = list_d[index]
                bk_result[gp_code][timestamp]['kdj_j'] = list_j[index]
                index += 1

        Log.debug(f"KDJ计算完毕，耗时{time.time() - start_time}秒")
        return bk_result

    @staticmethod
    def 生成动量数据(bk_result, window_size=10):
        """
        计算动量数据并将其添加到数据集中
        :param bk_result: 包含股票数据的字典
        :param window_size: 动量计算的窗口大小，默认10天
        :return: 更新后的数据字典，包含动量数据
        """

        def calculate_momentum(list_close_price, window_size):
            """
            计算动量值
            :param list_close_price: 收盘价列表
            :param window_size: 动量的计算周期
            :return: 动量列表
            """
            momentum_list = []
            for index, close_price in enumerate(list_close_price):
                if index < window_size:
                    momentum_list.append(0)  # 如果不足window_size天，则动量设为0
                else:
                    momentum = close_price - list_close_price[index - window_size]
                    momentum_list.append(momentum)
            return momentum_list

        # 日志开始计算动量
        start_time = time.time()
        print(f"开始计算动量数据")

        # 遍历每个股票的收盘价并计算动量
        for gp_code, gp_data in bk_result.items():
            list_收盘价列表 = [value['收盘价'] for value in gp_data.values()]
            list_momentum = calculate_momentum(list_收盘价列表, window_size)

            # 将动量数据加入到原始数据集中
            index = 0
            for timestamp, 所有股票当天数据字典 in gp_data.items():
                bk_result[gp_code][timestamp]['momentum'] = list_momentum[index]
                index += 1

        # 记录计算时间
        print(f"动量数据计算完毕，耗时{time.time() - start_time}秒")
        return bk_result

    @staticmethod
    def 生成ATR数据(bk_result, window_size=14):
        """
        计算ATR数据并将其添加到数据集中
        :param bk_result: 包含股票数据的字典
        :param window_size: ATR计算的窗口大小，默认14天
        :return: 更新后的数据字典，包含ATR数据
        """

        def calculate_true_range(high, low, prev_close):
            """
            计算真实波动范围（TR）
            :param high: 当天最高价
            :param low: 当天最低价
            :param prev_close: 前一天的收盘价
            :return: 真实波动范围值
            """
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            return max(tr1, tr2, tr3)

        def calculate_atr(high_list, low_list, close_list, window_size):
            """
            计算ATR值
            :param high_list: 最高价列表
            :param low_list: 最低价列表
            :param close_list: 收盘价列表
            :param window_size: ATR计算的窗口大小
            :return: ATR列表
            """
            atr_list = []
            true_range_list = []

            # 计算每一天的真实波动范围（TR）
            for i in range(1, len(close_list)):
                true_range = calculate_true_range(high_list[i], low_list[i], close_list[i - 1])
                true_range_list.append(true_range)

            # 计算初始ATR（前window_size天的均值）
            initial_atr = np.mean(true_range_list[:window_size])
            atr_list.append(initial_atr)

            # 计算后续ATR，使用公式：
            # ATR[i] = (ATR[i-1] * (window_size - 1) + TR[i]) / window_size
            for i in range(window_size, len(true_range_list)):
                atr = (atr_list[-1] * (window_size - 1) + true_range_list[i]) / window_size
                atr_list.append(atr)

            # 如果天数少于窗口大小，补0
            atr_list = [0] * (window_size) + atr_list
            return atr_list

        # 日志开始计算ATR
        start_time = time.time()
        print(f"开始计算ATR数据")

        # 遍历每个股票的收盘价并计算ATR
        for gp_code, gp_data in bk_result.items():
            list_high = [value['最高价'] for value in gp_data.values()]
            list_low = [value['最低价'] for value in gp_data.values()]
            list_close = [value['收盘价'] for value in gp_data.values()]
            list_atr = calculate_atr(list_high, list_low, list_close, window_size)
            # print(f"{len(list_atr)} {len(list_high)}")
            # 将ATR数据加入到原始数据集中
            index = 0
            for timestamp, 所有股票当天数据字典 in gp_data.items():
                bk_result[gp_code][timestamp]['ATR'] = list_atr[index]
                index += 1

        # 记录计算时间
        print(f"ATR数据计算完毕，耗时{time.time() - start_time}秒")
        return bk_result

    @staticmethod
    def 生成DMA数据(bk_result, short_window=10, long_window=50):
        """
        计算DMA数据并将其添加到数据集中
        :param bk_result: 包含股票数据的字典
        :param short_window: 短期均线窗口大小，默认10天
        :param long_window: 长期均线窗口大小，默认50天
        :return: 更新后的数据字典，包含DMA数据
        """

        def calculate_moving_average(price_list, window_size):
            """
            计算移动平均线
            :param price_list: 收盘价列表
            :param window_size: 移动平均的窗口大小
            :return: 移动平均值列表
            """
            return np.convolve(price_list, np.ones(window_size) / window_size, 'valid').tolist()

        def calculate_dma(short_ma_list, long_ma_list):
            """
            计算DMA值
            :param short_ma_list: 短期移动均线列表
            :param long_ma_list: 长期移动均线列表
            :return: DMA值列表
            """
            dma_list = []
            # 长期均线会比短期均线更晚生成，所以要补齐前面的差值
            for i in range(len(long_ma_list)):
                dma = short_ma_list[i] - long_ma_list[i]
                dma_list.append(dma)
            return dma_list

        # 日志开始计算DMA
        start_time = time.time()
        print(f"开始计算DMA数据")

        # 遍历每个股票的收盘价并计算DMA
        for gp_code, gp_data in bk_result.items():
            list_收盘价 = [value['收盘价'] for value in gp_data.values()]

            # 计算短期和长期移动平均线
            short_ma = calculate_moving_average(list_收盘价, short_window)
            long_ma = calculate_moving_average(list_收盘价, long_window)

            # 确保短期均线和长期均线的长度对齐
            if len(short_ma) > len(long_ma):
                short_ma = short_ma[-len(long_ma):]
            else:
                long_ma = long_ma[-len(short_ma):]

            # 计算DMA
            list_dma = calculate_dma(short_ma, long_ma)

            # 填充原始数据集，DMA计算会产生少量前置空白数据
            index = 0
            dma_start_position = len(list_收盘价) - len(list_dma)
            for timestamp, 所有股票当天数据字典 in gp_data.items():
                if index >= dma_start_position:
                    所有股票当天数据字典['DMA'] = list_dma[index - dma_start_position]
                else:
                    所有股票当天数据字典['DMA'] = 0  # 在无法计算DMA时，用0填充
                index += 1

        # 记录计算时间
        print(f"DMA数据计算完毕，耗时{time.time() - start_time}秒")
        return bk_result

    @staticmethod
    def 生成RSI数据(bk_result, window_size=14):
        """
        计算RSI指标并添加到数据包
        :param bk_result: 数据包
        :param window_size: RSI计算的窗口期
        :return: 包含RSI数据的结果
        """

        def calculate_rsi(list_close_price, window_size):
            """
            计算RSI指标
            :param list_close_price: 收盘价列表
            :param window_size: RSI的窗口大小
            :return: RSI列表
            """
            list_rsi = []
            delta_prices = np.diff(list_close_price)  # 计算每日的价格变动
            gain = np.where(delta_prices > 0, delta_prices, 0)  # 上涨部分
            loss = np.where(delta_prices < 0, -delta_prices, 0)  # 下跌部分

            avg_gain = np.mean(gain[:window_size])
            avg_loss = np.mean(loss[:window_size])

            for index in range(len(list_close_price)):
                if index < window_size:
                    list_rsi.append(0)  # 前 window_size 个没有 RSI 值
                else:
                    if avg_loss == 0:
                        rs = 100  # 避免除以零
                    else:
                        rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    list_rsi.append(rsi)

                    # 更新平均涨幅和平均跌幅
                    if index < len(delta_prices):
                        avg_gain = (avg_gain * (window_size - 1) + gain[index]) / window_size
                        avg_loss = (avg_loss * (window_size - 1) + loss[index]) / window_size

            return list_rsi

        Log.info(f"开始计算RSI数据")
        start_time = time.time()

        for gp_code, gp_data in bk_result.items():
            list_收盘价列表 = [value['收盘价'] for value in gp_data.values()]
            list_rsi = calculate_rsi(
                list_close_price=list_收盘价列表,
                window_size=window_size
            )
            # bk.测试绘画RSI数据(list_收盘价列表, list_rsi)
            index = 0
            for timestamp, 所有股票当天数据字典 in gp_data.items():
                bk_result[gp_code][timestamp]['rsi'] = list_rsi[index]
                index += 1

        Log.debug(f"RSI计算完毕，耗时{time.time() - start_time}秒")
        return bk_result
