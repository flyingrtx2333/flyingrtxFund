import os
import json
import time
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Type, Dict
from config import BK_DATA_DIR
import traceback
import random
from utils.bk_manager import BKManager
from utils.Log import datetime_to_timestamp, colorized_string, timestamp_to_datetime, Log


def data_pre_solve(sim_data: dict, date_timestamp_now: float, window_size: int) -> dict:
    """
    数据预处理，包括截取*今日及今日之前*的数据，以及改为股票代码索引；
    将日期索引的模拟数据字典提取出来分成多个以gp_code为键的数据列表，合成一个字典
    :return: {股票1:[{day1},{day2}],股票2:[{day1},{day2}]}
    {'开盘价': 6.82,
    '收盘价': 6.68,
    '最高价': 6.82,
    '最低价': 6.6,
    '总手': 107056.0,
    '金额': 75107317.0,
    '振幅': 3.24,
    '涨跌幅': -1.47,
    '涨跌额': -0.1,
    '换手率': 0.36,
    'date_timestamp': 1698768000.0,
    'booling_lower': 5.748083649984651,
    'booling_mid': 6.1976666666666524,
    'booling_upper': 6.647249683348654
    }
    """
    dict_result: dict = defaultdict(dict)
    date_timestamps = list(sim_data.keys())
    end_position = date_timestamps.index(date_timestamp_now)
    start_position = end_position - window_size
    if start_position < 0:
        start_position = 0
    # print(f"起始位置{start_position}")
    for date_timestamp in date_timestamps[start_position:end_position + 1]:
        if date_timestamp > date_timestamp_now:
            break
        for gp_code, gp_klines_data in sim_data[date_timestamp].items():
            dict_result.setdefault(gp_code, []).append(gp_klines_data)
    # 股票出现顺序打乱
    gp_codes = list(dict_result.keys())
    random.shuffle(gp_codes)
    dict_result_shuffle = {gp_code: dict_result[gp_code] for gp_code in gp_codes}
    return dict_result_shuffle


class BacktestEngine:
    """
    2023/3/10:函数速度优化了12倍，约240秒->20秒以内
        multiprocessing支持多cpu核心真线程，可占用全核心资源高速运行
    """

    def __init__(
            self,
            bk_codes: list[str],
            initial_money: int,
            strategy,
            parameters,
            days_limit=1000,
            start_date='2020-03-22',
    ):
        """
        :param bk_indexs: 板块代号，如["BK0655", "BK0145"]
        :param money: 起始资金
        :param method: 选股、买入卖出方法
        :param parameters: 给方法用的参数
        :param window_size: 窗口大小
        :param days_limit: 模拟的天数
        :param start_date_time: 起始时间的字符串
        """
        super().__init__()
        self.bk_index = bk_codes
        my_dic_result = BKManager.load_multi_data_from_file(bk_indexs=bk_codes)
        self.sim_data: dict = BKManager.produce_simData(my_dic_result, start_date_time=start_date)
        self.date_timestamp_now = 0
        self.money: float = initial_money
        self.assets_now: float = initial_money
        self.method = strategy(parameters=parameters)
        self.inventory = defaultdict(dict)
        self.print_interval: int = 1
        self.list_assets: list = []
        self.list_money = []  # 记录一下活动资金变化趋势
        self.window_size: int = parameters['window_size']  # 为防止后期数据量过大，simData预处理只截取最近部分个数据
        self.parameters: dict = parameters
        self.days_limit = days_limit
        self.start_date_time = start_date
        self.start_date_time_act = start_date  # 实际开始操作的时间，在第一次交易时更新该数据
        self.start_timestamp = datetime_to_timestamp(start_date)
        self.操作记录 = []
        self.清仓次数: int = 0
        self.清仓盈利次数: int = 0
        self.data_after_pre_solve_all = data_pre_solve(
            sim_data=self.sim_data,
            date_timestamp_now=list(self.sim_data.keys())[-1],
            window_size=len(self.sim_data.keys()) - 1
        )

    def run(self):
        """
        开始运行
        :return:
        """
        start_time = time.time()
        self.process_main()
        print(f"计算结束，总共耗时{time.time() - start_time}秒")

        # 计算几个资产走势的评估指标
        sharpe_ratio = self.sharpeRatioCalculate(self.list_assets)
        maximum_drawdown = self.maximumDrawdownCalculate(self.list_assets)
        volatility = self.volatilityCalculate(self.list_assets)
        年化收益率 = self.annualized_return(
            start_date_str=self.start_date_time_act,
            end_date_str=timestamp_to_datetime(self.date_timestamp_now),
            start_value=self.list_assets[0],
            end_value=self.list_assets[-1]
        )
        if self.清仓次数 != 0:
            胜率 = self.清仓盈利次数 / self.清仓次数
        else:
            胜率 = 0
        胜率 = f"{round(胜率 * 100, 2)}%"
        # 写入日志
        self.writeLog(sharpe_ratio, maximum_drawdown, volatility, 胜率)
        self.advice()
        print(colorized_string(f"{self.method.__name__} {self.parameters}运算完成 "
                               f"夏普:{sharpe_ratio} 回撤:{maximum_drawdown} 波动:{volatility} 胜率:{胜率} 年化收益率：{年化收益率}",
                               'g'))

    def process_main(self):
        """
        主进程
        :return:
        """
        index = 0

        for date_timestamp, gps_data in self.sim_data.items():
            index += 1
            self.date_timestamp_now = date_timestamp
            print(colorized_string(timestamp_to_datetime(date_timestamp), 'p'))
            data_after_pre_solve = data_pre_solve(
                sim_data=self.sim_data,
                date_timestamp_now=date_timestamp,
                window_size=self.window_size
            )
            dict_can_buy, dict_can_sell, _ = self.method.run(
                data_after_pre_solve=data_after_pre_solve,
                inventory=self.inventory,
                date_timestamp_now=date_timestamp,
                money_now=self.money
            )
            # self.提前知道本次购买结果(dict_can_buy=dict_can_buy)
            self.buy(dict_can_buy=dict_can_buy)
            self.sell(dict_can_sell=dict_can_sell)
            self.assets_count()
            if index % self.print_interval == 0:
                print(f"时间来到了{colorized_string(timestamp_to_datetime(date_timestamp), 'p')}:"
                      f"当前总资产为：{colorized_string(int(self.assets_now), 'g')}")
                for key, value in self.inventory.items():
                    print(timestamp_to_datetime(date_timestamp), "持仓情况", key, value)
            if index > self.days_limit:
                break

    def buy(self, dict_can_buy: dict, logging_level: int = 2):
        """
        模拟执行购买
        :param logging_level: 日志等级
        :param dict_can_buy: 可买股票字典，键为个股代码
        :return:
        """
        for gp_code, value in dict_can_buy.items():
            price: float = value['price']
            num: int = value['num']
            if num <= 0:
                continue
            takes = max(5, int(100 * price * num * 0.00025)) + 100 * price * num
            if self.money > takes:
                if logging_level >= 2:
                    if self.操作记录 == []:
                        self.start_date_time_act = timestamp_to_datetime(self.date_timestamp_now)
                    print(colorized_string(
                        f"{timestamp_to_datetime(self.date_timestamp_now)} 购买{num}手{gp_code},价格{price},耗费{takes}",
                        'r'))
                    self.操作记录.append(f"购买{num}手{gp_code}，价格{price}, 耗费{takes}")

                # self.inventory[gp_code]['num'] += num
                # self.inventory[gp_code]['takes'] += takes
                # self.inventory[gp_code]['last'] = price
                if gp_code not in self.inventory:
                    self.inventory[gp_code] = {'num': num, 'takes': takes, 'last': price}  # 记录一下购买总花费，以后算成本用，应该可以出现负成本了
                else:
                    self.inventory[gp_code]['num'] += num
                    self.inventory[gp_code]['takes'] += takes
                    self.inventory[gp_code]['last'] = price
                """计算一下当前持仓成本"""
                self.inventory[gp_code]['持仓成本'] = round(takes / 100 / num, 2)
                self.inventory[gp_code]['最后购买日期'] = timestamp_to_datetime(self.date_timestamp_now)
                self.money -= takes
            # try.js:
            #     price = value['price']
            #     num = value['num']
            #     print(num)
            #     if num == None:
            #         continue
            #     # print(f"当前剩余{self.money}, 方法申请购买{num}手，耗费{price}")
            #     fee = max(5, 100 * price * num * 0.00025)
            #     takes = fee + 100 * price * num
            #     if self.money > takes:
            #         if gp_code not in self.inventory:
            #             self.inventory[gp_code] = {'num': num, 'takes': takes}  # 记录一下购买总花费，以后算成本用，应该可以出现负成本了
            #         else:
            #             self.inventory[gp_code]['num'] += num
            #             self.inventory[gp_code]['takes'] += takes
            #         self.money = self.money - takes
            # except TypeError:
            #     print(dict_can_buy)
            #     a = input('continue?')

    def sell(self, dict_can_sell: dict, loging_level: int = 2):
        """
        模拟卖出辣
        :param loging_level: 日志等级
        :param dict_can_sell: 可卖股票字典，键为个股代码
        :return:
        """

        for gp_code, value in dict_can_sell.items():
            price: float = value['price']
            持仓成本 = self.inventory[gp_code]['持仓成本']
            try:
                num: int = min(value['num'], self.inventory[gp_code]['num'])
            except Exception:
                traceback.print_exc()
                print(value['num'], self.inventory, gp_code, self.inventory[gp_code]['num'])
                a = input('continue?')
            increment: float = 100 * price * num - max(5, int(100 * price * num * 0.00025))
            self.inventory[gp_code]['num'] -= num
            self.inventory[gp_code]['takes'] -= increment

            self.money += increment
            if self.inventory[gp_code]['num'] == 0:
                """清仓了"""
                del self.inventory[gp_code]
                self.清仓次数 += 1
                if price > 持仓成本:
                    self.清仓盈利次数 += 1
                    print(colorized_string(f"卖出{num}手{gp_code},得到{increment},清仓后实现盈利啦！！！！", 'g'))
                else:
                    print(colorized_string(f"卖出{num}手{gp_code},得到{increment},清仓后亏本了呜呜呜", 'g'))
            if loging_level >= 2:
                print(colorized_string(f"卖出{num}手{gp_code},得到{increment}", 'g'))
                self.操作记录.append(f"卖出{num}手{gp_code},得到{increment}")

    def assets_count(self):
        """
        每日资产统计
        :return:
        """

        def 计算天数差(timestamp, date_str):

            # 将时间戳转换为日期时间对象
            timestamp_date = datetime.fromtimestamp(timestamp)

            # 将日期字符串转换为日期时间对象
            date_object = datetime.strptime(date_str, "%Y-%m-%d")

            # 计算天数差
            days_diff = (timestamp_date - date_object).days

            return days_diff

        inventory_value: float = 0
        for gp_code, value in self.inventory.items():
            num: int = value['num']
            try:
                price_today_close: float = self.sim_data[self.date_timestamp_now][gp_code]['收盘价']
                self.inventory[gp_code]['last'] = price_today_close
                self.inventory[gp_code]['持仓时长'] = 计算天数差(self.date_timestamp_now,
                                                                 self.inventory[gp_code]['最后购买日期'])
                self.inventory[gp_code]['盈亏比'] = round(price_today_close / self.inventory[gp_code]['持仓成本'] - 1,
                                                          2) * 100
            except KeyError:
                # 今天出错了，那只股票没有数据，按前一天算好了
                price_today_close: float = value['last']
            inventory_value += price_today_close * num * 100
        self.assets_now = self.money + inventory_value
        self.list_assets.append(self.assets_now)
        self.list_money.append(self.money)

    def advice(self):
        """
        给出现实参考意见
        直接用最后一天的日期时间戳去计算dict_can_buy和dict_can_sell
        :return:
        """
        for i in range(-5, 0):
            last_date_timestamp = list(self.sim_data.keys())[-1]
            data_after_pre_solve = data_pre_solve(
                sim_data=self.sim_data,
                date_timestamp_now=last_date_timestamp,
                window_size=self.window_size,

            )
            self.method.product_mode = True
            dict_can_buy, dict_can_sell, sim_data_after_chosen = self.method.run(
                data_after_pre_solve=data_after_pre_solve,
                inventory=self.inventory,
                date_timestamp_now=last_date_timestamp,
                money_now=99999
            )
            print(f"根据最新日期到{timestamp_to_datetime(last_date_timestamp)}的{self.bk_index}板块股票数据")
            if sim_data_after_chosen:
                for gp_code, value in sim_data_after_chosen.items():
                    print(
                        f"可在下一日，考虑以低于{timestamp_to_datetime(value[-1]['date_timestamp'])} {value[-1]['收盘价']}的价格,参考{round(self.parameters['buy_percent'] * value[-1]['收盘价'], 2)}元买入{gp_code}")
                # print(f"建议关注股票：{sim_data_after_chosen.keys()}")
            # if dict_can_buy:
            #     print(f"建议买入{colorized_string(dict_can_buy, 'r')}")
            else:
                print(f"没有建议买入的股票")

    @staticmethod  # 夏普率计算
    def sharpeRatioCalculate(asset_values: list) -> float:
        """
        最后计算总资产走势的夏普率
        :param asset_values:
        :return:
        """
        # 计算年化收益率
        returns = []
        for i in range(1, len(asset_values)):
            returns.append((asset_values[i] - asset_values[i - 1]) / asset_values[i - 1])

        # 计算平均年化收益率
        Rp = np.mean(returns) * 252  # 252个交易日为一年

        # 假设无风险利率为3%
        Rf = 0.03

        # 计算年化收益率的标准差
        op = np.std(returns, ddof=1) * np.sqrt(252)  # ddof参数为自由度，取1表示样本标准差

        # 计算夏普率
        Sharpe_Ratio = (Rp - Rf) / op

        return Sharpe_Ratio

    @staticmethod  # 最大回撤比率计算
    def maximumDrawdownCalculate(values: list):
        """
        计算最终总资产走势的最大回撤的比率
        :param values:
        :return:
        """
        max_drawdown = 0
        peak = values[0]
        for i in range(1, len(values)):
            if values[i] > peak:
                peak = values[i]
            else:
                drawdown = (peak - values[i]) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        return max_drawdown

    @staticmethod  # 波动率计算
    def volatilityCalculate(values):
        """
        计算总资产走势的波动率
        :param values:
        :return:
        """
        return np.std(values)

    @staticmethod  # 年化收益率计算
    def annualized_return(start_date_str, end_date_str, start_value, end_value):
        # 将日期字符串转换为datetime对象
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        # 计算天数差异
        days_diff = (end_date - start_date).days

        # 将天数转换为年限
        years_diff = days_diff / 365.25  # 使用365.25来考虑闰年
        Log.debug(f"开始计算年化收益率，起止日期：{start_date_str},{end_date_str},起止收益：{start_value},{end_value}")
        # 计算年化收益率
        annual_return = (end_value / start_value) ** (1 / years_diff) - 1

        return annual_return

    def writeLog(self, sharpe_ratio, maximum_drawdown, volatility, 胜率):
        """
        在运行完成后，写入最终的策略收益指标，记录某个方法的回测结果
        :param sharpe_ratio:
        :param maximum_drawdown:
        :param volatility:
        :param 胜率:
        :return:
        """
        with open('sim_result.log', 'a+', encoding='utf-8') as f:
            f.write(
                f"{datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}:{self.method.__name__} 参数{self.parameters}在{self.bk_index}上:"
                f"夏普:{sharpe_ratio} 历史最大回撤:{maximum_drawdown} 波动率:{volatility} 清仓次数:{self.清仓次数} 胜率:{胜率} \n")


if __name__ == '__main__':
    from utils.SimSystem import XJSTradingMethod3

    parameters = {
        'assets_once': 3000,
        'sell_percent': 1.03,
        'buy_percent': 0.99,
        '探测起始天数': 60,
        '探测最小值偏移比例': 1,
        'window_size': 200,
        '止损比例': -3,  # 百分点
        '止损时长': 90,  # 天
    }
    engine = BacktestEngine(
        bk_codes=['BK0153'],
        initial_capital=1000000,
        trading_method=XJSTradingMethod3,
        parameters=parameters,
        days_limit=5000,
        start_date='2020-05-23'
    )

    results = engine.run()
    print(results)
