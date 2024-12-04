"""
2023/3/10
"""
import random
from datetime import datetime
import threading
import traceback
import pandas as pd
# import xgboost as xgb
from utils.Log import datetime_to_timestamp, timestamp_to_datetime, Log, colorized_string
from utils.bk_manager import BKManager
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from typing import Type, TypedDict
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

"""交易方法父类"""


class XJSTradingMethod:
    """
    2023/3/10
    基于method2开发的基础框架，后续请重写此方法
    """

    def __init__(self, parameters, product_mode=False):
        super().__init__()
        self.assets_once = 3000
        self.product_mode = product_mode  # 可继承
        self.parameters = parameters
        self.window_size = parameters['window_size']  # 可继承，在子类init处直接定值

    def run(self, data_after_pre_solve, inventory, date_timestamp_now, money_now) -> [dict, dict]:
        """
        主运行方法，无需重写，使用时调用此方法
        耗时分析：
        chose耗时99%
        buy_check和sell_check几乎不耗时
        :param money_now: 引入该参数的目的是,当money过少时,跳过选股步骤,节省时间
        :return:可购买股票、可出售股票、选中的股票（该项主要用于现实建议）
        """
        self.data_after_pre_solve = data_after_pre_solve  # 可继承
        self.inventory_now = inventory  # 可继承
        self.date_timestamp_now = date_timestamp_now  # 可继承

        start_time = time.time()
        """chose耗时最大"""
        if money_now > self.assets_once:
            sim_data_after_chosen = self.chose(self.data_after_pre_solve, self.date_timestamp_now)

            Log.trace(f"chose耗时{time.time() - start_time}")
            # start_time = time.time()
            dict_can_buy = self.buy_check(sim_data_after_chosen, self.inventory_now)
        else:
            dict_can_buy = {}
            sim_data_after_chosen = {}
        # print(f"canbuy耗时{time.time()-start_time}")
        # start_time = time.time()
        dict_can_sell = self.sell_check(self.data_after_pre_solve, self.inventory_now)
        # print(f"sellcheck耗时{time.time()-start_time}")
        return dict_can_buy, dict_can_sell, sim_data_after_chosen

    def chose(self, data_after_pre_solve: dict, date_timestamp: float) -> dict:
        """
        这个方法设计成无需重写，重写部分chose_check
        由于传入的数据包括今天，而实际上选只能基于昨日及之前的数据来选，所以砍掉一天数据
        但如果需要指导现实，比如在今晚计算，实际上需要考虑今日的数据，请开启生产模式
        """
        sim_data_after_chosen = {}
        for gp_code, value_list in data_after_pre_solve.items():
            """value_list是多天数据列表，每个元素是一个字典"""
            date_timestamp_in_data = value_list[-1]['date_timestamp']
            if date_timestamp != date_timestamp_in_data:  # 时间对不上！！
                # print(colorized_string(f"{gp_code}时间对不上",'r'))
                continue
            if not self.product_mode:
                use_value_list = value_list[:-1]
            else:
                use_value_list = value_list
            if len(value_list) < self.window_size:
                continue
            if value_list[-1]['收盘价'] < 1 or value_list[-1]['收盘价'] > 100:
                # 鬼知道是什么东西
                continue
            # 好了，开始正式确定吧
            # 传给自选检查函数一个股票每日值的列表，应当返回真或假是否选择此股票
            if self.chose_check(use_value_list, gp_code=gp_code):
                sim_data_after_chosen[gp_code] = value_list
            if len(sim_data_after_chosen.keys()) >= 5:
                break
        return sim_data_after_chosen

    def chose_check(self, value_list: list, gp_code="") -> bool:
        value_list_split: list = value_list[-self.window_size:]
        list_x = np.arange(len(value_list_split))
        list_y = np.array([value['收盘价'] for value in value_list_split])
        # print(min_index)
        coef = np.polyfit(list_x, list_y, 2)
        y_fit = np.polyval(coef, list_x)
        min_index: int = y_fit.argmin()
        if coef[0] > 0 and min_index < self.探测最小值最大偏移量:
            return True

    def buy_check(self, sim_data_after_chosen: dict, inventory: dict) -> dict:
        """
        买入检查，判断是否可以买入
        :param sim_data_after_chosen: 被选中的股票
        :param inventory: 持仓信息
        :return:
        """
        阈值 = self.buy_percent
        dict_result = defaultdict(dict)
        if sim_data_after_chosen:
            for gp_code, value_list in sim_data_after_chosen.items():
                if len(value_list) <= 2:
                    continue
                if gp_code in inventory:  # 绝不加仓
                    continue
                open_today = value_list[-1]['开盘价']
                if value_list[-2]['收盘价'] < 1 or open_today > 100:  # 莫名其妙
                    continue

                low_today = value_list[-1]['最低价']
                close_yesterday = value_list[-2]['收盘价']

                buy_price = round(close_yesterday * 阈值, 2)  # 达到这个价格则有意愿购买，看看最低低过去没有，低过去说明可以买
                print(f"股票{gp_code} 今天最低价{low_today} 昨日收盘价{close_yesterday} 购买价格{buy_price}")
                if buy_price < 0:
                    continue
                if low_today <= buy_price:
                    dict_result[gp_code] = {'price': buy_price,
                                            'num': self.buy_num_decide(assets_once=self.assets_once, price=buy_price)}
        else:
            pass
        return dict_result

    def sell_check(self, sim_data_after_pre_solve: dict, inventory: dict) -> dict:
        dict_result = defaultdict(dict)
        if inventory:
            for gp_code, value in inventory.items():
                num: int = value['num']
                if num == 0:  # 没有持仓
                    continue
                takes: float = value['takes']
                price_cost: float = takes / num / 100
                try:
                    high_today: float = sim_data_after_pre_solve[gp_code][-1]['最高价']
                except KeyError:
                    # print(colorized_string(f"{gp_code}sell_check时间对不上", 'r'))
                    continue
                # low_today = sim_data_after_pre_solve[gp_code][-1]['最低价']
                sell_price = price_cost * self.sell_percent  # 意向卖出价
                # sell_bad_price = price_cost * self.割肉点
                if high_today >= sell_price:
                    # print(f"{gp_code}成本{price_cost},当前价{high_today},意向{sell_price}")
                    dict_result[gp_code] = {'price': sell_price,
                                            'num': self.sell_num_decide(self.assets_once, sell_price, num)}
                # if low_today <= sell_bad_price:
                #     dict_result[gp_code] = {'price': low_today, 'num': self.sell_num_decide(low_today, num)}
        else:
            pass
        return dict_result

    @staticmethod
    def sell_num_decide(assets_once: int, price: float, remain_num: int) -> int:
        if price < 0:
            return 0
        for i in range(1, 100):
            if i * 100 * price <= assets_once:
                continue
            else:
                return max(i, remain_num)
        return remain_num

    @staticmethod
    def buy_num_decide(assets_once: int, price: float) -> int:
        """
        决定要买多少手
        :param assets_once: 一次买多少钱左右，典型值 3000
        :param price: 一股价格，典型值 3.00
        :return: 购买的手数，典型值 10
        """
        if price < 0:
            return 0
        max_i: int = 0
        i: int = 1
        while i < 1000 and i * 100 * price <= assets_once:
            max_i = i
            i += 1
        return max_i


class XJSTradingMethod1(XJSTradingMethod):
    """
    方法1
    """

    def __init__(self, data_after_pre_solve, inventory, date_timestamp_now, parameters, window_size=20,
                 product_mode=False):
        super().__init__(data_after_pre_solve, inventory, date_timestamp_now, product_mode, window_size)
        self.buy_percent = parameters['buy_percent']
        self.sell_percent = parameters['sell_percent']
        self.连续下跌天数 = parameters['连续下跌天数']
        self.震荡检查天数 = parameters['震荡检查天数']
        self.震荡幅度限制 = parameters['震荡幅度限制']

    def chose_check(self, value_list: list) -> bool:
        """
        自选检查，必须要通过连续下跌和震荡幅度在一定范围内两个条件的筛选
        :param value_list:股票值列表
        :return:True or False
        """
        pass_连续下跌 = True
        for i in range(1, self.连续下跌天数):
            last_value_1 = value_list[-i]['收盘价']
            last_value_2 = value_list[-i - 1]['收盘价']
            if last_value_1 <= last_value_2:
                continue
            else:
                pass_连续下跌 = False
                break
        if not pass_连续下跌:
            return False
        pass_震荡检查 = True
        for i in range(1, self.震荡检查天数):
            振幅 = value_list[-i]['振幅']  # 百分比数据
            if 振幅 < self.震荡幅度限制:
                continue
            else:
                pass_震荡检查 = False
        if not pass_震荡检查:
            return False
        return True

    def buy_check(self, sim_data_after_chosen: dict, inventory: dict) -> dict:
        阈值 = 1 - self.震荡幅度限制 * 0.01
        dict_result = defaultdict(dict)
        if sim_data_after_chosen:
            for gp_code, value_list in sim_data_after_chosen.items():
                if len(value_list) <= 2:
                    continue
                if gp_code in inventory:  # 绝不加仓
                    continue
                open_today = value_list[-1]['开盘价']
                if open_today > 100:
                    continue
                low_today = value_list[-1]['最低价']
                close_yesterday = value_list[-2]['收盘价']
                buy_price_1 = round(close_yesterday * 阈值, 2)  # 达到这个价格则有意愿购买，看看最低低过去没有，低过去说明可以买
                if buy_price_1 < 0:
                    continue
                if low_today <= buy_price_1:
                    dict_result[gp_code] = {'price': buy_price_1,
                                            'num': self.buy_num_decide(self.assets_once, buy_price_1)}
        else:
            pass
        return dict_result

    def sell_check(self, sim_data_after_pre_solve: dict, inventory: dict):
        dict_result = defaultdict(dict)
        if inventory:
            for gp_code, value in inventory.items():
                num = value['num']
                if num == 0:
                    continue
                takes = value['takes']
                price_cost = takes / num / 100
                try:
                    high_today: float = sim_data_after_pre_solve[gp_code][-1]['最高价']
                except KeyError:
                    # print(colorized_string(f"{gp_code}sell_check时间对不上", 'r'))
                    continue
                sell_price = price_cost * self.sell_percent  # 意向卖出价
                if high_today >= sell_price:
                    # print(f"{gp_code}成本{price_cost},当前价{high_today},意向{sell_price}")
                    dict_result[gp_code] = {'price': sell_price,
                                            'num': self.sell_num_decide(self.assets_once, sell_price, num)}
        else:
            pass
        return dict_result


class XJSTradingMethod2(XJSTradingMethod):
    """
    自选方法：二次拟合，开口向上者
    购买方法：价格低过开盘一定比例之后
    """

    def __init__(self, data_after_pre_solve, inventory, date_timestamp_now, parameters, window_size=60,
                 product_mode=False):
        """
        参考值：
        sell_percent = 1.02
        buy_percent = 0.98
        起始天数 = 30
        最大偏移量 = 0.6（极值点位置）
        """
        super().__init__(data_after_pre_solve, inventory, date_timestamp_now, product_mode, window_size)
        self.sell_percent = parameters[0]
        self.buy_percent = parameters[1]
        self.探测起始天数 = parameters[2]
        self.探测最小值最大偏移量 = parameters[3] * self.探测起始天数

    def chose_check(self, value_list: list):
        value_list_split: list = value_list[-self.window_size:]
        list_x = np.arange(len(value_list_split))
        list_y = np.array([value['收盘价'] for value in value_list_split])
        coef = np.polyfit(list_x, list_y, 2)
        y_fit = np.polyval(coef, list_x)
        min_index: int = y_fit.argmin()
        if coef[0] > 0 and min_index < self.探测最小值最大偏移量:
            return True

    def buy_check(self, sim_data_after_chosen: dict, inventory: dict) -> dict:
        阈值 = self.buy_percent
        dict_result = defaultdict(dict)
        if sim_data_after_chosen:
            for gp_code, value_list in sim_data_after_chosen.items():
                if len(value_list) <= 2:
                    continue
                if gp_code in inventory:  # 绝不加仓
                    continue
                open_today = value_list[-1]['开盘价']
                if value_list[-2]['收盘价'] < 1 or open_today > 100:  # 莫名其妙
                    continue

                low_today = value_list[-1]['最低价']
                close_yesterday = value_list[-2]['收盘价']
                buy_price_1 = round(close_yesterday * 阈值, 2)  # 达到这个价格则有意愿购买，看看最低低过去没有，低过去说明可以买
                if buy_price_1 < 0:
                    continue
                if low_today <= buy_price_1:
                    dict_result[gp_code] = {'price': buy_price_1,
                                            'num': self.buy_num_decide(assets_once=self.assets_once, price=buy_price_1)}
        else:
            pass
        return dict_result

    def sell_check(self, sim_data_after_pre_solve: dict, inventory: dict) -> dict:
        dict_result = defaultdict(dict)
        if inventory:
            for gp_code, value in inventory.items():
                num: int = value['num']
                if num == 0:  # 没有持仓
                    continue
                takes: float = value['takes']
                price_cost: float = takes / num / 100
                try:
                    high_today: float = sim_data_after_pre_solve[gp_code][-1]['最高价']
                except KeyError:
                    # print(colorized_string(f"{gp_code}sell_check时间对不上", 'r'))
                    continue
                # low_today = sim_data_after_pre_solve[gp_code][-1]['最低价']
                sell_price = price_cost * self.sell_percent  # 意向卖出价
                # sell_bad_price = price_cost * self.割肉点
                if high_today >= sell_price:
                    # print(f"{gp_code}成本{price_cost},当前价{high_today},意向{sell_price}")
                    dict_result[gp_code] = {'price': sell_price,
                                            'num': self.sell_num_decide(self.assets_once, sell_price, num)}
                # if low_today <= sell_bad_price:
                #     dict_result[gp_code] = {'price': low_today, 'num': self.sell_num_decide(low_today, num)}
        else:
            pass
        return dict_result


class XJSTradingMethod3(XJSTradingMethod):
    """
    自选方法1：二次拟合，开口向上者
    自选方法2:连续下跌
    购买方法：价格低过开盘一定比例之后
    """

    def __init__(self, parameters: dict):
        """
        参考值：
        sell_percent = 1.02
        buy_percent = 0.98
        起始天数 = 30
        最大偏移量 = 0.6（极值点位置）
        连续下跌天数 = 3
        """
        super().__init__(parameters)
        self.assets_once = parameters.get('assets_once', 10000)
        self.sell_percent = parameters.get('sell_percent', 1.1)
        self.buy_percent = parameters.get('buy_percent', 0.97)
        self.探测起始天数 = parameters['探测起始天数']
        self.探测最小值最大偏移量 = parameters['探测最小值偏移比例'] * self.探测起始天数
        self.止损比例 = parameters['止损比例']
        self.止损时长 = parameters['止损时长']
        # self.连续下跌天数 = parameters['连续下跌天数']
        self.满足正弦匹配的股票 = {
            # 'gp_code':{'振幅':"","偏移":"","频率":"","相位":""}
        }
        self.正弦匹配失败后冷却时间 = 15
        self.不满足正弦匹配但近期尝试过匹配的股票 = {
            # "gp_code": self.正弦匹配失败后冷却时间  # 冷却，一次-1
        }
        self.可视化结果 = False

        self.__name__ = "方法3 - 常规指标判别"

    def 拟合_二次函数(self, list_x, list_y):
        coef = np.polyfit(list_x, list_y, 2)
        y_fit = np.polyval(coef, list_x)
        min_index: int = y_fit.argmin()
        if coef[0] > 0 and min_index < self.探测最小值最大偏移量 and y_fit[-1] < y_fit[0]:
            # plt.plot(list_y)
            # plt.plot(y_fit)
            # plt.show()
            return True

    def chose_check(self, value_list: list, gp_code=""):
        """
        选股方法，检查是否符合正弦分布，若符合，则判断当前是否为底部
        :param value_list:
        :param gp_code:
        :return:
        """
        """为避免多次拟合不符合的股票，给出一个冷却期，如果近期尝试匹配过但不符合，则冷却一段时间不计算该股票"""
        if gp_code in self.不满足正弦匹配但近期尝试过匹配的股票:
            self.不满足正弦匹配但近期尝试过匹配的股票[gp_code] = self.不满足正弦匹配但近期尝试过匹配的股票[gp_code] - 1
            if self.不满足正弦匹配但近期尝试过匹配的股票[gp_code] == 0:
                del self.不满足正弦匹配但近期尝试过匹配的股票[gp_code]
            return False
        value_list_split: list = value_list[-self.探测起始天数:]  # 例如获取最后7天数据

        list_y = np.array([value['收盘价'] for value in value_list_split])  # 取收盘价做判断
        # 符合结果,r2,omega,phi, _, _ = 因子.拟合_正弦函数(list_y=list_y)
        # 符合结果 = 因子.布林经典(list_value=value_list)
        # 符合结果 = 因子.布林上穿中线(list_value=value_list)
        # 符合结果 = 因子.布林上穿底部(list_value=value_list)
        macd_histogram = value_list[-1]['macd_histogram']
        符合结果 = (
            True
            # and 因子.布林下穿底部(list_value=value_list)
            and 因子.布林上穿中线(list_value=value_list)
            # and 因子.EMA金叉(list_value=value_list)
            # and 因子.布林近期触底(list_value=value_list)
            # and 因子.换手率(list_value=value_list)
            and 因子.EMA200上涨(list_value=value_list)
            # and macd_histogram > 0
            # and not 因子.EMA死叉_近期出现(list_value=value_list)
        )
        if not 符合结果:
            self.不满足正弦匹配但近期尝试过匹配的股票[gp_code] = self.正弦匹配失败后冷却时间
        return 符合结果

    def buy_check(self, sim_data_after_chosen: dict, inventory: dict) -> dict:
        threshold = self.buy_percent
        dict_result = defaultdict(dict)
        if sim_data_after_chosen:
            for gp_code, value_list in sim_data_after_chosen.items():
                if len(value_list) <= 2:
                    continue
                if gp_code in inventory:  # 绝不加仓
                    continue
                open_today = value_list[-1]['开盘价']
                if value_list[-2]['收盘价'] < 1 or open_today > 100:  # 莫名其妙
                    continue

                low_today = value_list[-1]['最低价']
                close_yesterday = value_list[-2]['收盘价']
                buy_price_1 = round(open_today * threshold, 2)  # 达到这个价格则有意愿购买，看看最低低过去没有，低过去说明可以买
                # buy_price_1 = open_today
                if buy_price_1 < 0:
                    continue
                if low_today <= buy_price_1:
                    # print(f"股票{gp_code} 今天最低价{low_today} 昨日收盘价{close_yesterday} 购买价格{buy_price_1}")

                    dict_result[gp_code] = {'price': buy_price_1,
                                            'num': self.buy_num_decide(assets_once=self.assets_once, price=buy_price_1)}
        else:
            pass
        return dict_result

    def sell_check(self, sim_data_after_pre_solve: dict, inventory: dict) -> dict:

        dict_result = defaultdict(dict)
        if inventory:
            for gp_code, value in inventory.items():
                num: int = value['num']
                if num == 0:  # 没有持仓
                    continue
                takes: float = value['takes']
                price_cost: float = takes / num / 100
                try:
                    high_today: float = sim_data_after_pre_solve[gp_code][-1]['最高价']
                    open_today: float = sim_data_after_pre_solve[gp_code][-1]['开盘价']
                except KeyError:
                    # print(colorized_string(f"{gp_code}sell_check时间对不上", 'r'))
                    continue
                # low_today = sim_data_after_pre_solve[gp_code][-1]['最低价']
                止盈比例 = 动态止盈点(
                    origin_percent=self.sell_percent,
                    持仓时长=value['持仓时长']
                )
                止盈比例 = self.sell_percent
                sell_price = price_cost * 止盈比例  # 意向卖出价
                # sell_bad_price = price_cost * self.割肉点
                if high_today >= sell_price:
                    # print(f"{gp_code}成本{price_cost},当前价{high_today},意向{sell_price}")
                    dict_result[gp_code] = {'price': sell_price,
                                            'num': self.sell_num_decide(self.assets_once, sell_price, num)}
                if value['盈亏比'] < self.止损比例:
                    print(f"根据{self.__name__}止损策略应当止损{gp_code}")
                    if gp_code in dict_result:
                        print(f"止损时，当天其实高点也可赚钱")
                    dict_result[gp_code] = {'price': open_today,
                                            'num': self.sell_num_decide(self.assets_once, sell_price, num)}
                # if 因子.EMA死叉(list_value=sim_data_after_pre_solve[gp_code][:-1]):
                #     print(f"{gp_code} EMA死叉")
                #     dict_result[gp_code] = {'price': open_today,
                #                             'num': self.sell_num_decide(self.assets_once, sell_price, num)}
                # 天数差 = 计算天数差(timestamp=)
                # if low_today <= sell_bad_price:
                #     dict_result[gp_code] = {'price': low_today, 'num': self.sell_num_decide(low_today, num)}
        else:
            pass
        return dict_result


class XJSTradingMethod4(XJSTradingMethod):
    """
    自选方法4:
    """

    def __init__(self, data_after_pre_solve, inventory, date_timestamp_now, parameters: dict, window_size=60,
                 product_mode=False):
        """
        参考值：
        sell_percent = 1.02
        buy_percent = 0.98
        起始天数 = 30
        最大偏移量 = 0.6（极值点位置）
        连续下跌天数 = 3
        """
        super().__init__(data_after_pre_solve, inventory, date_timestamp_now, product_mode, window_size)
        self.method_name = "阿巴"
        self.sell_percent = parameters['sell_percent']
        self.buy_percent = parameters['buy_percent']
        self.window_size = parameters['window_size']
        self.boolling_band_width = parameters['boolling_band_width']
        self.割肉比例 = 0.8

    def chose_check(self, value_list: list):
        """
        检查是否符合自选规则
        :param value_list: 该个股的数据列表，列表套字典如：[{'收','开'},{'收','开'},...]
        :return:
        """
        """裁剪出部分数据"""
        value_list_split: list = value_list[-self.window_size:]
        date_timestamp_last = value_list_split[-1]['date_timestamp']
        """开始计算布林"""
        closing_prices = [data_point['收盘价'] for data_point in value_list_split]
        lowest_prices = [data_point['最低价'] for data_point in value_list_split]
        list_upper_band = [data_point['booling_upper'] for data_point in value_list_split]
        list_mid_band = [data_point['booling_mid'] for data_point in value_list_split]
        list_lower_band = [data_point['booling_lower'] for data_point in value_list_split]

        list_x = range(len(closing_prices))
        # # list_upper_band_20, list_mid_band_20, list_lower_band_20 = calculate_boolling_band(
        # #     list_close_price=closing_prices,
        # #     window_size=20
        # # )
        # # list_upper_band_140, list_mid_band_140, list_lower_band_140 = calculate_boolling_band(
        # #     list_close_price=closing_prices,
        # #     window_size=140
        # # )
        # # list_upper_band_80, list_mid_band_80, list_lower_band_80 = calculate_boolling_band(
        # #     list_close_price=closing_prices,
        # #     window_size=80
        # # )
        # list_upper_band, list_mid_band, list_lower_band = calculate_boolling_band(
        #     list_close_price=closing_prices,
        #     window_size=30
        # )
        '''如果前一天下摸布林下轨，就开始注意，今天就可以买，所以返回真'''
        if lowest_prices[-2] <= list_lower_band[-2]:
            # plt.plot(list_x, closing_prices, label='close_price')
            # plt.plot(list_x, list_lower_band)
            # plt.plot(list_x, list_mid_band)
            # plt.plot(list_x, list_upper_band)
            # plt.title(timestamp_to_datetime(date_timestamp_last))
            # plt.show()
            # a = input('continue?')
            return True

    def buy_check(self, sim_data_after_chosen: dict, inventory: dict) -> dict:
        threshold = self.buy_percent
        dict_result = defaultdict(dict)
        if sim_data_after_chosen:
            for gp_code, value_list in sim_data_after_chosen.items():
                # Log.debug(f"{gp_code}")
                if len(value_list) <= 2:
                    continue
                if gp_code in inventory:  # 绝不加仓
                    continue
                open_today = value_list[-1]['开盘价']
                if value_list[-2]['收盘价'] < 1 or open_today > 100:  # 莫名其妙
                    continue

                low_today = value_list[-1]['最低价']
                close_yesterday = value_list[-2]['收盘价']
                buy_price_1 = round(close_yesterday * threshold, 2)  # 达到这个价格则有意愿购买，看看最低低过去没有，低过去说明可以买
                if buy_price_1 < 0:
                    continue
                if low_today <= buy_price_1:
                    dict_result[gp_code] = {'price': buy_price_1,
                                            'num': self.buy_num_decide(assets_once=self.assets_once, price=buy_price_1)}
        else:
            pass
        return dict_result

    def sell_check(self, sim_data_after_pre_solve: dict, inventory: dict) -> dict:
        """

        :param sim_data_after_pre_solve:
        :param inventory:
        :return: dict_result[gp_code] = {
                'price': sell_price,
                'num': self.sell_num_decide(self.assets_once, sell_price, num)
            }
        """
        割肉比例 = self.割肉比例
        dict_result = defaultdict(dict)
        if inventory:
            for gp_code, value in inventory.items():
                num: int = value['num']
                持仓成本 = value['持仓成本']
                if num == 0:  # 没有持仓
                    continue
                takes: float = value['takes']
                price_cost: float = takes / num / 100
                try:
                    high_today: float = sim_data_after_pre_solve[gp_code][-1]['最高价']
                    close_today: float = sim_data_after_pre_solve[gp_code][-1]['收盘价']
                except KeyError:
                    # print(colorized_string(f"{gp_code}sell_check时间对不上", 'r'))
                    continue
                # low_today = sim_data_after_pre_solve[gp_code][-1]['最低价']
                sell_price = price_cost * self.sell_percent  # 意向卖出价
                # sell_bad_price = price_cost * self.割肉点
                if high_today >= sell_price:
                    # print(f"{gp_code}成本{price_cost},当前价{high_today},意向{sell_price}")
                    dict_result[gp_code] = {'price': sell_price,
                                            'num': self.sell_num_decide(self.assets_once, sell_price, num)}
                elif close_today <= 割肉比例 * 持仓成本:
                    Log.warning(f"*!!!割肉{gp_code}!!!*", 'y')
                    sell_price = close_today
                    dict_result[gp_code] = {'price': sell_price,
                                            'num': self.sell_num_decide(self.assets_once, sell_price, num)}
            # if low_today <= sell_bad_price:
            #     dict_result[gp_code] = {'price': low_today, 'num': self.sell_num_decide(low_today, num)}
        else:
            pass
        return dict_result


class XJSTradingMethod5(XJSTradingMethod):
    """
    自选方法5
    机器学习
    """

    def __init__(self, parameters: dict):
        """
        参考值：
        sell_percent = 1.02
        buy_percent = 0.98
        起始天数 = 30
        最大偏移量 = 0.6（极值点位置）
        连续下跌天数 = 3
        """

        super().__init__(parameters)
        self.assets_once = parameters.get('assets_once', 10000)
        self.sell_percent = parameters.get('sell_percent', 1.1)
        self.buy_percent = parameters.get('buy_percent', 0.97)
        self.止损比例 = parameters['止损比例']
        self.止损时长 = parameters['止损时长']
        self.model_判涨 = xgb.Booster()
        self.model_判涨.load_model('machine_learning_models/xgboost_up_model.json')
        # self.model_判跌 = xgb.Booster()
        # self.model_判跌.load_model('machine_learning_models/xgboost_down_model.json')
        self.__name__ = "方法5 - 机器学习策略"

    def chose_check(self, value_list: list, gp_code=""):
        """
        机器学习配合其他因子
        :param value_list:一个包含该股票多天值的列表，每个值是一个dict，里面包含'开盘价'、'收盘价'、'最高价'、'最低价'、'总手'、'涨跌幅'、'换手率'等
        :param gp_code:股票代号
        :return:
        """
        if not value_list:
            return False
        因子_判涨_机器学习, 判涨概率 = 因子.机器学习因子(value_list=value_list, model=self.model_判涨)
        因子_EMA200上涨 = 因子.EMA200上涨(list_value=value_list)
        # 因子_判跌_机器学习, 判跌概率 = 因子.机器学习因子(value_list=value_list, model=self.model_判跌)
        因子_机器学习 = 因子_判涨_机器学习
        if 因子_机器学习:
            return True
        else:
            return False

    def buy_check(self, sim_data_after_chosen: dict, inventory: dict) -> dict:
        threshold = self.buy_percent
        dict_result = defaultdict(dict)
        if sim_data_after_chosen:
            for gp_code, value_list in sim_data_after_chosen.items():
                if len(value_list) <= 2:
                    continue
                if gp_code in inventory:  # 绝不加仓
                    continue
                open_today = value_list[-1]['开盘价']
                if value_list[-2]['收盘价'] < 1 or open_today > 100:  # 莫名其妙
                    continue

                low_today = value_list[-1]['最低价']
                close_yesterday = value_list[-2]['收盘价']
                buy_price_1 = round(open_today * threshold, 2)  # 达到这个价格则有意愿购买，看看最低低过去没有，低过去说明可以买
                # buy_price_1 = open_today
                if buy_price_1 < 0:
                    continue
                if low_today <= buy_price_1:
                    dict_result[gp_code] = {'price': buy_price_1,
                                            'num': self.buy_num_decide(assets_once=self.assets_once, price=buy_price_1)}
        else:
            pass
        return dict_result

    def sell_check(self, sim_data_after_pre_solve: dict, inventory: dict) -> dict:
        dict_result = defaultdict(dict)
        if inventory:
            for gp_code, value in inventory.items():
                num: int = value['num']
                if num == 0:  # 没有持仓
                    continue
                takes: float = value['takes']
                price_cost: float = takes / num / 100
                try:
                    high_today: float = sim_data_after_pre_solve[gp_code][-1]['最高价']
                    open_today: float = sim_data_after_pre_solve[gp_code][-1]['开盘价']
                except KeyError:
                    # print(colorized_string(f"{gp_code}sell_check时间对不上", 'r'))
                    continue
                # low_today = sim_data_after_pre_solve[gp_code][-1]['最低价']
                sell_price = price_cost * self.sell_percent  # 意向卖出价
                # sell_bad_price = price_cost * self.割肉点
                if high_today >= sell_price:
                    # print(f"{gp_code}成本{price_cost},当前价{high_today},意向{sell_price}")
                    dict_result[gp_code] = {'price': sell_price,
                                            'num': self.sell_num_decide(self.assets_once, sell_price, num)}
                if value['盈亏比'] < self.止损比例:
                    print(f"根据{self.__name__}止损策略应当止损{gp_code}")
                    dict_result[gp_code] = {'price': open_today,
                                            'num': self.sell_num_decide(self.assets_once, sell_price, num)}
                # 天数差 = 计算天数差(timestamp=)
                # if low_today <= sell_bad_price:
                #     dict_result[gp_code] = {'price': low_today, 'num': self.sell_num_decide(low_today, num)}
        else:
            pass
        return dict_result


def 动态止盈点(origin_percent, 持仓时长):
    """
    我希望随持仓时间增加,预期止盈点下降
    例如原本10%止盈,每持仓增加10天,预期盈利点下降1%
    :param origin_percent:1.1
    :param 持仓时长:200
    :return:
    """
    new_percent = origin_percent - 0.01 * (持仓时长 % 20)
    if new_percent <= 1.02:
        new_percent = 1.02
    return new_percent


class 因子:
    """
    所有因子都放在这里吧
    """

    @staticmethod
    def 拟合_正弦函数(list_y, 拟合度要求=0.6, 可视化结果=False):
        """
        尝试用正弦函数拟合股票，当目前趋势为波谷姿态时返回真
        :param list_y: 收盘价列表
        :param 拟合度要求:
        :return:是否符合,r2,omega,phi, period, A
        """

        def sinusoidal(x, A, omega, phi, offset):
            """正弦函数模型"""
            return A * np.sin(omega * x + phi) + offset

        def r_squared(y_true, y_pred):
            """计算决定系数 R²"""
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)

        def is_trough_at_end(y_fit):
            """检查拟合曲线末尾是否为波谷"""
            first_derivative = np.diff(y_fit)
            last_slope = first_derivative[-1]
            is_trough = last_slope > 0  # 斜率大于0
            return is_trough

        def 拟合结果可视化(gp_code='', phi='unknown'):
            plt.title(f"{gp_code} 相移{phi}")
            plt.plot(list_x, list_y, label='原始数据')
            plt.plot(list_x, y_fit, label='拟合正弦函数')
            plt.legend()
            plt.show()
            a = input('')

        # 初始猜测参数
        n = len(list_y)
        list_x = np.arange(n)  # 只做坐标轴，取个长度就行
        guess_amplitude = (np.max(list_y) - np.min(list_y)) / 2
        guess_offset = np.mean(list_y)
        guess_frequency = 2 * np.pi / n
        guess_phase = 0
        guess = [guess_amplitude, guess_frequency, guess_phase, guess_offset]

        # 拟合正弦函数
        try:
            # 拟合正弦函数
            popt, _ = curve_fit(sinusoidal, list_x, list_y, p0=guess, maxfev=3000)

            # 提取拟合参数
            A, omega, phi, offset = popt

            if phi >= 0:
                raise Exception("相位不符合要求")

            # 计算拟合曲线
            y_fit = sinusoidal(list_x, A, omega, phi, offset)

            # 计算拟合度 R²
            r2 = r_squared(list_y, y_fit)

            if r2 < 拟合度要求:
                raise Exception("拟合度不符合要求")

            # 检查拟合曲线末尾是否为波谷
            at_trough = is_trough_at_end(y_fit)

            # 检查周期，周期 T = 2π / ω
            period = 2 * np.pi / omega
            is_complete_cycle = period <= n

            # 检查周期，周期 T = 2π / ω
            period = 2 * np.pi / omega
            is_two_complete_cycles = period * 2 <= n

            if r2 >= 拟合度要求 and at_trough:

                # 输出拟合结果
                # print(f"振幅: {A}, 频率: {omega}, 相位偏移: {phi}, 垂直偏移: {offset} 拟合度 R²: {r2}")
                # print(f"周期: {period}, 至少包含一个完整周期: {is_complete_cycle}")

                # 可视化拟合结果
                if 可视化结果:
                    拟合结果可视化(phi=phi)

                return True, r2, omega, phi, period, A
            else:
                return False, r2, omega, phi, period, A
        except Exception as e:
            return False, None, None, None, None, None

    @staticmethod
    def wztong():
        a = random.random()
        if a>=0.5:
            return True

    @staticmethod
    def 布林经典(list_value):
        value_today = list_value[-1]
        value_yesterday = list_value[-2]
        booling_lower = value_today['booling_lower']
        booling_mid = value_today['booling_mid']
        booling_upper = value_today['booling_upper']
        if value_today['收盘价'] > value_today['booling_upper'] and value_yesterday['收盘价'] < value_yesterday[
            'booling_upper']:
            return True

    @staticmethod
    def 布林上穿中线(list_value):
        value_today = list_value[-1]
        value_yesterday = list_value[-2]
        if value_today['收盘价'] > value_today['booling_mid'] and value_yesterday['收盘价'] < value_yesterday[
            'booling_mid']:
            return True
        else:
            return False

    @staticmethod
    def 布林上穿底部(list_value):
        value_today = list_value[-1]
        value_yesterday = list_value[-2]
        if value_today['收盘价'] > value_today['booling_lower'] and value_yesterday['收盘价'] < value_yesterday[
            'booling_lower']:
            return True
        else:
            return False

    @staticmethod
    def 布林下穿底部(list_value):
        """
        股价下跌跌破布林带
        :param list_value:
        :return:
        """
        value_today = list_value[-1]
        value_yesterday = list_value[-2]
        if value_today['收盘价'] <= value_today['booling_lower'] and value_yesterday['收盘价'] > value_yesterday[
            'booling_lower']:
            return True
        else:
            return False

    @staticmethod
    def 布林近期触底(list_value, window=7):
        result = False
        start_pos = len(list_value) - window
        for i in range(start_pos, len(list_value)):
            value_today = list_value[i]
            if value_today['最低价'] <= value_today['booling_lower']:
                result = True
        return result

    @staticmethod
    def RSI过低(list_value, 阈值=30):
        if list_value[-1]['rsi'] < 阈值:
            return True
        else:
            return False

    @staticmethod
    def EMA金叉(list_value):
        """
        判别股价短期趋势
        :param list_value:
        :return:
        """
        if list_value[-2]['short_ema'] < list_value[-2]['long_ema'] and list_value[-1]['short_ema'] >= list_value[-1][
            'long_ema']:
            return True
        else:
            return False

    @staticmethod
    def EMA死叉(list_value):
        """
        判别股价短期趋势
        :param list_value:
        :return:
        """
        if list_value[-2]['short_ema'] >= list_value[-2]['long_ema'] and list_value[-1]['short_ema'] < list_value[-1][
            'long_ema']:
            return True
        else:
            return False

    @staticmethod
    def EMA死叉_近期出现(list_value):
        """
        如果近期出现EMA死叉
        :param list_value:
        :return:
        """
        start_pos = len(list_value) - 30
        for i in range(start_pos, len(list_value)):
            if list_value[i - 1]['short_ema'] >= list_value[i - 1]['long_ema'] and list_value[i]['short_ema'] < \
                    list_value[i]['long_ema']:
                return True

    @staticmethod
    def KDJ金叉(list_value):
        """
        K线上穿D线
        :param list_value:
        :return:
        """
        if list_value[-2]['kdj_k'] < list_value[-2]['kdj_d'] and list_value[-1]['kdj_k'] >= list_value[-1]['kdj_d']:
            return True

    @staticmethod
    def EMA200上涨(list_value):
        """
        判别股价长期趋势
        :param list_value:
        :return:
        """
        # list_value[-1]['收盘价'] >= list_value[-1]['ema_200'] and
        if list_value[-1]['ema_200'] > list_value[-2]['ema_200'] > list_value[-3]['ema_200'] > list_value[-4][
            'ema_200']:
            return True
        else:
            return False

    @staticmethod
    def EMA200_近期均大于(list_value):
        """
        如果近期所有收盘价均在均线上
        :param list_value:
        :return:
        """
        result = True
        for value in list_value[-30:]:
            if value['收盘价'] <= value['ema_200']:
                result = False
        return result

    @staticmethod
    def EMA26_近期上涨(list_value):
        if list_value[-1]['long_ema'] >= list_value[-30]['long_ema']:
            return True

    @staticmethod
    def RSI大于50(list_value):
        """
        确保股票动能充足，在50上方说明上涨趋势
        :param list_value:
        :return:
        """
        if list_value[-1]['rsi'] >= 50:
            return True
        else:
            return False

    @staticmethod
    def 换手率(list_value):
        if list_value[-1]['换手率'] >= 0.5:
            return True
        else:
            return False

    @staticmethod
    def 海龟因子(list_value):
        if list_value[-1]['收盘价'] > max(x['最高价'] for x in list_value[-20:-1]):
            return True


    @staticmethod
    def 机器学习因子(value_list, model, 天数=15):
        """
        :param value_list:一个包含该股票多天值的列表，每个值是一个dict，里面包含'rsi'、'换手率'等
        :param gp_code:股票代号
        :return:
        """
        start_time = time.time()
        value_list = value_list[-天数:]

        for value in value_list:
            if value['ema_200'] == 0:
                return False, 0
            value['收盘相对ema_200'] = value['收盘价'] / value['ema_200']
        # 将 value_list 转换为 DataFrame
        print(value_list)
        df = pd.DataFrame(value_list)

        features = [
            "DMA",
            "ATR",
            "momentum",
            "收盘相对ema_200",
            "振幅",
            "涨跌幅",
            'rsi',
            "换手率",
            'kdj_k',
            'kdj_d',
            'kdj_j',
            'wr',
            'macd_line',
            'signal_line',
            # 'macd_histogram'  # 加入此项后准确率反而下降
        ]

        feature_data = df[features]  # 取特征数据

        # 将 DataFrame 转换为 NumPy 数组，作为模型输入
        feature_data = feature_data.values.reshape(1, -1)  # 将数据 reshape 为 1 行，便于模型输入

        # 确保数据中包含模型所需的特征
        # print(f"数据预处理耗时{time.time() - start_time}")
        start_time = time.time()
        # 使用模型进行预测
        prediction = model.predict(xgb.DMatrix(feature_data))
        print(prediction)
        # print(f"预测耗时{time.time() - start_time}, 结果{prediction}")
        # 返回判断结果，例如预测结果为 1 表示买入
        if prediction[0] >= 0.5:
            return True, prediction[0]
        else:
            return False, prediction[0]


def 绘画仿真结果(list_assets: list, list_money: list, sharp: float, parameters: dict, 策略名: str, 年化收益率, 胜率):
    """
    2024/5/2
    绘画仿真结果，资产走势
    :param 策略名:
    :param list_assets: 资产浮点值列表，有多少天就有多少个值
    :param sharp: 夏普率
    :param parameters: 用于策略的参数
    :return:
    """
    plt.plot(list_assets, label='总资产')
    plt.plot(list_money, label='资金')
    title = f"model:{策略名}   夏普比率:{round(sharp, 2)} 年化收益:{round(年化收益率 * 100, 1)}% 胜率:{胜率}"
    y_offset = 0.78
    for key, value in parameters.items():
        plt.figtext(0.25, y_offset, f"{key}：{value}", ha='center', fontsize=10)
        y_offset -= 0.05
    plt.title(title)
    plt.figtext(0.5, 0.02, f"本次测试时间：{timestamp_to_datetime(time.time())}",
                ha='center', fontsize=12)
    plt.legend()
    plt.show()


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


class Sim(threading.Thread):
    """
    2023/3/10:函数速度优化了12倍，约240秒->20秒以内
        multiprocessing支持多cpu核心真线程，可占用全核心资源高速运行
    """

    def __init__(
            self,
            bk_indexs: list[str],
            money: int,
            method: Type[XJSTradingMethod],
            parameters,
            days_limit=1000,
            start_date_time='2020-03-22',
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
        self.bk_index = bk_indexs
        my_dic_result = BKManager.load_multi_data_from_file(bk_indexs=bk_indexs)
        self.sim_data: dict = BKManager.produce_simData(my_dic_result, start_date_time=start_date_time)
        self.date_timestamp_now = 0
        self.money: float = money
        self.assets_now: float = money
        self.method: XJSTradingMethod = method(parameters=parameters)
        self.inventory = defaultdict(dict)
        self.print_interval: int = 1
        self.list_assets: list = []
        self.list_money = []  # 记录一下活动资金变化趋势
        self.window_size: int = parameters['window_size']  # 为防止后期数据量过大，simData预处理只截取最近部分个数据
        self.parameters: dict = parameters
        self.days_limit = days_limit
        self.start_date_time = start_date_time
        self.start_date_time_act = start_date_time  # 实际开始操作的时间，在第一次交易时更新该数据
        self.start_timestamp = datetime_to_timestamp(start_date_time)
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
        绘画仿真结果(
            list_assets=self.list_assets,
            list_money=self.list_money,
            parameters=self.parameters,
            策略名=self.method.__name__,
            sharp=sharpe_ratio,
            年化收益率=年化收益率,
            胜率=胜率
        )

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

    def 提前知道本次购买结果(self, dict_can_buy, 向后观察窗口=30):
        """

        :param dict_can_buy: gp_code:{'price','num'}
        :return:
        """
        for gp_code, gp_buy_value in dict_can_buy.items():
            Log.trace(f"正在提前偷窥{timestamp_to_datetime(self.date_timestamp_now)}买入{gp_code}的结果")
            gp_buy_price = gp_buy_value['price']
            gp_data = self.data_after_pre_solve_all[gp_code]
            list_day_data = []
            已观察天数 = 0
            for day_data in gp_data:
                date_timestamp = day_data['date_timestamp']
                if date_timestamp >= self.date_timestamp_now:
                    Log.trace(f"计入{timestamp_to_datetime(date_timestamp)}")
                    list_day_data.append(day_data)
                    已观察天数 += 1
                if 已观察天数 == 向后观察窗口:
                    break
            plt.plot([x['收盘价'] for x in list_day_data])
            plt.title(gp_code)
            plt.show()
            a = input('')


class 因子测试(threading.Thread):
    def __init__(self, bk_indexs, start_date_time='2020-03-22'):
        super().__init__()
        self.bk_index = bk_indexs
        my_dic_result = bk.load_multi_data_from_file(bk_indexs=bk_indexs)
        self.sim_data: dict = bk.produce_simData(my_dic_result, start_date_time=start_date_time)
        self.list_sim_data_values = list(self.sim_data.values())
        self.data_after_pre_solve_all = data_pre_solve(
            sim_data=self.sim_data,
            date_timestamp_now=list(self.sim_data.keys())[-1],
            window_size=len(self.sim_data.keys()) - 1
        )
        self.model_判涨 = xgb.Booster()
        self.model_判涨.load_model('machine_learning_models/xgboost_up_model.json')
        # self.model_判跌 = xgb.Booster()
        # self.model_判跌.load_model('machine_learning_models/xgboost_down_model.json')

    def run(self):
        交易次数 = 0
        获胜次数 = 0
        for date_timestamp, gps_data in self.sim_data.items():
            data_after_pre_solve = data_pre_solve(
                sim_data=self.sim_data,
                date_timestamp_now=date_timestamp,
                window_size=180
            )
            当日交易次数 = 0
            最大当日测试交易 = 100000  # 节省时间

            # print(timestamp_to_datetime(date_timestamp))
            for gp_code, gp_days_data in data_after_pre_solve.items():
                if 当日交易次数 >= 最大当日测试交易:
                    break
                # print(len(gp_days_data))
                if len(gp_days_data) < 180:
                    # print(f"还没达到{180}天，不测试因子")
                    continue
                # 是否符合, r2, omega, phi, period, A = 因子.拟合_正弦函数(
                #     list_y=np.array([value['收盘价'] for value in gp_days_data[-180:]])
                # )
                因子_EMA金叉 = 因子.EMA金叉(list_value=gp_days_data)
                因子_KDJ金叉 = 因子.KDJ金叉(list_value=gp_days_data)
                因子_EMA200上涨 = 因子.EMA200上涨(list_value=gp_days_data)
                因子_EMA200_均大于 = 因子.EMA200_近期均大于(list_value=gp_days_data)
                因子_EMA死叉_近期 = 因子.EMA死叉_近期出现(list_value=gp_days_data)
                因子_布林 = 因子.布林下穿底部(list_value=gp_days_data)
                因子_布林近期触底 = 因子.布林近期触底(list_value=gp_days_data)
                # 因子_判涨_机器学习, 判涨概率 = 因子.机器学习因子(value_list=gp_days_data, model=self.model_判涨)
                # 因子_判跌_机器学习, 判跌概率 = 因子.机器学习因子(value_list=gp_days_data, model=self.model_判跌)
                # 因子_机器学习 = 因子_判涨_机器学习
                # wztong = 因子.wztong()
                rsi = gp_days_data[-1]['rsi']
                # wr = gp_days_data[-1]['wr']
                macd_line = gp_days_data[-1]['macd_line']
                signal_line = gp_days_data[-1]['signal_line']
                macd_histogram = gp_days_data[-1]['macd_histogram']

                换手率 = gp_days_data[-1]['换手率']
                if (
                        因子_EMA金叉
                        # 因子_布林
                        # and not 因子_EMA200上涨
                        # and 因子_布林近期触底
                        # and 换手率 > 0.5
                        # and 因子_EMA200_均大于
                        # and macd_histogram > 0
                ):
                    当日交易次数 += 1
                    交易次数 += 1
                    涨幅, 最高价, 最高价日期 = self.计算后N天内最大涨幅(date_timestamp_now=date_timestamp,
                                                                        gp_code=gp_code)
                    if 涨幅 <= 1.05:
                        print(colorized_string(f"{gp_code} 失败涨幅:{涨幅}", 'g'), )
                    # print(f"{gp_code} {r2} {omega} {phi} {period} {A} 涨幅:{涨幅}")
                    else:
                        获胜次数 += 1
                        print(colorized_string(
                            f"{gp_code} {timestamp_to_datetime(date_timestamp)}获胜涨幅:{涨幅} 在{最高价日期}达到{最高价}",
                            'r'))
                        # self.画出前后N天走势(date_timestamp_now=date_timestamp, gp_code=gp_code)
                    print(f"总交易次数{交易次数} 获胜次数{获胜次数} 胜率{获胜次数 / 交易次数}")

        print(f"总交易次数{交易次数} 获胜次数{获胜次数} 胜率{获胜次数 / 交易次数}")

    def \
            计算后N天内最大涨幅(self, date_timestamp_now, gp_code, window=15):
        date_timestamps = list(self.sim_data.keys())

        now_position = date_timestamps.index(date_timestamp_now)
        end_position = min(now_position + window, len(date_timestamps) - 1)
        if now_position == end_position:
            return 1.0, 0, date_timestamp_now
        end_timestamp = date_timestamps[min(len(date_timestamps), end_position)]
        # print(f"正在查看{gp_code}之后的最大涨幅，从{timestamp_to_datetime(date_timestamp_now)}到{timestamp_to_datetime(end_timestamp)}")
        list_收盘价 = []
        最高价 = 0
        最高价日期 = ""
        for i in range(now_position + 1, end_position):
            gps_data = self.list_sim_data_values[i]
            if gp_code not in gps_data:
                # print(f"{gp_code}停牌？")
                continue
            gp_data = gps_data[gp_code]
            # print(gp_data)
            list_收盘价.append(gp_data['收盘价'])
            if gp_data['最高价'] >= 最高价:
                最高价 = gp_data['最高价']
                最高价日期 = timestamp_to_datetime(gp_data['date_timestamp'])
            # print(list_收盘价)
        if list_收盘价 == []:
            return 1.0, 0, 0
        涨幅 = 最高价 / list_收盘价[0]

        return 涨幅, 最高价, 最高价日期

    def 画出前后N天走势(self, date_timestamp_now, gp_code, window=15):
        """
        绘制收盘价、KDJ、MACD的30天前后走势图，并新增EMA数据展示
        :param window: 前后天数，如30、60，加入画30天，就是前后各30共60天数据
        :param date_timestamp_now: 当前日期时间戳
        :param gp_code: 股票代码
        """
        # 获取日期序列并确定范围
        date_timestamps = list(self.sim_data.keys())
        now_position = date_timestamps.index(date_timestamp_now)
        start_position = max(0, now_position - window)
        end_position = min(now_position + window, len(date_timestamps) - 1)

        # 初始化数据列表
        list_收盘价 = []
        list_k = []
        list_d = []
        list_j = []
        list_macd = []
        list_signal = []
        list_histogram = []
        list_ema_200 = []
        list_ema_short = []
        list_ema_long = []

        list_booling_lower = []
        list_booling_mid = []
        list_booling_upper = []

        # 提取相关数据
        for i in range(start_position, end_position):
            gps_data = self.sim_data[date_timestamps[i]]
            if gp_code not in gps_data:
                continue  # 如果股票停牌或无数据，则跳过

            gp_data = gps_data[gp_code]
            收盘价 = gp_data['收盘价']
            list_收盘价.append(收盘价)

            # 获取EMA和其他指标
            ema_200_value = gp_data.get('ema_200', 0)
            list_ema_200.append(ema_200_value)

            ema_short_value = gp_data.get('short_ema', 0)
            ema_long_value = gp_data.get('long_ema', 0)
            list_ema_short.append(ema_short_value)
            list_ema_long.append(ema_long_value)

            # 获取布林带
            list_booling_lower.append(gp_data.get('booling_lower', 0))
            list_booling_mid.append(gp_data.get('booling_mid', 0))
            list_booling_upper.append(gp_data.get('booling_upper', 0))

            # 获取KDJ数据
            k_value = gp_data.get('kdj_k', 0)
            d_value = gp_data.get('kdj_d', 0)
            j_value = gp_data.get('kdj_j', 0)
            list_k.append(k_value)
            list_d.append(d_value)
            list_j.append(j_value)

            # 获取MACD数据
            macd_value = gp_data.get('macd', 0)
            signal_value = gp_data.get('macd_signal', 0)
            histogram_value = gp_data.get('macd_histogram', 0)
            list_macd.append(macd_value)
            list_signal.append(signal_value)
            list_histogram.append(histogram_value)

        # 开始绘图
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        # 绘制收盘价和ema_200
        axs[0].plot(list_收盘价, label="Close Price", color='blue')
        axs[0].plot(list_booling_lower, label="bolling_lower")
        axs[0].plot(list_booling_mid, label="bolling_mid")
        axs[0].plot(list_booling_upper, label="bolling_upper")
        axs[0].plot(list_ema_200, label="EMA 200", color='green', linestyle='--')
        axs[0].set_title(f"{gp_code} 收盘价和EMA 200")
        axs[0].legend()

        # 绘制KDJ指标
        axs[1].plot(list_k, label="K", color='r')
        axs[1].plot(list_d, label="D", color='b')
        axs[1].plot(list_j, label="J", color='g')
        axs[1].set_title(f"{gp_code} KDJ")
        axs[1].legend()

        # 绘制MACD指标
        axs[2].plot(list_macd, label="MACD", color='purple')
        axs[2].plot(list_signal, label="Signal", color='orange')
        axs[2].bar(range(len(list_histogram)), list_histogram, label="Histogram", color='grey', alpha=0.5)
        axs[2].set_title(f"{gp_code} MACD")
        axs[2].legend()

        # 绘制EMA短期和长期
        axs[3].plot(list_ema_short, label="EMA Short", color='red')
        axs[3].plot(list_ema_long, label="EMA Long", color='blue')
        axs[3].set_title(f"{gp_code} EMA Short vs EMA Long")
        axs[3].legend()

        # 设置共享x轴的标签
        plt.xlabel("Days")
        plt.tight_layout()
        plt.show()
        a = input('input')


class 一些参数:
    class Method3:
        parameter1 = {
            'sell_percent': 1.05,
            'buy_percent': 0.97,
            '探测起始天数': 30,
            '探测最小值偏移比例': 1,
            '连续下跌天数': 2,
            'boolling_band_width': 7,
            'window_size': 200
        }
        parameter2 = {
            'assets_once': 3000,
            'sell_percent': 1.03,
            'buy_percent': 0.99,
            '探测起始天数': 60,
            '探测最小值偏移比例': 1,
            'window_size': 200,
            '止损比例': -3,  # 百分点
            '止损时长': 90,  # 天
        }


if __name__ == '__main__':
    # for sell_percent in range(101, 105):
    #     for buy_percent in range(95, 105):
    #         Sim(bk_index='BK0153', money=10000, method=method2_new,
    #             parameters=[sell_percent / 100, buy_percent / 100, 30, 0.6]).start()
    # break
    # break

    # for gp_code, value in my_dic_result.items():
    #     print(f"{gp_code}")
    #     print(value)
    #     break
    # print(Sim.annualized_return(
    #     start_date_str='2020-04-21',
    #     end_date_str='2024-04-21',
    #     start_value=50000,
    #     end_value=150000
    # ))
    Sim(bk_indexs=[
        # 'BK0145',
        'BK0153'
    ],
        money=1000000,
        method=XJSTradingMethod3,
        parameters=一些参数.Method3.parameter2,
        days_limit=5000,
        start_date_time='2020-05-23'
    ).start()
    # Sim(bk_indexs=[
    #     'BK0145',
    #     # 'BK0153'
    # ],
    #     money=10000,
    #     method=XJSTradingMethod3,
    #     parameters=一些参数.Method3.parameter2,
    #     days_limit=5000,
    #     start_date_time='2019-05-23'
    #     ).advice()
    # 因子测试(
    #     bk_indexs=['BK0153'],
    #     start_date_time='2020-05-23'
    # ).run()
