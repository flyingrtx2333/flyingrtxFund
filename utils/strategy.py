from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
import numpy as np
import time
from collections import defaultdict
from utils.factors import Factors
from utils.Log import Log

@dataclass
class StrategyMeta:
    """策略元数据"""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]

class BaseStrategy:
    """策略基类"""

    def __init__(self, parameters, product_mode=False):
        super().__init__()
        self.assets_once = 3000
        self.product_mode = product_mode  # 可继承
        self.parameters = parameters
        self.window_size = parameters.get('window_size', 100)

    @classmethod
    @abstractmethod
    def get_meta(cls) -> StrategyMeta:
        """获取策略元数据"""
        pass

    def run(self, data_after_pre_solve, inventory, date_timestamp_now, money_now) -> tuple[dict, dict]:
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
        dict_can_sell = self.sell_check(self.data_after_pre_solve, self.inventory_now)
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


class XJSTradingMethod1(BaseStrategy):
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


class XJSTradingMethod2(BaseStrategy):
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

    @classmethod
    def get_meta(cls) -> StrategyMeta:
        return StrategyMeta(
            name="方法2 - 二次拟合，开口向上者",
            description="二次拟合，开口向上者",
            parameters={
                "assets_once": {
                    "type": "float",
                    "default": 3000,
                    "description": "单次交易金额"
                },
                "sell_percent": {
                    "type": "float", 
                    "default": 1.03,
                    "description": "止盈比例"
                },
                "buy_percent": {
                    "type": "float",
                    "default": 0.99,
                    "description": "买入比例"
                },
                "探测起始天数": {
                    "type": "int",
                    "default": 60,
                    "description": "技术指标计算天数"
                }
            }
        )

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


class XJSTradingMethod3(BaseStrategy):
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

    @classmethod
    def get_meta(cls) -> StrategyMeta:
        return StrategyMeta(
            name="方法3 - 二次拟合，开口向上者",
            description="二次拟合，开口向上者",
            parameters={
                "assets_once": {
                    "type": "float",
                    "default": 3000,
                    "description": "单次交易金额"
                },
                "sell_percent": {
                    "type": "float", 
                    "default": 1.03,
                    "description": "止盈比例"
                },
                "buy_percent": {
                    "type": "float",
                    "default": 0.99,
                    "description": "买入比例"
                },
                "探测起始天数": {
                    "type": "int",
                    "default": 60,
                    "description": "技术指标计算天数"
                }
            }
        )


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
            and Factors.布林上穿中线(list_value=value_list)
            # and 因子.EMA金叉(list_value=value_list)
            # and 因子.布林近期触底(list_value=value_list)
            # and 因子.换手率(list_value=value_list)
            and Factors.EMA200上涨(list_value=value_list)
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
                止盈比例 = self.sell_percent
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

# 策略注册表
STRATEGY_REGISTRY = {
    "xjs_method3": XJSTradingMethod3,
    # 可以继续注册其他策略...
} 