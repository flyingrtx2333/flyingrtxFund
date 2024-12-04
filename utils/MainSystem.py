"""
2023/2/11:开始记录
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numpy.linalg
import requests
from modules.modulePublic import ColorizedString, drawK线
import ModuleGet
import matplotlib.pyplot as plt


class 选股器:
    def __init__(self, list_value, 指针, 连续下跌天数=4):
        self.选股结果1 = self.选股条件1(
            list_value=list_value,
            指针=指针
        )
        self.选股结果2 = self.选股条件2(
            list_value=list_value,
            指针=指针,
            连续下跌天数=连续下跌天数
        )
        self.选股结果3 = self.选股条件3_二次拟合(
            list_value=list_value,
            指针=指针
        )

    def 选股(self):
        if self.选股结果1 and self.选股结果2 and self.选股结果3:
            # if self.选股结果1 and self.选股结果3:
            # if self.选股结果3:
            return True
        else:
            return False

    def 选股条件1(self, list_value, 指针):
        """判断是否连续10天涨跌幅都在一定范围内"""
        list_十日涨跌幅绝对值 = []
        if len(list_value) < 11:
            return False
        else:
            for i in range(1, 11):
                try:
                    涨跌幅 = abs(list_value[指针 - i]['涨跌幅'])
                except TypeError:
                    continue
                if 3 >= 涨跌幅 >= 0.2:
                    continue
                else:
                    return False
            return True

    def 选股条件2(self, list_value, 指针, 连续下跌天数):
        """判断是否连续数天下跌"""
        if len(list_value) < 连续下跌天数 + 1:
            return False
        else:
            for i in range(1, 连续下跌天数 + 1):
                try:
                    涨跌幅 = list_value[指针 - i]['涨跌幅']
                except TypeError:
                    continue
                if 涨跌幅 <= 0:
                    continue
                else:
                    return False
            return True

    def 选股条件3_二次拟合(self, list_value, 指针):
        list_x = range(max(0, len(list_value) - 50), len(list_value))
        list_y = []
        for i in range(0, len(list_x)):
            横坐标 = list_x[i]
            list_y.append(list_value[横坐标]['收盘价'])
        try:
            coef = np.polyfit(list_x, list_y, 2)
            y_fit = np.polyval(coef, list_x)
            a = coef[0]
            最小值 = y_fit.min()
            # if a > 0 and y_fit[-1] < y_fit[0] and 最小值!=y_fit[-1]:
            if a > 0:
                return True
        except numpy.linalg.LinAlgError:
            return False


# 全体股票数据, 以时间为索引的股票数据, 全个股代号表 = ModuleGet.超级数据库读取('data/core_要用的板块.log').start()
以时间为索引的股票数据 = ModuleGet.bk.produce_simData(
    bk_result=ModuleGet.bk.load_data_from_file('BK0153')
)


class 仿真:
    def __init__(self):
        self.session = requests.session()
        self.日期 = None
        self.持仓 = {}  # 个股代号:手数
        self.dic_买入历史 = {}  # {个股代号:{日期:{价格, 手数},日期:{价格, 手数},...}}
        self.list_资产走势 = []
        self.code = None
        self.操作时间限制 = 1000
        self.set_一次操作资金 = 3000

    def start(self, 股票数据, 仿真条件, 是否输出=True):
        self.data = 股票数据
        self.是否输出 = 是否输出
        self.初始资金 = 仿真条件['初始资金']
        self.资金 = self.初始资金
        self.买点, self.卖点 = 仿真条件['买点'], 仿真条件['卖点'],
        self.选股条件_连续下跌天数 = 仿真条件['选股条件_连续下跌天数']
        # 解析后股票数据 = self.获取数据解析(jsonobj=股票数据)
        self.process_main(data=self.data)
        return self.list_资产走势

    def get_klines_data(self, 股票代码):
        毫秒时间戳 = int(time.time() * 1000)
        关键值 = "35106668583059676032"
        url_1 = "http://21.push2his.eastmoney.com/api/qt/stock/kline/get"
        call_back = f"jQuery{关键值}_{毫秒时间戳}"
        url_2 = f"?cb={call_back}"
        url_3 = f"&secid=1.{股票代码}{'&ut=fa5fd1943c7b386f172d6893dbfba10b'}"
        url_4 = "&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6"
        url_5 = "&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61"
        url_6 = "&klt=101"
        url_7 = "&fqt=1"
        url_8 = "&end=20500101"
        url_9 = "&lmt=1000000"
        url_10 = f"&_={毫秒时间戳}"
        url = f"{url_1}{url_2}{url_3}{url_4}{url_5}{url_6}{url_7}{url_8}{url_9}{url_10}"
        res = self.session.get(url)
        try:
            jsonobj = json.loads(res.text.split("(")[1][:-2])
            data = jsonobj['data']
            code = data['code']
        except:
            url_1 = "http://92.push2his.eastmoney.com/api/qt/stock/kline/get"
            call_back = f"jQuery{关键值}_{毫秒时间戳}"
            url_2 = f"{'?cb='}{call_back}"
            url_3 = f"{'&secid=0.'}{股票代码}{'&ut=fa5fd1943c7b386f172d6893dbfba10b'}"
            url_4 = "&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6"
            url_5 = "&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61"
            url_6 = "&klt=101"
            url_7 = "&fqt=1"
            url_8 = "&end=20500101"
            url_9 = "&lmt=1000000"
            url_10 = f"&_={毫秒时间戳}"
            url = f"{url_1}{url_2}{url_3}{url_4}{url_5}{url_6}{url_7}{url_8}{url_9}{url_10}"

            res = requests.get(url)
            jsonobj = json.loads(res.text.split(call_back + "(")[1][:-2])
        return jsonobj

    def 获取数据解析(self, jsonobj):
        data = jsonobj['data']
        klines = data['klines']
        dic_klines = {}
        for i in range(0, len(klines)):
            klines[i] = klines[i].split(',')
            日期 = klines[i][0]
            dic_klines[日期] = {
                '开盘价': float(klines[i][1]),
                '收盘价': float(klines[i][2]),
                '最高价': float(klines[i][3]),
                '最低价': float(klines[i][4]),
                '总手': float(klines[i][5]),
                '金额': float(klines[i][6]),
                '振幅': float(klines[i][7]),
                '涨跌幅': float(klines[i][8]),
                '涨跌额': float(klines[i][9]),
                "换手率": float(klines[i][10])
            }
        return dic_klines

    def process_main(self, data):
        list_日期 = []
        list_value = []
        dic_每个个股数据记录 = {}
        for 日期, value in data.items():
            list_日期.append(日期)
            list_value.append(value)
        for i in range(0, len(list_日期)):
            日期 = list_日期[i]
            value = list_value[i]
            self.日期 = 日期
            self.输出(ColorizedString(f"现在时间来到了{self.日期}，指针{i}", 'p'))
            for 个股代号, 个股当天数据 in value.items():
                if 个股代号 not in dic_每个个股数据记录:
                    dic_每个个股数据记录[个股代号] = {}
                    dic_每个个股数据记录[个股代号]['list_日期_个股'] = []
                    dic_每个个股数据记录[个股代号]['list_value_个股'] = []
                dic_每个个股数据记录[个股代号]['list_日期_个股'].append(self.日期)
                dic_每个个股数据记录[个股代号]['list_value_个股'].append(个股当天数据)
                self.code = 个股代号

                [选股结果1, 选股结果2] = 选股器(
                    list_value=dic_每个个股数据记录[个股代号]['list_value_个股'],
                    指针=i,
                    连续下跌天数=self.选股条件_连续下跌天数
                ).选股()

                if 选股结果1 and 选股结果2:
                    self.输出(ColorizedString(f"{self.code}满足选股条件1和2", 'g'))
                else:
                    """不满足选股条件，如果没有持仓就不关注了"""
                    if self.code not in self.持仓:
                        continue
                    else:
                        if self.持仓[self.code] == 0:
                            continue
                开盘价 = 个股当天数据['开盘价']
                收盘价 = 个股当天数据['收盘价']
                最高价 = 个股当天数据['最高价']
                最低价 = 个股当天数据['最低价']
                买入价格, 买入手数 = self.买入条件(
                    list_日期=dic_每个个股数据记录[个股代号]['list_日期_个股'],
                    list_value=dic_每个个股数据记录[个股代号]['list_value_个股'],
                    指针=i,
                    买点=self.买点
                )
                if 买入手数 > 0:
                    self.买入(买入手数, 买入价格)
                # 最低价格日期, 最低买入价格 = self.查看最低买入价格()
                list_卖出价格, list_卖出手数, list_对应日期 = self.卖出条件(
                    list_value=dic_每个个股数据记录[个股代号]['list_value_个股'],
                    指针=i,
                    卖点=self.卖点
                )
                for k in range(0, len(list_卖出价格)):
                    if list_卖出价格[k] > 0:
                        卖出结果 = self.卖出(list_卖出手数[k], list_卖出价格[k])
                        if 卖出结果:
                            del self.dic_买入历史[self.code][list_对应日期[k]]
            """开始统计当天资产"""
            总资产 = self.资金
            for 个股代号, 持仓手数 in self.持仓.items():
                总资产 += 持仓手数 * 100 * data[self.日期][个股代号]['收盘价']
            self.list_资产走势.append(总资产)
            self.输出(ColorizedString(f"今天统计资金/总资产：{self.资金}/{总资产},持仓状态：{self.持仓}", 'b'))
        self.输出(f"初始资金：{self.初始资金},最终总资产：{self.list_资产走势[-1]}")
        return self.list_资产走势
        # plt.plot(self.list_资产走势)
        # plt.show()

    def 操作手数决定(self, 价格):
        手数 = 0
        while True:
            if 手数 * 100 * 价格 > self.set_一次操作资金:
                break
            else:
                手数 += 1
                continue
        return 手数

    def 查看最低买入价格(self):
        最低价格 = 99999
        最低价格日期 = None
        for 日期, value in self.dic_买入历史[self.code].items():
            买入价格 = value['价格']
            if 买入价格 <= 最低价格:
                最低价格 = 买入价格
                最低价格日期 = 日期
        return 最低价格日期, 最低价格

    def 买入(self, 手数, 价格):
        总价 = 手数 * 价格 * 100
        if 总价 > 20000:
            手续费 = 总价 * 0.00025
        else:
            手续费 = 5
        需花费 = 总价 + 手续费
        if self.资金 >= 需花费:
            if self.code not in self.持仓:
                self.持仓[self.code] = 手数
            else:
                self.持仓[self.code] += 手数
            self.资金 = round(self.资金 - 需花费, 2)
            if self.code not in self.dic_买入历史:
                self.dic_买入历史[self.code] = {}
            self.dic_买入历史[self.code] = {
                self.日期: {
                    "价格": 价格,
                    "手数": 手数
                }
            }

            self.输出(ColorizedString(f"{self.日期}以{价格}买入股票{self.code}：{手数}手", 'r'))
            # self.输出(f"当前{self.code}持仓手数：{self.持仓[self.code]}，剩余资金：{self.资金}")
            return True
        else:
            return False

    def 卖出(self, 手数, 价格):
        总价 = 手数 * 价格 * 100
        if 总价 > 20000:
            手续费 = 总价 * 0.00025
        else:
            手续费 = 5
        可得到 = 总价 - 手续费
        if self.code not in self.持仓:
            self.持仓[self.code] = 0
        if self.持仓[self.code] >= 手数:
            self.持仓[self.code] -= 手数
            self.资金 = round(self.资金 + 可得到, 2)
            self.输出(ColorizedString(f"{self.日期}以{价格}卖出股票{self.code}：{手数}手", 'g'))
            # self.输出(f"当前{self.code}持仓手数：{self.持仓[self.code]}，剩余资金：{self.资金}")
            return True
            # self.输出(f"初始资金：{self.初始资金},最终资金：{self.资金},最终持仓:{self.持仓手数 * 100 * 价格}")
        else:
            return False

    def 买入条件(self, list_日期, list_value, 指针, 买点=0.99):
        """返回买入价格和买入手数"""
        if self.code in self.持仓:
            if self.持仓[self.code] != 0:
                return 0, 0
        if 指针 > 0:
            今日开盘价 = list_value[指针]['开盘价']
            今日收盘价 = list_value[指针]['收盘价']
            今日最低价 = list_value[指针]['最低价']
            今日最高价 = list_value[指针]['最高价']
            昨日开盘价 = list_value[指针 - 1]['开盘价']
            昨日收盘价 = list_value[指针 - 1]['收盘价']
            买入价格 = round(今日开盘价 * 买点, 2)
            买入手数 = self.操作手数决定(买入价格)
            if 今日最低价 <= 买入价格:
                return 买入价格, 买入手数
            else:
                return 0, 0
        else:
            return 0, 0

    def 卖出条件(self, list_value, 指针, 卖点=1.01):
        list_卖出价格 = []
        list_卖出手数 = []
        list_对应日期 = []
        if self.code not in self.持仓 or self.持仓[self.code] == []:
            return list_卖出价格, list_卖出手数, list_对应日期
        if 指针 > 0:
            今日开盘价 = list_value[指针]['开盘价']
            今日收盘价 = list_value[指针]['收盘价']
            今日最高价 = list_value[指针]['最高价']
            卖出价格 = 0
            卖出手数 = 0
            for 日期, value in self.dic_买入历史[self.code].items():
                if 日期 == self.日期:
                    """是当天的买入历史，不可以当天卖出"""
                    continue
                买入价格 = value['价格']
                买入手数 = value['手数']
                卖出价格 = round(买入价格 * 卖点, 2)
                if 今日最高价 >= 卖出价格:
                    利润 = self.利润计算(买入价格, 卖出价格, 买入手数)
                    self.输出(ColorizedString(
                        f"{self.日期}，{self.code}可以以{卖出价格}卖出,对应买入价格{买入价格}可以卖出,利润:{利润}", 'g'))
                    卖出手数 = 买入手数
                    list_卖出价格.append(卖出价格)
                    list_卖出手数.append(卖出手数)
                    list_对应日期.append(日期)
                else:
                    # self.输出(ColorizedString(f"{self.日期}最高价{今日最高价},对应买入价格{买入价格}，不可以卖出", 'y'))
                    pass
        return list_卖出价格, list_卖出手数, list_对应日期

    def 利润计算(self, 买入价, 卖出价, 手数):
        if 买入价 * 手数 * 100 < 20000:
            买入手续费 = 5
        else:
            买入手续费 = 买入价 * 手数 * 100 * 0.00025
        if 卖出价 * 手数 * 100 < 20000:
            卖出手续费 = 5
        else:
            卖出手续费 = 买入价 * 手数 * 100 * 0.00025
        利润 = 卖出价 * 手数 * 100 - 买入价 * 手数 * 100 - 买入手续费 - 卖出手续费
        return 利润

    def 输出(self, 字符串):
        if self.是否输出:
            print(字符串)


# noinspection PyAttributeOutsideInit
class 手动仿真:
    def __init__(self, 股票数据):
        self.session = requests.session()
        self.日期 = None
        self.持仓 = {}  # 个股代号:手数
        self.dic_买入历史 = {}  # {个股代号:{日期:{价格, 手数},日期:{价格, 手数},...}}
        self.list_资产走势 = []
        self.股票数据 = 股票数据

        self.code = None
        self.操作时间限制 = 1000
        self.set_一次操作资金 = 3000

        self.统计_list现金走势 = []
        self.统计_买入次数 = 0
        self.统计_清仓次数 = 0
        self.统计_list持仓时长 = []

    def start(self, 仿真条件, 是否输出=True):
        """开始仿真"""
        self.是否输出 = 是否输出
        self.初始资金 = 仿真条件['初始资金']
        self.资金 = self.初始资金
        self.买点, self.卖点 = 仿真条件['买点'], 仿真条件['卖点'],
        self.选股条件_连续下跌天数 = 仿真条件['选股条件_连续下跌天数']
        # 解析后股票数据 = self.获取数据解析(jsonobj=股票数据)
        self.process_main(data=self.股票数据)
        return self.list_资产走势

    def get_klines_data(self, 股票代码):
        毫秒时间戳 = int(time.time() * 1000)
        关键值 = "35106668583059676032"
        url_1 = "http://21.push2his.eastmoney.com/api/qt/stock/kline/get"
        call_back = f"jQuery{关键值}_{毫秒时间戳}"
        url_2 = f"?cb={call_back}"
        url_3 = f"&secid=1.{股票代码}{'&ut=fa5fd1943c7b386f172d6893dbfba10b'}"
        url_4 = "&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6"
        url_5 = "&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61"
        url_6 = "&klt=101"
        url_7 = "&fqt=1"
        url_8 = "&end=20500101"
        url_9 = "&lmt=1000000"
        url_10 = f"&_={毫秒时间戳}"
        url = f"{url_1}{url_2}{url_3}{url_4}{url_5}{url_6}{url_7}{url_8}{url_9}{url_10}"
        res = self.session.get(url)
        try:
            jsonobj = json.loads(res.text.split("(")[1][:-2])
            data = jsonobj['data']
            code = data['code']
        except:
            url_1 = "http://92.push2his.eastmoney.com/api/qt/stock/kline/get"
            call_back = f"jQuery{关键值}_{毫秒时间戳}"
            url_2 = f"{'?cb='}{call_back}"
            url_3 = f"{'&secid=0.'}{股票代码}{'&ut=fa5fd1943c7b386f172d6893dbfba10b'}"
            url_4 = "&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6"
            url_5 = "&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61"
            url_6 = "&klt=101"
            url_7 = "&fqt=1"
            url_8 = "&end=20500101"
            url_9 = "&lmt=1000000"
            url_10 = f"&_={毫秒时间戳}"
            url = f"{url_1}{url_2}{url_3}{url_4}{url_5}{url_6}{url_7}{url_8}{url_9}{url_10}"

            res = requests.get(url)
            jsonobj = json.loads(res.text.split(call_back + "(")[1][:-2])
        return jsonobj

    def 获取数据解析(self, jsonobj):
        data = jsonobj['data']
        klines = data['klines']
        dic_klines = {}
        for i in range(0, len(klines)):
            klines[i] = klines[i].split(',')
            日期 = klines[i][0]
            dic_klines[日期] = {
                '开盘价': float(klines[i][1]),
                '收盘价': float(klines[i][2]),
                '最高价': float(klines[i][3]),
                '最低价': float(klines[i][4]),
                '总手': float(klines[i][5]),
                '金额': float(klines[i][6]),
                '振幅': float(klines[i][7]),
                '涨跌幅': float(klines[i][8]),
                '涨跌额': float(klines[i][9]),
                "换手率": float(klines[i][10])
            }
        return dic_klines

    def process_main(self, data):
        list_日期 = []
        list_value = []
        dic_每个个股数据记录 = {}
        """首先建立所有个股单独空表"""
        for 个股代号 in 全个股代号表:
            dic_每个个股数据记录[个股代号] = {}
            dic_每个个股数据记录[个股代号]['list_日期_个股'] = []
            dic_每个个股数据记录[个股代号]['list_value_个股'] = []
        for 日期, value in data.items():
            list_日期.append(日期)
            list_value.append(value)
        for i in range(0, len(list_日期)):
            日期 = list_日期[i]
            value = list_value[i]
            self.日期 = 日期
            self.输出(ColorizedString(f"现在时间来到了{self.日期}，指针{i}", 'p'))
            for j in range(0, len(全个股代号表)):
                个股代号 = 全个股代号表[j]
                if 个股代号 in value:
                    """这只股票当天有数据"""
                    个股当天数据 = value[个股代号]
                    dic_每个个股数据记录[个股代号]['list_日期_个股'].append(self.日期)
                    dic_每个个股数据记录[个股代号]['list_value_个股'].append(个股当天数据)
                    self.code = 个股代号
                    """查看股票是否满足选股要求"""
                    选股结果 = 选股器(
                        list_value=dic_每个个股数据记录[个股代号]['list_value_个股'],
                        指针=i,
                        连续下跌天数=self.选股条件_连续下跌天数
                    ).选股()
                    """只关注满足选股条件的股票或者有持仓的股票"""
                    if 选股结果:
                        self.输出(ColorizedString(f"{self.code}满足选股条件", 'g'))
                        # drawK线(dic_每个个股数据记录[self.code], self.code)
                    else:
                        """不满足选股条件，如果没有持仓就不关注了"""
                        if self.code not in self.持仓:
                            continue
                        else:
                            if self.持仓[self.code] == 0:
                                continue
                    """到这里就开始关注该股票，可以检测买入或者卖出条件，符合则执行相应操作"""
                    开盘价 = 个股当天数据['开盘价']
                    收盘价 = 个股当天数据['收盘价']
                    最高价 = 个股当天数据['最高价']
                    最低价 = 个股当天数据['最低价']
                    买入价格, 买入手数 = self.买入条件(
                        list_日期=dic_每个个股数据记录[个股代号]['list_日期_个股'],
                        list_value=dic_每个个股数据记录[个股代号]['list_value_个股'],
                        指针=i,
                        买点=self.买点
                    )
                    if 买入手数 > 0:
                        self.买入(买入手数, 买入价格)
                    # 最低价格日期, 最低买入价格 = self.查看最低买入价格()
                    list_卖出价格, list_卖出手数, list_对应日期 = self.卖出条件(
                        list_value=dic_每个个股数据记录[个股代号]['list_value_个股'],
                        指针=i,
                        卖点=self.卖点
                    )
                    for k in range(0, len(list_卖出价格)):
                        if list_卖出价格[k] > 0:
                            卖出结果 = self.卖出(list_卖出手数[k], list_卖出价格[k])
                            if 卖出结果:
                                del self.dic_买入历史[self.code][list_对应日期[k]]
                                # self.统计_list持仓时长.append()
                                self.统计_清仓次数 += 1
                else:
                    """这只股票当天可能出现停牌等意外情况，没有数据"""
                    dic_每个个股数据记录[个股代号]['list_日期_个股'].append(self.日期)
                    dic_每个个股数据记录[个股代号]['list_value_个股'].append(None)
            # for 个股代号, 个股当天数据 in value.items():
            #     if 个股代号 not in dic_每个个股数据记录:
            #         dic_每个个股数据记录[个股代号] = {}
            #         dic_每个个股数据记录[个股代号]['list_日期_个股'] = []
            #         dic_每个个股数据记录[个股代号]['list_value_个股'] = []
            #     dic_每个个股数据记录[个股代号]['list_日期_个股'].append(self.日期)
            #     dic_每个个股数据记录[个股代号]['list_value_个股'].append(个股当天数据)
            #     self.code = 个股代号
            #
            #     选股结果 = 选股器(
            #         list_value=dic_每个个股数据记录[个股代号]['list_value_个股'],
            #         指针=i,
            #         连续下跌天数=self.选股条件_连续下跌天数
            #     ).选股()
            #
            #     if 选股结果:
            #         self.输出(ColorizedString(f"{self.code}满足选股条件", 'g'))
            #         # drawK线(dic_每个个股数据记录[self.code], self.code)
            #     else:
            #         """不满足选股条件，如果没有持仓就不关注了"""
            #         if self.code not in self.持仓:
            #             continue
            #         else:
            #             if self.持仓[self.code] == 0:
            #                 continue
            #     # '''已经有仓位的不再关注'''
            #     # if self.code in self.持仓:
            #     #     if self.持仓[self.code] != 0:
            #     #         continue
            #     开盘价 = 个股当天数据['开盘价']
            #     收盘价 = 个股当天数据['收盘价']
            #     最高价 = 个股当天数据['最高价']
            #     最低价 = 个股当天数据['最低价']
            #     买入价格, 买入手数 = self.买入条件(
            #         list_日期=dic_每个个股数据记录[个股代号]['list_日期_个股'],
            #         list_value=dic_每个个股数据记录[个股代号]['list_value_个股'],
            #         指针=i,
            #         买点=self.买点
            #     )
            #     if 买入手数 > 0:
            #         self.买入(买入手数, 买入价格)
            #     # 最低价格日期, 最低买入价格 = self.查看最低买入价格()
            #     list_卖出价格, list_卖出手数, list_对应日期 = self.卖出条件(
            #         list_value=dic_每个个股数据记录[个股代号]['list_value_个股'],
            #         指针=i,
            #         卖点=self.卖点
            #     )
            #     for k in range(0, len(list_卖出价格)):
            #         if list_卖出价格[k] > 0:
            #             卖出结果 = self.卖出(list_卖出手数[k], list_卖出价格[k])
            #             if 卖出结果:
            #                 del self.dic_买入历史[self.code][list_对应日期[k]]
            #                 # self.统计_list持仓时长.append()
            #                 self.统计_清仓次数 += 1
            """开始统计当天资产"""
            总资产 = self.资金
            for 个股代号, 持仓手数 in self.持仓.items():
                总资产 += 持仓手数 * 100 * data[self.日期][个股代号]['收盘价']
            self.list_资产走势.append(总资产)
            self.统计_list现金走势.append(self.资金)
            self.输出(ColorizedString(f"今天统计资金/总资产：{self.资金}/{总资产},持仓状态：{self.持仓}", 'b'))
        self.输出(f"初始资金：{self.初始资金},最终总资产：{self.list_资产走势[-1]}")
        print(f"买入股票次数：{self.统计_买入次数},股票清仓次数：{self.统计_清仓次数}")
        plt.plot(self.list_资产走势)
        # plt.plot(self.统计_list现金走势)
        plt.show()
        return self.list_资产走势, self.统计_list现金走势
        # plt.plot(self.list_资产走势)
        # plt.show()

    def 操作手数决定(self, 价格):
        手数 = 0
        while True:
            if 手数 * 100 * 价格 > self.set_一次操作资金:
                break
            else:
                手数 += 1
                continue
        return 手数

    def 查看最低买入价格(self):
        最低价格 = 99999
        最低价格日期 = None
        for 日期, value in self.dic_买入历史[self.code].items():
            买入价格 = value['价格']
            if 买入价格 <= 最低价格:
                最低价格 = 买入价格
                最低价格日期 = 日期
        return 最低价格日期, 最低价格

    def 买入(self, 手数, 价格):
        总价 = 手数 * 价格 * 100
        if 总价 > 20000:
            手续费 = 总价 * 0.00025
        else:
            手续费 = 5
        需花费 = 总价 + 手续费
        if self.资金 >= 需花费:
            if self.code not in self.持仓:
                self.持仓[self.code] = 手数
            else:
                self.持仓[self.code] += 手数
            self.资金 = round(self.资金 - 需花费, 2)
            if self.code not in self.dic_买入历史:
                self.dic_买入历史[self.code] = {}
            self.dic_买入历史[self.code] = {
                self.日期: {
                    "价格": 价格,
                    "手数": 手数
                }
            }
            self.统计_买入次数 += 1
            self.输出(ColorizedString(f"{self.日期}以{价格}买入股票{self.code}：{手数}手", 'r'))
            # self.输出(f"当前{self.code}持仓手数：{self.持仓[self.code]}，剩余资金：{self.资金}")
            return True
        else:
            return False

    def 卖出(self, 手数, 价格):
        总价 = 手数 * 价格 * 100
        if 总价 > 20000:
            手续费 = 总价 * 0.00025
        else:
            手续费 = 5
        可得到 = 总价 - 手续费
        if self.code not in self.持仓:
            self.持仓[self.code] = 0
        if self.持仓[self.code] >= 手数:
            self.持仓[self.code] -= 手数
            self.资金 = round(self.资金 + 可得到, 2)
            self.输出(ColorizedString(f"{self.日期}以{价格}卖出股票{self.code}：{手数}手", 'g'))
            # self.输出(f"当前{self.code}持仓手数：{self.持仓[self.code]}，剩余资金：{self.资金}")
            return True
            # self.输出(f"初始资金：{self.初始资金},最终资金：{self.资金},最终持仓:{self.持仓手数 * 100 * 价格}")
        else:
            return False

    def 买入条件(self, list_日期, list_value, 指针, 买点=0.99):
        """返回买入价格和买入手数"""
        if self.code in self.持仓:
            if self.持仓[self.code] != 0:
                return 0, 0
        if 指针 > 0:
            今日开盘价 = list_value[指针]['开盘价']
            今日收盘价 = list_value[指针]['收盘价']
            今日最低价 = list_value[指针]['最低价']
            今日最高价 = list_value[指针]['最高价']
            昨日开盘价 = list_value[指针 - 1]['开盘价']
            昨日收盘价 = list_value[指针 - 1]['收盘价']
            买入价格 = round(今日开盘价 * 买点, 2)
            买入手数 = self.操作手数决定(买入价格)
            if 今日最低价 <= 买入价格:
                return 买入价格, 买入手数
            else:
                return 0, 0
        else:
            return 0, 0

    def 卖出条件(self, list_value, 指针, 卖点=1.01):
        list_卖出价格 = []
        list_卖出手数 = []
        list_对应日期 = []
        if self.code not in self.持仓 or self.持仓[self.code] == []:
            return list_卖出价格, list_卖出手数, list_对应日期
        if 指针 > 0:
            今日开盘价 = list_value[指针]['开盘价']
            今日收盘价 = list_value[指针]['收盘价']
            今日最高价 = list_value[指针]['最高价']
            卖出价格 = 0
            卖出手数 = 0
            for 日期, value in self.dic_买入历史[self.code].items():
                if 日期 == self.日期:
                    """是当天的买入历史，不可以当天卖出"""
                    continue
                买入价格 = value['价格']
                买入手数 = value['手数']
                卖出价格 = round(买入价格 * 卖点, 2)
                if 今日最高价 >= 卖出价格:
                    利润 = self.利润计算(买入价格, 卖出价格, 买入手数)
                    self.输出(ColorizedString(
                        f"{self.日期}，{self.code}可以以{卖出价格}卖出,对应买入价格{买入价格}可以卖出,利润:{利润}", 'g'))
                    卖出手数 = 买入手数
                    list_卖出价格.append(卖出价格)
                    list_卖出手数.append(卖出手数)
                    list_对应日期.append(日期)
                else:
                    # self.输出(ColorizedString(f"{self.日期}最高价{今日最高价},对应买入价格{买入价格}，不可以卖出", 'y'))
                    pass
        return list_卖出价格, list_卖出手数, list_对应日期

    def 利润计算(self, 买入价, 卖出价, 手数):
        if 买入价 * 手数 * 100 < 20000:
            买入手续费 = 5
        else:
            买入手续费 = 买入价 * 手数 * 100 * 0.00025
        if 卖出价 * 手数 * 100 < 20000:
            卖出手续费 = 5
        else:
            卖出手续费 = 买入价 * 手数 * 100 * 0.00025
        利润 = 卖出价 * 手数 * 100 - 买入价 * 手数 * 100 - 买入手续费 - 卖出手续费
        return 利润

    def 输出(self, 字符串):
        if self.是否输出:
            print(字符串)


def 仿真控制器(a, b, c, d):
    开始时间 = time.time()
    dic_结果 = {}
    list_对应条件 = []
    list_results = []
    exe = ThreadPoolExecutor(max_workers=200)
    总仿真次数 = int((b - a) * (d - c))
    仿真次数 = 0
    股票数据 = 以时间为索引的股票数据
    for 千倍买点 in range(a, b):
        买点 = round(千倍买点 / 1000, 3)
        for 千倍卖点 in range(c, d):
            仿真次数 += 1
            卖点 = round(千倍卖点 / 1000, 3)
            仿真条件 = {
                '初始资金': 10000,
                '买点': 买点,
                '卖点': 卖点,
                '选股条件_连续下跌天数': 2
            }
            print(f"正在进行第{仿真次数}次仿真：{仿真条件}, 总需:{总仿真次数}次仿真")
            list_对应条件.append(json.dumps(仿真条件, ensure_ascii=False))
            list_results.append(exe.submit(手动仿真().start, 股票数据, 仿真条件, False))
            # dic_结果[f"买点{买点}卖点{卖点}"]=仿真(仿真条件,False).start()
    for i in range(0, len(list_对应条件)):
        dic_结果[list_对应条件[i]] = list_results[i].result()
    print(ColorizedString("全体仿真完毕，现在开始输出结果", "g"))
    最优结果, 最优条件 = 0, None
    for 仿真条件, 资产走势 in dic_结果.items():
        print(f"仿真条件：{仿真条件}，最终资产：{资产走势[-1]}")
        当前结果 = 资产走势[-1]
        if 当前结果 >= 最优结果:
            最优结果 = 当前结果
            最优条件 = 仿真条件
    print(ColorizedString(f"最优结果：{最优结果}，对应条件：{最优条件}", 'g'))
    plt.plot(dic_结果[最优条件])
    plt.show()
    print(f"仿真耗时：{time.time() - 开始时间}")


def 指导现实(最新交易日期="2023-01-30"):
    股票数据 = 全体股票数据
    list_满足要求的股票 = []
    for 个股代号, 个股数据 in 股票数据.items():
        list_日期_个股 = []
        list_value_个股 = []
        for 日期, value in 个股数据.items():
            list_日期_个股.append(日期)
            list_value_个股.append(value)
        if 日期 != 最新交易日期:
            print(ColorizedString(f"{个股代号}的最新交易日期与显示不符合：{日期}", 'r'))
        选股结果 = 选股器(
            list_value=list_value_个股,
            指针=0,
            连续下跌天数=4
        ).选股()
        if 选股结果:
            print(ColorizedString(f"代号：{个股代号}的股票是满足要求的", 'g'))
            list_满足要求的股票.append(个股代号)
        else:
            continue
    if len(list_满足要求的股票) == 0:
        print(ColorizedString(f"没有符合要求的股票", 'r'))


if __name__ == "__main__":
    # 仿真控制器(975,999,1010,1020)
    # 指导现实()
    手动仿真(
        股票数据=以时间为索引的股票数据
    ).start(
        仿真条件={
            '初始资金': 100000,
            '买点': 0.99,
            '卖点': 1.01,
            '选股条件_连续下跌天数': 4
        })
