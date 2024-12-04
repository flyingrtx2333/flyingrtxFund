"""
技术分析因子模块
包含各类技术指标的因子判断方法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Factors:
    """
    技术分析因子集合
    包含布林带、EMA、KDJ、RSI、MACD等技术指标的判断方法
    """
    
    @staticmethod
    def 拟合_正弦函数(list_y, 拟合度要求=0.6, 可视化结果=False):
        """
        尝试用正弦函数拟合股票，当目前趋势为波谷姿态时返回真
        :param list_y: 收盘价列表
        :param 拟合度要求: R²最小要求
        :param 可视化结果: 是否显示拟合结果图像
        :return: 是否符合,r2,omega,phi,period,A
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
        list_x = np.arange(n)
        guess_amplitude = (np.max(list_y) - np.min(list_y)) / 2
        guess_offset = np.mean(list_y)
        guess_frequency = 2 * np.pi / n
        guess_phase = 0
        guess = [guess_amplitude, guess_frequency, guess_phase, guess_offset]

        try:
            popt, _ = curve_fit(sinusoidal, list_x, list_y, p0=guess, maxfev=3000)
            A, omega, phi, offset = popt
            
            if phi >= 0:
                raise Exception("相位不符合要求")

            y_fit = sinusoidal(list_x, A, omega, phi, offset)
            r2 = r_squared(list_y, y_fit)

            if r2 < 拟合度要求:
                raise Exception("拟合度不符合要求")

            at_trough = is_trough_at_end(y_fit)
            period = 2 * np.pi / omega
            is_complete_cycle = period <= n

            if r2 >= 拟合度要求 and at_trough:
                if 可视化结果:
                    拟合结果可视化(phi=phi)
                return True, r2, omega, phi, period, A
            else:
                return False, r2, omega, phi, period, A
        except Exception as e:
            return False, None, None, None, None, None

    @staticmethod
    def 布林经典(list_value):
        """
        股价上穿布林上轨
        """
        value_today = list_value[-1]
        value_yesterday = list_value[-2]
        if (value_today['收盘价'] > value_today['booling_upper'] and 
            value_yesterday['收盘价'] < value_yesterday['booling_upper']):
            return True
        return False

    @staticmethod
    def 布林上穿中线(list_value):
        """
        股价上穿布林中轨
        """
        value_today = list_value[-1]
        value_yesterday = list_value[-2]
        if (value_today['收盘价'] > value_today['booling_mid'] and 
            value_yesterday['收盘价'] < value_yesterday['booling_mid']):
            return True
        return False

    @staticmethod
    def 布林上穿底部(list_value):
        """
        股价上穿布林下轨
        """
        value_today = list_value[-1]
        value_yesterday = list_value[-2]
        if (value_today['收盘价'] > value_today['booling_lower'] and 
            value_yesterday['收盘价'] < value_yesterday['booling_lower']):
            return True
        return False

    @staticmethod
    def 布林下穿底部(list_value):
        """
        股价下穿布林下轨
        """
        value_today = list_value[-1]
        value_yesterday = list_value[-2]
        if (value_today['收盘价'] <= value_today['booling_lower'] and 
            value_yesterday['收盘价'] > value_yesterday['booling_lower']):
            return True
        return False

    @staticmethod
    def 布林近期触底(list_value, window=7):
        """
        近期是否触及布林下轨
        """
        result = False
        start_pos = len(list_value) - window
        for i in range(start_pos, len(list_value)):
            value_today = list_value[i]
            if value_today['最低价'] <= value_today['booling_lower']:
                result = True
        return result

    @staticmethod
    def RSI过低(list_value, 阈值=30):
        """
        RSI是否低于阈值
        """
        if list_value[-1]['rsi'] < 阈值:
            return True
        return False

    @staticmethod
    def RSI大于50(list_value):
        """
        RSI是否大于50,确保股票动能充足
        """
        if list_value[-1]['rsi'] >= 50:
            return True
        return False

    @staticmethod
    def EMA金叉(list_value):
        """
        判别股价短期趋势,EMA快线上穿慢线
        """
        if (list_value[-2]['short_ema'] < list_value[-2]['long_ema'] and 
            list_value[-1]['short_ema'] >= list_value[-1]['long_ema']):
            return True
        return False

    @staticmethod
    def EMA死叉(list_value):
        """
        判别股价短期趋势,EMA快线下穿慢线
        """
        if (list_value[-2]['short_ema'] >= list_value[-2]['long_ema'] and 
            list_value[-1]['short_ema'] < list_value[-1]['long_ema']):
            return True
        return False

    @staticmethod
    def EMA死叉_近期出现(list_value):
        """
        近期是否出现EMA死叉
        """
        start_pos = len(list_value) - 30
        for i in range(start_pos, len(list_value)):
            if (list_value[i - 1]['short_ema'] >= list_value[i - 1]['long_ema'] and 
                list_value[i]['short_ema'] < list_value[i]['long_ema']):
                return True
        return False

    @staticmethod
    def EMA200上涨(list_value):
        """
        判别股价长期趋势,EMA200连续上涨
        """
        if (list_value[-1]['ema_200'] > list_value[-2]['ema_200'] > 
            list_value[-3]['ema_200'] > list_value[-4]['ema_200']):
            return True
        return False

    @staticmethod
    def EMA200_近期均大于(list_value):
        """
        近期收盘价是否都在EMA200之上
        """
        result = True
        for value in list_value[-30:]:
            if value['收盘价'] <= value['ema_200']:
                result = False
        return result

    @staticmethod
    def EMA26_近期上涨(list_value):
        """
        EMA26是否高于30天前
        """
        if list_value[-1]['long_ema'] >= list_value[-30]['long_ema']:
            return True
        return False

    @staticmethod
    def KDJ金叉(list_value):
        """
        KDJ指标K线上穿D线
        """
        if (list_value[-2]['kdj_k'] < list_value[-2]['kdj_d'] and 
            list_value[-1]['kdj_k'] >= list_value[-1]['kdj_d']):
            return True
        return False

    @staticmethod
    def MACD金叉(list_value):
        """
        MACD金叉,MACD线上穿信号线
        """
        if (list_value[-2]['macd_line'] < list_value[-2]['signal_line'] and 
            list_value[-1]['macd_line'] >= list_value[-1]['signal_line']):
            return True
        return False

    @staticmethod
    def MACD柱形图转正(list_value):
        """
        MACD柱形图由负转正
        """
        if (list_value[-2]['macd_histogram'] < 0 and 
            list_value[-1]['macd_histogram'] >= 0):
            return True
        return False

    @staticmethod
    def MACD趋势(list_value, window=5):
        """
        MACD柱形图连续上升趋势
        """
        if len(list_value) < window + 1:
            return False
            
        for i in range(1, window):
            if list_value[-i]['macd_histogram'] <= list_value[-(i+1)]['macd_histogram']:
                return False
        return True

    @staticmethod
    def MACD零轴上方(list_value, window=3):
        """
        MACD在零轴上方运行
        """
        if len(list_value) < window:
            return False
            
        for i in range(window):
            if list_value[-i-1]['macd_line'] <= 0:
                return False
        return True

    @staticmethod
    def 换手率(list_value, 阈值=0.5):
        """
        换手率是否大于阈值
        """
        if list_value[-1]['换手率'] >= 阈值:
            return True
        return False

    @staticmethod
    def 海龟因子(list_value):
        """
        海龟交易法则,突破20日新高
        """
        if list_value[-1]['收盘价'] > max(x['最高价'] for x in list_value[-20:-1]):
            return True
        return False 