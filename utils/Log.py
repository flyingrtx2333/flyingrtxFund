import json
import time
import inspect
import threading
from datetime import datetime
import asyncio
from typing import Set, Optional
import weakref
from flask_socketio import emit

log_lock = threading.Lock()


def generate_time(timestamp=time.time()) -> str:
    """生成一个当前时间字符串，因为太长所以做成函数好了"""
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))


def colorized_string(content, color: str | None = None, background: str | None = None) -> str:
    """
    2023/1/11:美化输出，何乐不为
    :param background: 背景色
    :param content: 字符串
    :param color: 想要的颜色
    :return:
    """
    bg_colors = {
        "black": "40",
        "red": "41",
        "green": "42",
        "yellow": "43",
        "blue": "44",
        "magenta": "45",
        "cyan": "46",
        "white": "47",
    }
    foreground_str = ""
    background_str = ""
    if background:
        background_str = f"\033[{bg_colors[background]}m"
    if color == "red" or color == 'r':
        foreground_str = "\033[31m"
    elif color == "green" or color == 'g':
        foreground_str = "\033[32m"
    elif color == "yellow" or color == 'y':
        foreground_str = "\033[33m"
    elif color == "blue" or color == 'b':
        foreground_str = "\033[34m"
    elif color == "purple" or color == 'p':
        foreground_str = "\033[35m"
    elif color == "qing" or color == 'q':
        foreground_str = "\033[36m"
    elif color == "white" or color == 'w':
        foreground_str = "\033[37m"

    return foreground_str + background_str + str(content) + "\033[0m"

    # if color == "red" or color == 'r':
    #     return "\033[31m" + background_str + str(content) + "\033[0m"
    # elif color == "green" or color == 'g':
    #     return "\033[32m" + background_str + str(content) + "\033[0m"
    # elif color == "yellow" or color == 'y':
    #     return "\033[33m" + background_str + str(content) + "\033[0m"
    # elif color == "blue" or color == 'b':
    #     return "\033[34m" + background_str + str(content) + "\033[0m"
    # elif color == "purple" or color == 'p':
    #     return "\033[35m" + background_str + str(content) + "\033[0m"
    # elif color == "qing" or color == 'q':
    #     return "\033[36m" + background_str + str(content) + "\033[0m"
    # elif color == "white" or color == 'w':
    #     return "\033[37m" + background_str + str(content) + "\033[0m"


def colorized_string_table(content_table: str, 分隔符: str, color_list: list) -> str:
    """给一组需要多个颜色区分且有固定格式的字符串上色"""
    content_list = content_table.split(分隔符)
    content_result = ""
    for i in range(0, len(content_list)):
        color = color_list[i]
        content = content_list[i]
        content_color = colorized_string(content, color)
        if i == len(content_list) - 1:
            content_result = content_result + content_color
        else:
            content_result = content_result + content_color + 分隔符
    return content_result



class Log:
    @staticmethod
    def debug(content):
        caller_frame = inspect.stack()[1]
        try:
            module = inspect.getmodule(caller_frame[0])
            module_name = module.__name__
        except AttributeError:
            module_name = "未知模块名"
        line_number = caller_frame[2]
        print_string = f" {module_name}, Line {line_number} D {content}"
        
        # 控制台输出
        print(colorized_string(generate_time(time.time()), background='green') + 
              colorized_string(print_string, 'g'))
        
        try:
            # 发送到前端，指定命名空间为根路径
            emit('log', {
                'level': 'DEBUG',
                'timestamp': generate_time(time.time()),
                'content': print_string
            }, namespace='/', broadcast=True)
        except Exception as e:
            print(f"Socket emit error: {e}")

    @staticmethod
    def trace(content):
        caller_frame = inspect.stack()[1]
        try:
            module = inspect.getmodule(caller_frame[0])
            module_name = module.__name__
        except AttributeError:
            module_name = "未知模块名"
        line_number = caller_frame[2]
        print_string = f" {module_name}, Line {line_number} T {content}"
        
        # 控制台输出
        print(colorized_string(generate_time(time.time()), background='white') + 
              colorized_string(print_string, 'w'))
        
        try:
            emit('log', {
                'level': 'TRACE',
                'timestamp': generate_time(time.time()),
                'content': print_string
            }, namespace='/', broadcast=True)
        except Exception as e:
            print(f"Socket emit error: {e}")

    @staticmethod
    def info(content):
        caller_frame = inspect.stack()[1]
        try:
            module = inspect.getmodule(caller_frame[0])
            module_name = module.__name__
        except AttributeError:
            module_name = "未知模块名"
        line_number = caller_frame[2]
        print_string = f" {module_name}, Line {line_number} I {content}"
        
        # 控制台输出
        print(colorized_string(generate_time(time.time()), background='blue') + 
              colorized_string(print_string, 'b'))
        
        try:
            emit('log', {
                'level': 'INFO',
                'timestamp': generate_time(time.time()),
                'content': print_string
            }, namespace='/', broadcast=True)
        except Exception as e:
            print(f"Socket emit error: {e}")

    @staticmethod
    def warning(content):
        caller_frame = inspect.stack()[1]
        try:
            module = inspect.getmodule(caller_frame[0])
            module_name = module.__name__
        except AttributeError:
            module_name = "未知模块名"
        line_number = caller_frame[2]
        print_string = f" {module_name}, Line {line_number} W {content}"
        
        # 控制台输出
        print(colorized_string(generate_time(time.time()), background='yellow') + 
              colorized_string(print_string, 'y'))
        
        try:
            emit('log', {
                'level': 'WARNING',
                'timestamp': generate_time(time.time()),
                'content': print_string
            }, namespace='/', broadcast=True)
        except Exception as e:
            print(f"Socket emit error: {e}")

    @staticmethod
    def error(content):
        caller_frame = inspect.stack()[1]
        try:
            module = inspect.getmodule(caller_frame[0])
            module_name = module.__name__
        except AttributeError:
            module_name = "未知模块名"
        line_number = caller_frame[2]
        print_string = f" {module_name}, Line {line_number} E {content}"
        
        # 控制台输出
        print(colorized_string(generate_time(time.time()), background='red') + 
              colorized_string(print_string, 'r'))
        
        try:
            emit('log', {
                'level': 'ERROR',
                'timestamp': generate_time(time.time()),
                'content': print_string
            }, namespace='/', broadcast=True)
        except Exception as e:
            print(f"Socket emit error: {e}")

    @staticmethod
    def record(content):
        print_string = f"{generate_time(time.time())} record {content}"
        print(print_string)


def print_log(content, level, end="\n"):
    """
    日志打印于控制台上
    :param end:
    :param content: 打印内容
    :param level: 日志等级，分为I、W、E
    :return: 无
    """
    caller_frame = inspect.stack()[1]
    module = inspect.getmodule(caller_frame[0])
    module_name = module.__name__
    line_number = caller_frame[2]
    print_string = f"{generate_time(time.time())} {module_name},Line {line_number} {level} {content}"
    if level == 'I' or level == 'i':
        print(print_string, end=end)
    elif level == 'P' or level == 'p':
        print(colorized_string(print_string, 'g'), end=end)
    elif level == 'N' or level == 'n':
        print(colorized_string(print_string, 'b'), end=end)
    elif level == 'L' or level == 'l':
        print(colorized_string(print_string, 'p'), end=end)
    elif level == 'W' or level == 'w':
        print(colorized_string(print_string, 'y'), end=end)
    elif level == 'E' or level == 'e':
        print(colorized_string(print_string, 'r'), end=end)
    elif level == 'EE':
        print(colorized_string(print_string, 'r'), end=end)
        # with open(file_path_log_error, 'w', encoding='utf-8') as f:
        #     f.write(print_string)
        #     f.close()

def timestamp_to_datetime(date_timestamp):
    """
    将时间戳转换为时间字符串
    :param date_timestamp:
    :return:
    """
    date = time.strftime("%Y-%m-%d", time.localtime(date_timestamp))
    return date


def datetime_to_timestamp(date_str):
    """
    将"2011-03-22"类似字符串转换为时间戳，用于生成仿真数据时截取开始时间
    :param date_str: "2011-03-22"
    :return: timestamp
    """
    date_object = datetime.strptime(date_str, "%Y-%m-%d")

    # 使用timestamp()函数将datetime对象转换为时间戳
    timestamp = date_object.timestamp()

    return timestamp

if __name__ == '__main__':
    # 示例用法
    Log.debug('hi')
