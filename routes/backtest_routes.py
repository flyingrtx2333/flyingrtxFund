import traceback
from flask import Blueprint, render_template, request, jsonify
from utils.bk_manager import BKManager
from utils.backtest_engine import BacktestEngine
from utils.strategy import STRATEGY_REGISTRY

# 创建Blueprint
backtest_bp = Blueprint('backtest', __name__)

@backtest_bp.route('/backtest')
def backtest():
    bks = BKManager.get_available_bks()
    return render_template('backtest.html', 
                         bks=bks,
                         strategies=STRATEGY_REGISTRY)

@backtest_bp.route('/api/backtest', methods=['POST'])
def run_backtest():
    data = request.get_json()
    bk_codes = data.get('bk_codes', [])
    strategy_id = data.get('strategy_id')
    initial_money = data.get('initial_money', 1000000)
    strategy_params = data.get('strategy_params', {})
    
    if not bk_codes:
        return jsonify({
            'status': 'error',
            'message': '请选择至少一个板块',
            'data': None
        })
    
    if strategy_id not in STRATEGY_REGISTRY:
        return jsonify({
            'status': 'error',
            'message': '无效的策略选择',
            'data': None
        })
    
    strategy = STRATEGY_REGISTRY[strategy_id]
    
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
    try:
        engine = BacktestEngine(
            bk_codes=bk_codes,
            strategy=strategy,
            initial_money=int(initial_money),
            parameters=parameters,
            days_limit=5000,
            start_date='2020-05-23'
        )
        result = engine.run()
        return jsonify(result) 
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'data': None
        })
