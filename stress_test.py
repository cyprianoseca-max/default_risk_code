import os
import pandas as pd
import numpy as np
from config import CONFIG
from env.credit_env import CreditEnv
from models.rl_agent import PPOAgent
from experiments.baseline_strategy import conservative_strategy, rule_based_strategy
from experiments.evaluate import evaluate_strategy
from utils.plotting import plot_strategy_comparison, plot_stress_test_comparison_chart
from utils.data_utils import ensure_dirs

def apply_macro_shock(df):
    """
    模拟宏观经济冲击（如失业率上升、收入下降、违约率上升）
    """
    shocked_df = df.copy()
    # 收入下降 20%
    shocked_df["Income"] = shocked_df["Income"] * 0.8
    # 负债率上升
    shocked_df["Debt_Ratio"] = np.clip(shocked_df["Debt_Ratio"] + 0.2, 0, 1)
    # 逾期次数增加
    shocked_df["Overdue_Count"] = shocked_df["Overdue_Count"] + np.random.poisson(1, size=len(df))
    # 额度使用率上升
    shocked_df["Utilization"] = np.clip(shocked_df["Utilization"] + 0.3, 0, 1)
    return shocked_df

def run_stress_test():
    ensure_dirs()
    df = pd.read_csv(CONFIG["raw_data_path"])
    
    # 应用压力情景
    shocked_df = apply_macro_shock(df)
    
    env = CreditEnv(
        df=shocked_df,
        risk_model=None,
        initial_credit_limit=CONFIG["initial_credit_limit"],
        min_credit_limit=CONFIG["min_credit_limit"],
        max_credit_limit=CONFIG["max_credit_limit"],
        adjust_ratio=CONFIG["adjust_ratio"],
        max_steps=CONFIG["max_steps"],
        interest_rate=CONFIG["interest_rate"],
        lgd=CONFIG["lgd"],
        adjust_cost=CONFIG["adjust_cost"],
        high_risk_threshold=CONFIG["high_risk_threshold"]
    )

    # 正常情景环境
    normal_env = CreditEnv(
        df=df,
        risk_model=None,
        initial_credit_limit=CONFIG["initial_credit_limit"],
        min_credit_limit=CONFIG["min_credit_limit"],
        max_credit_limit=CONFIG["max_credit_limit"],
        adjust_ratio=CONFIG["adjust_ratio"],
        max_steps=CONFIG["max_steps"],
        interest_rate=CONFIG["interest_rate"],
        lgd=CONFIG["lgd"],
        adjust_cost=CONFIG["adjust_cost"],
        high_risk_threshold=CONFIG["high_risk_threshold"]
    )
    
    agent_path = os.path.join(CONFIG["results_path"], "ppo_credit_agent.zip")
    if os.path.exists(agent_path):
        agent = PPOAgent(env=env).load(agent_path, env=env)
    else:
        print(f"Warning: RL Agent not found at {agent_path}. Skipping RL stress test.")
        agent = None

    normal_results = {}
    print("【正常场景】评估保守策略...")
    normal_results["Conservative"] = evaluate_strategy(normal_env, strategy_func=conservative_strategy)
    print("【正常场景】评估基于规则的策略...")
    normal_results["Rule-based"] = evaluate_strategy(normal_env, strategy_func=rule_based_strategy)
    if agent is not None:
        print("【正常场景】评估强化学习策略...")
        normal_results["RL-PPO"] = evaluate_strategy(normal_env, agent=agent)

    stress_results = {}
    
    print("【压力测试】评估保守策略...")
    stress_results["Conservative"] = evaluate_strategy(env, strategy_func=conservative_strategy)
    
    print("【压力测试】评估基于规则的策略...")
    stress_results["Rule-based"] = evaluate_strategy(env, strategy_func=rule_based_strategy)
    
    if agent is not None:
        print("【压力测试】评估强化学习策略...")
        stress_results["RL-PPO"] = evaluate_strategy(env, agent=agent)

    print("\n正常场景评估结果 (平均每Episode):")
    for k, v in normal_results.items():
        print(f"{k}: {v}")

    print("\n压力测试评估结果 (平均每Episode):")
    for k, v in stress_results.items():
        print(f"{k}: {v}")

    # 将正常场景和压力场景的结果保存为CSV表格，方便写论文使用
    table_data = []
    # 这里以 RL-PPO 策略的结果为例（如果有），如果没有则用基于规则的策略
    best_strategy = "RL-PPO" if agent is not None else "Rule-based"
    
    table_data.append({
        "场景": "常规场景",
        "平均累计奖励": normal_results[best_strategy]["total_reward"],
        "平均违约损失": normal_results[best_strategy]["total_default_loss"],
        "净收益": normal_results[best_strategy]["total_reward"]  # 在代码逻辑中，total_reward 即代表净收益 (利息 - 损失 - 成本)
    })
    
    table_data.append({
        "场景": "压力场景",
        "平均累计奖励": stress_results[best_strategy]["total_reward"],
        "平均违约损失": stress_results[best_strategy]["total_default_loss"],
        "净收益": stress_results[best_strategy]["total_reward"]
    })
    
    df_table = pd.DataFrame(table_data)
    csv_path = os.path.join(CONFIG["results_path"], "stress_test_table_5_4.csv")
    df_table.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n表 5-4 压力测试结果对比表 数据已保存至: {csv_path}")
    print(df_table)

    plot_stress_test_comparison_chart(
        normal_results, 
        stress_results, 
        os.path.join(CONFIG["results_path"], "stress_test_comparison.png")
    )
    print("压力测试对比图已保存。")
