import os
import pandas as pd
import numpy as np
from config import CONFIG
from env.credit_env import CreditEnv
from models.rl_agent import PPOAgent
from experiments.baseline_strategy import conservative_strategy, rule_based_strategy
from utils.plotting import plot_strategy_comparison
from utils.data_utils import ensure_dirs

def evaluate_strategy(env, agent=None, strategy_func=None, episodes=100):
    total_reward = 0
    total_default_loss = 0
    total_interest_income = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if agent is not None:
                action = agent.predict(obs)
            else:
                action = strategy_func(obs)
            
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            total_default_loss += info["default_loss"]
            total_interest_income += info["interest_income"]

    return {
        "total_reward": total_reward / episodes,
        "total_default_loss": total_default_loss / episodes,
        "total_interest_income": total_interest_income / episodes
    }

def run_full_evaluation():
    ensure_dirs()
    df = pd.read_csv(CONFIG["raw_data_path"])
    
    env = CreditEnv(
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
        print(f"Warning: RL Agent not found at {agent_path}. Skipping RL evaluation.")
        agent = None

    results = {}
    
    print("评估保守策略...")
    results["Conservative"] = evaluate_strategy(env, strategy_func=conservative_strategy)
    
    print("评估基于规则的策略...")
    results["Rule-based"] = evaluate_strategy(env, strategy_func=rule_based_strategy)
    
    if agent is not None:
        print("评估强化学习策略...")
        results["RL-PPO"] = evaluate_strategy(env, agent=agent)

    print("\n策略评估结果 (平均每Episode):")
    for k, v in results.items():
        print(f"{k}: {v}")

    plot_strategy_comparison(results, os.path.join(CONFIG["results_path"], "strategy_comparison.png"))
    print("策略对比图已保存。")
