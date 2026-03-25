# experiments/train_rl.py

import os
import pandas as pd
from config import CONFIG
from utils.data_utils import ensure_dirs
from env.credit_env import CreditEnv
from models.rl_agent import PPOAgent
from stable_baselines3.common.monitor import Monitor
from utils.plotting import plot_training_rewards

def run_rl_training():
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
    
    log_dir = os.path.join(CONFIG["results_path"], "ppo_logs")
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    agent = PPOAgent(env=env, seed=CONFIG["random_seed"])
    agent.train(total_timesteps=CONFIG["ppo_total_timesteps"])
    agent.save(os.path.join(CONFIG["results_path"], "ppo_credit_agent"))
    print("PPO训练完成并已保存。")
    
    # 绘制 图 5-2 PPO训练奖励变化曲线
    plot_training_rewards(os.path.join(log_dir, "monitor.csv"), os.path.join(CONFIG["results_path"], "ppo_training_rewards.png"))