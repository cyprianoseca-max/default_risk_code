# config.py

CONFIG = {
    # 数据参数
    "random_seed": 42,
    "n_samples": 5000,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # 风险模型参数
    "tabnet_epochs": 50,
    "tabnet_batch_size": 256,
    "lr_max_iter": 1000,
    "rf_n_estimators": 200,

    # 授信环境参数
    "initial_credit_limit": 10000,
    "min_credit_limit": 1000,
    "max_credit_limit": 100000,
    "adjust_ratio": 0.1,
    "max_steps": 50,
    "interest_rate": 0.01,
    "lgd": 0.3,
    "adjust_cost": 5,
    "high_risk_threshold": 0.5,

    # PPO参数
    "ppo_total_timesteps": 20000,

    # 输出路径
    "raw_data_path": "data/raw/simulated_credit_data.csv",
    "processed_data_path": "data/processed/processed_credit_data.csv",
    "results_path": "data/results/"
}