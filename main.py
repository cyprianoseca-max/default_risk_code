# main.py

from experiments.train_risk_model import run_risk_model_experiment
from experiments.train_rl import run_rl_training
from experiments.evaluate import run_full_evaluation
from experiments.stress_test import run_stress_test

def main():
    print("========== 第一步：训练风险预测模型 ==========")
    run_risk_model_experiment()

    print("\n========== 第二步：训练动态授信强化学习策略 ==========")
    run_rl_training()

    print("\n========== 第三步：评估模型与策略 ==========")
    run_full_evaluation()

    print("\n========== 第四步：压力测试 ==========")
    run_stress_test()

    print("\n全部流程运行完成。")

if __name__ == "__main__":
    main()