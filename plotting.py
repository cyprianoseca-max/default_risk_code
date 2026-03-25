import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

def plot_default_distribution(df, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Default', data=df)
    plt.title('违约标签分布图')
    plt.xlabel('是否违约 (0: 否, 1: 是)')
    plt.ylabel('数量')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_distributions(df, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    features = ['Income', 'Debt_Ratio', 'Utilization', 'Overdue_Count', 'Recent_Inquiries']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.histplot(df[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'{feature} 分布')
    axes[-1].axis('off') 
    plt.suptitle('核心特征分布图', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(roc_data, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 6))
    for model_name, (fpr, tpr, auc_score) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('风险预测模型ROC曲线对比图')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_rewards(log_file, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    try:
        import pandas as pd
        df = pd.read_csv(log_file, skiprows=1)
        # df contains 'r', 'l', 't' -> r is reward
        rewards = df['r'].rolling(window=10).mean() # smoothed
        plt.figure(figsize=(10, 5))
        plt.plot(df['l'].cumsum(), df['r'], alpha=0.3, color='blue', label='原始奖励')
        plt.plot(df['l'].cumsum(), rewards, color='red', label='滑动平均奖励')
        plt.xlabel('Timesteps')
        plt.ylabel('Rewards')
        plt.title('PPO训练奖励变化曲线')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Plotting rewards failed: {e}")

def plot_stress_test_comparison_chart(normal_results, stress_results, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    strategies = list(normal_results.keys())
    x = np.arange(len(strategies))
    width = 0.35
    
    normal_rewards = [normal_results[s]["total_reward"] for s in strategies]
    stress_rewards = [stress_results[s]["total_reward"] for s in strategies]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, normal_rewards, width, label='正常场景总奖励')
    ax.bar(x + width/2, stress_rewards, width, label='压力场景总奖励')
    
    ax.set_ylabel('金额 (Amount)')
    ax.set_title('压力测试结果对比图')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importances(feature_names, importances, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 按重要性排序
    idx = np.argsort(importances)
    sorted_features = [feature_names[i] for i in idx]
    sorted_importances = importances[idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, color='skyblue')
    plt.xlabel('特征重要性 (Feature Importance)')
    plt.title('TabNet 特征重要性分析')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_strategy_comparison(results, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    strategies = list(results.keys())
    rewards = [results[s]["total_reward"] for s in strategies]
    defaults = [results[s]["total_default_loss"] for s in strategies]
    incomes = [results[s]["total_interest_income"] for s in strategies]

    x = range(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width for i in x], rewards, width, label='总奖励 (Total Reward)')
    ax.bar(x, incomes, width, label='利息收入 (Interest Income)')
    ax.bar([i + width for i in x], defaults, width, label='违约损失 (Default Loss)')

    ax.set_ylabel('金额 (Amount)')
    ax.set_title('动态授信策略收益与损失对比图')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
