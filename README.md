# 信用风险评估与动态授信策略研究代码复现

本项目为论文《信用风险评估与动态授信策略研究》的代码复现实现。整个项目包含数据生成、风险模型训练与评估、基于强化学习的动态授信策略以及宏观经济压力测试的完整流程。所有步骤高度集成，一键即可运行出论文中展示的所有图表与数据。

## 项目目录结构

```text
信用风险评估/
├── appendix_code/                # 附录代码
│   └── full_demo.py              # 论文中提供的精简版demo代码
├── data/                         # 数据目录
│   ├── raw/                      # 存放生成的原始模拟数据 (simulated_credit_data.csv)
│   ├── processed/                # 存放预处理后的数据
│   └── results/                  # 存放所有模型产出、图表以及表格数据
├── env/                          # 强化学习环境
│   └── credit_env.py             # 动态授信环境定义 (兼容 gymnasium API)
├── experiments/                  # 核心实验流程
│   ├── baseline_strategy.py      # 定义基准策略 (保守策略与基于规则策略)
│   ├── evaluate.py               # 策略评估逻辑
│   ├── stress_test.py            # 宏观经济压力测试逻辑
│   ├── train_risk_model.py       # 风险预测模型训练 (逻辑回归、随机森林、TabNet)
│   └── train_rl.py               # 强化学习PPO模型训练
├── models/                       # 模型定义
│   ├── risk_models.py            # 传统机器学习风险预测模型
│   ├── rl_agent.py               # 强化学习PPO Agent封装
│   └── tabnet_model.py           # TabNet深度学习模型
├── utils/                        # 工具与辅助模块
│   ├── data_utils.py             # 数据生成与预处理工具
│   ├── explain.py                # 特征解释性相关
│   ├── metrics.py                # 评估指标 (AUC, KS等)计算
│   └── plotting.py               # 所有论文所需图表的可视化绘制函数
├── config.py                     # 全局参数配置文件
├── main.py                       # 项目一键运行入口
└── requirements.txt              # 依赖库清单
```

## 环境安装与配置

本项目基于 Python 开发，推荐使用虚拟环境运行。

1. **创建并激活虚拟环境** (以 PowerShell 为例)
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
   *注意：依赖中包含 `seaborn` 用于图表绘制，以及 `gymnasium` 适配最新版的强化学习库。*

## 运行指南

只需执行项目根目录下的 `main.py` 即可按顺序运行全部实验：

```bash
python main.py
```

运行过程中，控制台会输出模型评估结果以及训练进度，流程分为以下四步：
1. **第一步：训练风险预测模型** (生成数据，训练逻辑回归、随机森林和TabNet模型并输出指标)。
2. **第二步：训练动态授信强化学习策略** (使用 PPO 算法进行策略学习)。
3. **第三步：评估模型与策略** (将 PPO 策略与保守策略、基于规则策略进行收益对决)。
4. **第四步：压力测试** (模拟宏观经济环境恶化，评估各策略的抗压能力，并输出最终对比表格)。

## 输出结果与图表说明

运行完成后，所有的输出文件都将保存在 `data/results/` 目录下。这些图表完全对应论文中的相关插图：

1. **default_distribution.png** (对应论文 图4-2)：违约标签分布图，展示样本中违约与非违约用户的数量比例。
2. **feature_distributions.png** (对应论文 图4-3)：核心特征分布图，展示收入、负债率、额度使用率等特征的核密度分布。
3. **roc_curves_comparison.png** (对应论文 图5-1)：风险预测模型ROC曲线对比图，包含逻辑回归、随机森林和TabNet的AUC对比。
4. **ppo_training_rewards.png** (对应论文 图5-2)：PPO训练奖励变化曲线，展示强化学习在训练过程中的奖励收敛情况。
5. **strategy_comparison.png** (对应论文 图5-3)：动态授信策略收益与损失对比图，展示在正常情况下不同策略的奖励与违约损失。
6. **tabnet_feature_importances.png** (对应论文 图5-4)：TabNet特征重要性排序图，以条形图形式直观展示各特征的决策权重。
7. **stress_test_comparison.png** (对应论文 图5-5)：压力测试结果对比图，展示正常场景与压力场景下各策略表现。
8. **stress_test_table_5_4.csv** (对应论文 表5-4)：压力测试结果对比表数据，包含“常规场景”与“压力场景”下的平均累计奖励、平均违约损失和净收益。可以直接使用 Excel 打开，或复制到 Word 论文中。

除此之外，该目录还会保存模型训练的参数权重 (`ppo_credit_agent.zip`) 和指标详情 (`risk_model_results.json`)。

## 许可证
本项目代码仅供学术研究和学习交流使用。
