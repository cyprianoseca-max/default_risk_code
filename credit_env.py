# env/credit_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class CreditEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        risk_model=None,
        initial_credit_limit=10000,
        min_credit_limit=1000,
        max_credit_limit=100000,
        adjust_ratio=0.1,
        max_steps=50,
        interest_rate=0.01,
        lgd=0.3,
        adjust_cost=5,
        high_risk_threshold=0.5
    ):
        super(CreditEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.risk_model = risk_model

        self.initial_credit_limit = initial_credit_limit
        self.min_credit_limit = min_credit_limit
        self.max_credit_limit = max_credit_limit
        self.adjust_ratio = adjust_ratio
        self.max_steps = max_steps
        self.interest_rate = interest_rate
        self.lgd = lgd
        self.adjust_cost = adjust_cost
        self.high_risk_threshold = high_risk_threshold

        # 动作：0=降额，1=维持，2=提额
        self.action_space = spaces.Discrete(3)

        # 状态：pd, utilization, debt_ratio, overdue_count, recent_inquiries, credit_limit
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.current_step = 0
        self.customer = None
        self.done = False

    def _sample_customer(self):
        customer = self.df.sample(1).iloc[0].copy()
        customer["CreditLimit"] = self.initial_credit_limit
        return customer

    def _estimate_pd(self, customer):
        if self.risk_model is None:
            pd_prob = min(
                1.0,
                0.1 +
                0.5 * customer["Debt_Ratio"] +
                0.2 * customer["Utilization"] +
                0.05 * customer["Overdue_Count"]
            )
            return float(np.clip(pd_prob, 0, 1))
        else:
            x = np.array([[
                customer["Income"],
                customer["Debt_Ratio"],
                customer["Credit_History"],
                customer["Age"],
                customer["Recent_Inquiries"],
                customer["Utilization"],
                customer["Overdue_Count"]
            ]])
            return float(self.risk_model.predict_proba(x)[0])

    def _get_obs(self):
        pd_prob = self._estimate_pd(self.customer)
        return np.array([
            pd_prob,
            self.customer["Utilization"],
            self.customer["Debt_Ratio"],
            self.customer["Overdue_Count"],
            self.customer["Recent_Inquiries"],
            self.customer["CreditLimit"]
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.customer = self._sample_customer()
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        old_limit = self.customer["CreditLimit"]
        pd_before = self._estimate_pd(self.customer)

        # 风控约束：高风险客户禁止提额
        if pd_before > self.high_risk_threshold and action == 2:
            action = 1

        # 执行动作
        if action == 0:
            self.customer["CreditLimit"] *= (1 - self.adjust_ratio)
        elif action == 2:
            self.customer["CreditLimit"] *= (1 + self.adjust_ratio)

        self.customer["CreditLimit"] = np.clip(
            self.customer["CreditLimit"],
            self.min_credit_limit,
            self.max_credit_limit
        )

        # 状态演化：简化版
        self.customer["Utilization"] = np.clip(
            self.customer["Utilization"] + np.random.normal(0, 0.05),
            0, 1
        )
        self.customer["Debt_Ratio"] = np.clip(
            self.customer["Debt_Ratio"] + np.random.normal(0, 0.03),
            0, 1
        )
        self.customer["Recent_Inquiries"] = max(
            0, int(self.customer["Recent_Inquiries"] + np.random.choice([-1, 0, 1]))
        )
        self.customer["Overdue_Count"] = max(
            0, int(self.customer["Overdue_Count"] + np.random.choice([0, 0, 1]))
        )

        pd_after = self._estimate_pd(self.customer)

        # 奖励函数
        interest_income = (
            self.customer["Utilization"] *
            self.customer["CreditLimit"] *
            self.interest_rate
        )
        ead = self.customer["CreditLimit"]
        default_loss = pd_after * self.lgd * ead
        action_cost = self.adjust_cost if action != 1 else 0

        reward = interest_income - default_loss - action_cost

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        obs = self._get_obs()
        info = {
            "interest_income": interest_income,
            "default_loss": default_loss,
            "pd_before": pd_before,
            "pd_after": pd_after,
            "old_limit": old_limit,
            "new_limit": self.customer["CreditLimit"],
            "action": action
        }
        return obs, reward, self.done, False, info

    def render(self, mode="human"):
        print(f"Step={self.current_step}, Customer={self.customer.to_dict()}")