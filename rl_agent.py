# models/rl_agent.py

from stable_baselines3 import PPO

class PPOAgent:
    def __init__(self, env, seed=42):
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=seed
        )

    def train(self, total_timesteps=20000):
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def save(self, path):
        self.model.save(path)

    def load(self, path, env=None):
        self.model = PPO.load(path, env=env)
        return self