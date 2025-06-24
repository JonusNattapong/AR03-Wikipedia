# client.py
import flwr as fl
from stable_baselines3 import PPO
import gym

class RLClient(fl.client.NumPyClient):
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.model = PPO("MlpPolicy", self.env)

    def get_parameters(self):
        return self.model.policy.state_dict()

    def set_parameters(self, parameters):
        self.model.policy.load_state_dict(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.learn(total_timesteps=1000)
        return self.get_parameters(), len(self.env.envs), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        reward = self.model.evaluate(self.env, n_eval_episodes=5)
        return {}, reward, {}

if __name__ == "__main__":
    fl.client.start_numpy_client("localhost:8080", client=RLClient())


# ให้หลาย client (เช่น mobile device) ฝึก policy RL บน data 
# เฉพาะตัว แล้วรวม gradient ที่ server แบบ federated