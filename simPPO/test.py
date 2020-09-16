import gym
import time
env = gym.make('CartPole-v0')
# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)
env.reset()
for _ in range(1000):
    env.render()
    env.step(0) # take a random action
    time.sleep(0.1)
env.close()