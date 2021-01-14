from DQN import Dqn, Replay_Buffer
import torch
import numpy as np
import gym
import time

def train():
    ############## Hyperparameters ##############
    env_name = "CartPole-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    render = False
    solved_reward = 200  # stop training if avg_reward > solved_reward
    log_interval = 20  # print and save reward in the interval
    max_episodes = 50000  # max training episodes
    max_timesteps = 300  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer

    lr = 1e-3

    gamma = 0.9  # discount factor

    random_seed = 2000

    batch_size = 64
    collect_per_step = 100      #the number of frames the collector would collect before the network update
    update_per_step = 10      #the number of times the policy network would be updated after frames be collected.
    steps_per_epoch = 1000       #一个epoch有多少steps
    update_target_nerwork = 320
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)


    memory = Replay_Buffer.Memory(20000)
    timestep = 0
    dqn_policy = Dqn.dqn(state_dim,action_dim,n_latent_var,gamma,lr)
    reward_arr = []
    round_reward = 0
    state = env.reset()
    for i_episode in range(1, max_episodes+1):
        steps_episode = 0
        while(steps_episode < steps_per_epoch):

            for t in range(max_timesteps):

                action = dqn_policy.act(state, timestep)

                n_state, reward, done, _ = env.step(action.item())

                if done:
                    n_state = None


                memory.push(state,action,reward, n_state)

                if (timestep+1) % collect_per_step == 0:
                    for _ in range(update_per_step):
                        dqn_policy.learn(memory, batch_size)

                round_reward += reward
                state = n_state
                timestep += 1
                steps_episode += 1

                if (timestep+1) % update_target_nerwork == 0:
                    dqn_policy.target_net.load_state_dict(dqn_policy.policy_net.state_dict())


                if (timestep + 1) % log_interval == 0:
                    np.save("reward", round_reward)

                if render:
                    env.render()

                if done:
                    print(round_reward)
                    state = env.reset()

                    reward_arr.append(round_reward)
                    round_reward = 0
                    break

if __name__ == "__main__":
    train()