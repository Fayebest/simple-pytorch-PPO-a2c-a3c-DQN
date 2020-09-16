from simPPO import buffer
from simPPO import model
from simPPO import ppo
import torch
import numpy as np
import gym
import time

def main():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    solved_reward = 230  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 50000  # max training episodes
    max_timesteps = 300  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 2000  # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None

    batch_size = 64
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = buffer.Memory()
    ppo_policy = ppo.PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    reward_arr = []
    # training loop
    st = time.time()
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        tmpreward = 0
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = ppo_policy.policy.act(state, memory)
            state, reward, done, _ = env.step(action)
            tmpreward += reward
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            # ppo.update(memory)
            # update if its time
            if timestep % update_timestep == 0:
                ppo_policy.update(memory,batch_size)
                memory.reset_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            print("end time", time.time() - st )
            torch.save(ppo_policy.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

        reward_arr.append(tmpreward)
        np.save('reward', reward_arr)



if __name__ == '__main__':
    main()