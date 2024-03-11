from algorithms.base_aglo import BaseAlgo
from policy.base_policy import BasePolicy
import gymnasium as gym
import time

def run_learning_loop(env : gym.Env, policy: BasePolicy, rl_algo: BaseAlgo, episodes=100, max_steps_per_episode=100):
    solved_tims = 0
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        action_sequence = []
        for step in range(max_steps_per_episode):
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            rl_algo.update(state, action, reward, next_state, done)
            action_sequence.append(action)
            state = next_state
            total_reward += reward
            
            if done:
                solved_tims += 1
                msg = f"\rSolved in {step+1} steps! Total reward: {total_reward}"
                print("\r" +" "*500, end="", flush=True)
                print(msg, end="", flush=True)
                # Pause for a while
                time.sleep(1)
                print("\r" +" "*500, end="", flush=True)
                action_seq_translated = [env.action_space.actions[action] for action in action_sequence]
                msg = f"\rAction sequence length: {len(action_seq_translated)}"
                print(msg, end="", flush=True)
                # Pause for a while
                time.sleep(1)
                print("\r" +" "*500, end="", flush=True)
                break
        msg = f"\rEpisode {episode+1}/{episodes}, Step {step+1}, Total Reward: {total_reward} solved {solved_tims} times"
        print("\r" +" "*len(msg), end="", flush=True)
        print(msg, end="", flush=True)