import uuid
from algorithms.base_aglo import BaseAlgo
from policy.base_policy import BasePolicy
from tools.log_utils import setup_logger
import gymnasium as gym
import time
import json
import os
import typing

def run_policy(
        env : gym.Env, 
        policy: BasePolicy, 
        rl_algo: BaseAlgo, 
        episodes=100,
        learn=True,
        log=False,
        log_file_prefix="symrl",
        ckpt_num=0,
        ckpt_period=1000,
        render_func_action_callback: typing.Callable = None,
        eval_func_action_callback: typing.Callable = None,
        verbose=False,
        time_str=None):
    time_str = time.strftime('%Y%m%d_%H%M%S') if time_str is None else time_str
    log_folder = f".logs/{log_file_prefix}/{time_str}"
    os.makedirs(log_folder, exist_ok=True)
    policy_log_file = f"{log_folder}/policy.log"
    random_guid = str(uuid.uuid4())
    policy_logger = setup_logger(f"policy_logger_{log_file_prefix}_{random_guid}", policy_log_file)
    replay_log_file = f"{log_folder}/replay.log"
    replay_format = '{"time": "%(asctime)s", "details": %(message)s}'
    replay_logger = setup_logger(f"stats_logger_{log_file_prefix}_{random_guid}", replay_log_file, level="INFO", format=replay_format)
    summary_stats_file = f"{log_folder}/summary.log"
    summary_stats_format = '{"time": "%(asctime)s", "stats": %(message)s}'
    summary_stats_logger = setup_logger(f"summary_stats_logger_{log_file_prefix}_{random_guid}", summary_stats_file, level="INFO", format=summary_stats_format)
    solved_times = 0
    max_msg_len = 0
    running_avg_reward = 0
    last_reward = 0
    info = {}
    model_folder = f"{log_folder}/model"
    ckpt_folder = f"{model_folder}/ckpt"
    os.makedirs(ckpt_folder, exist_ok=True)
    policy_logger.info(f"[START]:\nStarting policy = {policy}, RL Algorithm = {rl_algo}, Episodes = {episodes}, Learn = {learn}, Log = {log}, Log File Prefix = {log_file_prefix}, Time = {time_str}")
    avg_steps = 0
    for episode in range(episodes):
        state = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        if render_func_action_callback is not None:
            render_func_action_callback(env, state, None, None, None, None, None, None)
        start_state = state
        total_reward = 0
        action_sequence = []
        truncated = False
        done = False
        step = 0
        rewards = []
        while not done and not truncated:
            action = policy.select_action(state)
            try:
                next_state, reward, done, truncated, info = env.step(action)
            except Exception as e:
                policy_logger.error(f"[ERROR]:\nError in episode {episode+1}, step {step+1}: {e}")
                raise
            if log and verbose:
                action_prime = policy.pretty_print_action(action)
                state_prime = policy.pretty_print_state(state)
                policy_logger.info(f"[STEP]:\neps={episode+1} state={state_prime} action={action_prime} reward={reward} done={done}")
            if render_func_action_callback is not None:
                render_func_action_callback(env, state, action, next_state, reward, done, truncated, info)
            if learn:
                rl_algo.update(state, action, reward, next_state, done or truncated)
            action_sequence.append(action)
            rewards.append(reward)
            state = next_state
            total_reward += reward
            step += 1
            if done:
                solved_times += 1
                if log:
                    msg = f"[SOLVED]:\n{start_state} in {step} steps! Total reward: {total_reward}"
                    policy_logger.info(msg)
        last_reward = total_reward
        running_avg_reward = (running_avg_reward * episode + total_reward) / (episode + 1)
        avg_steps = (avg_steps * episode + step) / (episode + 1)
        replay_logger.info(
            json.dumps(
            {
                "episode": episode+1,
                "start_state": str(start_state),
                "steps": step,
                "action_sequence": [policy.pretty_print_action(action) for action in action_sequence],
                "rewards": rewards,
                "solved": done,
                "truncated": truncated
            }))
        summary_stats_logger.info(
            json.dumps(
            {
                "episode": episode+1,
                "last_reward": last_reward,
                "last_steps": step,
                "solved": done,
                "truncated": truncated,
                "avg_steps": avg_steps,
                "running_avg_reward": running_avg_reward,
                "solved_times": solved_times,
                "info": info
            }))
        if episode % 25 == 0 and episode > 0:
            msg = f"[STATS]:\nEpisode {episode+1}/{episodes}, Step {step+1}, Avg Steps: {avg_steps}, Avg Reward: {running_avg_reward}, Last Reward: {last_reward}, Solved: {solved_times} times, Solve Rate: {solved_times/(episode+1)*100:.2f}%"
            max_msg_len = max(max_msg_len, len(msg))
            msg = msg.ljust(max_msg_len)
            policy_logger.info(msg)
            if log and episode % 25 == 0 and episode > 0 and learn:
                msg = policy.pretty_print_policy()
                policy_logger.info(f"[POLICY PRINT]:\n{msg}")
        if episode % ckpt_period == 0 and episode > 0 and learn:
            ckpt_num = episode
            chkpt_num_folder = f"{ckpt_folder}/{ckpt_num}"
            os.makedirs(chkpt_num_folder, exist_ok=True)
            policy.save(chkpt_num_folder)
            if eval_func_action_callback is not None:
                eval_func_action_callback(policy)
            if log:
                policy_logger.info(f"[CHKPT]:\nSaved checkpoint at episode {episode+1}.")
    if log:
        policy_logger.info(f"[FINAL]:\nSolved {solved_times} times in {episodes} episodes.")
    if learn:
        policy.save(model_folder)
        if log:
            policy_logger.info(f"[SAVE]:\nSaving model to {model_folder}")