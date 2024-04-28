import os
import json
import typing
import numpy as np
import argparse
from dataclasses import dataclass
from matplotlib import pyplot as plt

# replay log looks something like
# {
#   "time": "2024-04-21 01:33:44,048", 
#   "details": {
#       "episode": 1, 
#       "start_state": "5*x = -5*x/3 - 2", 
#       "steps": 11, 
#       "action_sequence": ["CLV(x)", "CLV(x)", "CLV(x)", "CLV(x)", "CLV(x)", "CLV(x)", "CLV(x)", "CLV(x)", "CLV(x)", "CLV(x)", "CLV(x)"], 
#   "rewards": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
#   "solved": false, "truncated": true
#  }
# }

@dataclass
class ReplaySummary:
    solved_percentatage: int
    avg_steps: float
    avg_reward: float

def summarize_replay(replay_list):
    """
    Summarize the replay list
    """
    num_solved = 0
    total_steps = 0
    total_reward = 0
    solved_percentage = 0
    for episode in replay_list:
        num_solved += episode["details"]["solved"]
        total_steps += episode["details"]["steps"]
        total_reward += sum(episode["details"]["rewards"])
    avg_steps = total_steps / len(replay_list)
    avg_reward = total_reward / len(replay_list)
    solved_percentage = num_solved / len(replay_list)
    return ReplaySummary(solved_percentage, avg_steps, avg_reward)

def replay_analyse(folder) -> typing.List[ReplaySummary]:
    """
    load the replay files from the folder and analyse them
    """
    # check if the folder exists
    assert os.path.exists(folder), f"Folder {folder} does not exist"
    replay_path = os.path.join(folder, "replay.log")
    assert os.path.exists(replay_path), f"Replay file {replay_path} does not exist"
    # load the replay file
    with open(replay_path, "r") as f:
        replay = [json.loads(line) for line in f.readlines()]
    # analyse the replay
    total_episodes = len(replay)
    episode_numbers = [episode["details"]["episode"] for episode in replay]
    unique_episodes = set(episode_numbers)
    replay_summaries = []
    # This means the replay file has duplicate episodes and we need to look at
    # various subsets of the unique episodes
    assert total_episodes % len(unique_episodes) == 0, "Replay file has duplicate episodes but the number of episodes is not a multiple of the unique episodes"
    # get the number of times each unique episode is repeated
    episode_idx = 0
    while episode_idx < total_episodes:
        subset_replay = replay[episode_idx:episode_idx+len(unique_episodes)]
        # analyse the subset replay
        replay_summaries.append(summarize_replay(subset_replay))
        episode_idx += len(unique_episodes)
    return replay_summaries

def plot_replay_summary(
        replay_summaries: typing.List[ReplaySummary], 
        experiment_name, 
        plot_folder: str,
        combined_ax_list: typing.List[plt.Axes],
        max_idx: int = -1):
    """
    Plot the replay summaries
    """
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    solved_percentages = [summary.solved_percentatage for summary in replay_summaries]
    avg_steps = [summary.avg_steps for summary in replay_summaries]
    avg_rewards = [summary.avg_reward for summary in replay_summaries]
    # Plot the running average
    running_avg = 0
    _idx = 0
    running_avg_solved = []
    while _idx < len(solved_percentages):
        running_avg = (running_avg * _idx + solved_percentages[_idx]) / (_idx + 1)
        running_avg = round(running_avg, 4)
        running_avg_solved.append(running_avg)
        _idx += 1
    _idx = 0
    running_avg = 0
    running_avg_steps = []
    while _idx < len(avg_steps):
        running_avg = (running_avg * _idx + avg_steps[_idx]) / (_idx + 1)
        running_avg = round(running_avg, 4)
        running_avg_steps.append(running_avg)
        _idx += 1
    _idx = 0
    running_avg = 0
    running_avg_rewards = []
    while _idx < len(avg_rewards):
        running_avg = (running_avg * _idx + avg_rewards[_idx]) / (_idx + 1)
        running_avg = round(running_avg, 4)
        running_avg_rewards.append(running_avg)
        _idx += 1

    std_solved_percentages = np.std(solved_percentages)
    std_avg_steps = np.std(avg_steps)
    std_avg_rewards = np.std(avg_rewards)

    # Plot the moving average and +/- 1 std deviation
    # Color the area between the bounds
    # Do fill plots only when std is not zero
    if not np.isclose(std_solved_percentages, 0, atol=1e-5):
        ax[0].fill_between(range(len(solved_percentages)), solved_percentages - std_solved_percentages, solved_percentages + std_solved_percentages, color="skyblue")
    if not np.isclose(std_avg_steps, 0, atol=1e-5):
        ax[1].fill_between(range(len(avg_steps)), avg_steps - std_avg_steps, avg_steps + std_avg_steps, color="skyblue")
    if not np.isclose(std_avg_rewards, 0, atol=1e-5):
        ax[2].fill_between(range(len(avg_rewards)), avg_rewards - std_avg_rewards, avg_rewards + std_avg_rewards, color="skyblue")
    
    # Plot the moving average
    ax[0].plot(running_avg_solved, color="red")
    ax[1].plot(running_avg_steps, color="red")
    ax[2].plot(running_avg_rewards, color="red")

    # Add the labels
    ax[0].set_title("Solved Percentage")
    ax[0].set_xlabel("Episode bucket")
    ax[0].set_ylabel("Percentage")
    ax[1].set_title("Average Steps")
    ax[1].set_xlabel("Episode bucket")
    ax[1].set_ylabel("Steps")
    ax[2].set_title("Average Reward")
    ax[2].set_xlabel("Episode bucket")
    ax[2].set_ylabel("Reward")
    fig.suptitle(experiment_name)
    plot_path = os.path.join(plot_folder, f"{experiment_name}.png")
    plt.savefig(plot_path)
    print(f"[{experiment_name}] Solved Percentage: {running_avg_solved[-1]}, Average Steps: {running_avg_steps[-1]}, Average Reward: {running_avg_rewards[-1]}")
    if max_idx == -1:
        max_solved_episode = np.argmax(solved_percentages)
    else:
        max_solved_episode = max_idx
    print(f"[{experiment_name}] Max Solved Episode: {max_solved_episode}")
    print(f"[{experiment_name}] Max Solved Percentage: {solved_percentages[max_solved_episode]}, Average Steps: {avg_steps[max_solved_episode]}, Average Reward: {avg_rewards[max_solved_episode]}")
    min_avg_steps_episode = np.argmin(avg_steps)
    print(f"[{experiment_name}] Min Average Steps Episode: {min_avg_steps_episode}")
    print(f"[{experiment_name}] Min Average Steps: {avg_steps[min_avg_steps_episode]}, Solved Percentage: {solved_percentages[min_avg_steps_episode]}, Average Reward: {avg_rewards[min_avg_steps_episode]}")

    # add the plot to the combined plot
    combined_ax_list[0].plot(running_avg_solved, label=experiment_name)
    combined_ax_list[1].plot(running_avg_steps, label=experiment_name)
    combined_ax_list[2].plot(running_avg_rewards, label=experiment_name)
    pass

def plot_experiments(experiment_details: list, plot_folder: str, plot_name: str = "combined", idx: int = -1):
    """
    Plot the experiments
    """
    os.makedirs(plot_folder, exist_ok=True)
    combined_fig, combined_ax = plt.subplots(1, 3, figsize=(20, 5))
    all_replay_summaries : typing.List[typing.List[ReplaySummary]] = []
    for experiment in experiment_details:
        experiment_name = experiment["name"]
        folder = experiment["folder"]
        replay_summaries = replay_analyse(folder)
        all_replay_summaries.append(replay_summaries)
    # Make sure that the number of episodes in each replay summary is the same
    # extrapolate the replay summaries to have the same number of episodes
    max_episodes = max([len(replay_summary) for replay_summary in all_replay_summaries])
    for replay_summary in all_replay_summaries:
        while len(replay_summary) < max_episodes:
            replay_summary.append(replay_summary[-1])
    for replay_summaries, experiment in zip(all_replay_summaries, experiment_details):
        experiment_name = experiment["name"]
        plot_replay_summary(replay_summaries, experiment_name, plot_folder, combined_ax, idx)
    combined_plot_path = os.path.join(plot_folder, "combined.png")
    combined_ax[0].set_title("Solved Percentage")
    combined_ax[0].set_xlabel("Episode bucket")
    combined_ax[0].set_ylabel("Percentage")
    combined_ax[1].set_title("Average Steps")
    combined_ax[1].set_xlabel("Episode bucket")
    combined_ax[1].set_ylabel("Steps")
    combined_ax[2].set_title("Average Reward")
    combined_ax[2].set_xlabel("Episode bucket")
    combined_ax[2].set_ylabel("Reward")

    # Add only one legend to the combined plot to avoid clutter on bottom right corner
    combined_ax[2].legend(loc="lower right")
    combined_fig.suptitle(plot_name)
    combined_fig.savefig(combined_plot_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--analysis_json", type=str, help="The json file containing the analysis details", default="analysis.json")
    args.add_argument("--plot_folder", type=str, help="The folder to save the plots", default=".logs/plots/")
    args.add_argument("--plot_name", type=str, help="The name of the combined plot", default="combined")
    args.add_argument("--idx", type=int, help="The index of the experiment to plot", default=-1)
    args = args.parse_args()
    # Load the analysis json
    assert os.path.exists(args.analysis_json), f"Analysis json file {args.analysis_json} does not exist"
    with open(args.analysis_json, "r") as f:
        experiment_details = json.load(f)
    # experiment_details = [
    #     {
    #         "name": "test_comp_term_var_const",
    #         "folder": "src/.logs/good_runs/100_25_test__simpl_term_var_const_count_lin__td__gr_eval/20240418_064630"
    #     },
    #     {
    #         "name": "test_term_var_const",
    #         "folder": "src/.logs/good_runs/100_25_test__term_var_const_count_lin__td__gr_eval/20240418_061200"
    #     },
    #     {
    #         "name": "test_comp_op_var",
    #         "folder": "src/.logs/good_runs/100_25_test__op_var_count_lin__td__gr_eval/20240418_061200"
    #     },
    #     {
    #         "name": "test_op",
    #         "folder": "src/.logs/good_runs/100_25_test__op_count_lin__td__gr_eval/20240418_061200"
    #     }
    # ]
    plot_folder = args.plot_folder
    plot_name = args.plot_name
    idx = args.idx
    plot_experiments(experiment_details, plot_folder, plot_name, idx)
    pass