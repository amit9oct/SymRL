from environment.sympy_env import SympyEnv, EqRewriteActionSpace
from policy.eps_greedy_policy import EpsilonGreedyPolicy
from policy.greedy_policy import GreedyPolicy
from algorithms.td_zero import TDZero
from algorithms.mc import MonteCarlo
from func_approximator.linear_fun_approx import LinearFuncApproximator
from func_approximator.nn_fun_approx import NeuralFuncApproximator
from func_approximator.base_approx import FeatureExtractor, BaseFuncApproximator
import threading
import numpy as np
from argparse import ArgumentParser

try:
    from .run_learning import run_policy
except ImportError:
    from run_learning import run_policy

train_eqns = [
    "7*x - 8*x - 3 = -8 - 6",
    "7*x + 3*x - 3 = -7 + 9",
    "-10*x + 6*x + 7 = 4 - 4",
    "-5*x - 9 = 5 + 4",
    "7*x - 1*x - 8 = 8 + 4",
    "-9*x + 6*x + 4 = 1 - 1",
    "6*x + 1 = -10 - 7",
    "7*x + 4*x + 8*x + 6 = -1 - 10",
    "6*x + 6 = 4 - 2",
    "3*x + 9*x - 1*x - 1 = -8 + 8",
    "2*x + 4*x - 2 = 6 - 3",
    "-3*x + 5*x + 10 = -5 + 2",
    "4*x - 7*x + 5 = 3 - 7",
    "-8*x + 2*x - 9 = -6 + 5",
    "9*x - 3*x + 4 = 0 - 2",
    "1*x + 2*x + 3 = -4 + 9",
    "-6*x + 7*x - 4 = 5 + 3",
    "5*x - 2*x + 8 = -7 + 6",
    "8*x - 4*x - 5 = 2 - 8",
    "-2*x + 3*x + 9 = -1 - 10",
    "10*x - 5*x + 7 = 4 + 2",
    "-1*x + 6*x - 8 = -3 + 7",
    "3*x - 9*x + 2 = 8 - 4",
    "4*x + 8*x - 6 = -5 - 6",
    "-7*x + 4*x + 10 = -2 + 8"
]

test_equations = [
    "10*x + 8*x - 5 = 10 - 8",
    "5*x + 10*x - 9 = 6 - 3",
    "-6*x - 9*x + 10 = -7 + 4",
    "-10*x + 5*x - 7 = -2 + 7",
    "5*x + 6 = -3 + 8",
    "3*x - 7*x + 2 = 5 - 9",
    "8*x + 9*x - 4 = 1 + 6",
    "2*x - 3*x + 7 = -4 + 3",
    "9*x + 4*x - 8 = 2 - 7",
    "-4*x + 6*x + 9 = -5 + 10",
    "1*x - 2*x - 3 = 4 - 8",
    "7*x + 5*x + 6 = -7 + 2",
    "-8*x - 1*x - 10 = 3 + 9",
    "6*x - 4*x + 5 = -6 - 4",
    "-7*x + 3*x - 6 = 8 - 3",
    "4*x + 7*x + 8 = -2 - 5",
    "-5*x + 2*x - 9 = 1 + 4",
    "10*x - 8*x + 3 = 7 - 1",
    "9*x - 5*x - 4 = -3 + 6",
    "-1*x + 9*x + 7 = 5 + 2",
    "2*x + 6*x - 5 = -4 + 7",
    "-3*x - 6*x + 1 = 8 - 9",
    "8*x - 2*x - 7 = 3 + 5",
    "-9*x + 1*x + 8 = -7 + 4",
    "7*x - 9*x + 10 = 2 + 6"
]

class OpCountFeatureExtractor(FeatureExtractor):
    def __init__(self, env: SympyEnv):
        self.env = env

    def __call__(self, observation):
        lhs, rhs = SympyEnv.get_lhs_rhs_op_count(observation)
        return np.array([lhs, rhs])
    
    def pretty_print_feature_extractor(self) -> str:
        feature_names = ["op_count_lhs", "op_count_rhs"]
        return str(feature_names)
    
    def pretty_print_state(self, state) -> str:
        state_str = str(self(state))
        return state_str
    
    def pretty_print_action(self, action) -> str:
        action_str = self.env.action_space.actions[action]
        return action_str

class OpVarCountFeatureExtractor(FeatureExtractor):
    def __init__(self, env: SympyEnv):
        self.env = env

    def __call__(self, observation):
        lhs_term_cnt, rhs_term_cnt = SympyEnv.get_lhs_rhs_term_count(observation)
        lhs_op_cnt, rhs_op_cnt = SympyEnv.get_lhs_rhs_op_count(observation)
        lhs, rhs = (lhs_term_cnt + lhs_op_cnt), (rhs_term_cnt + rhs_op_cnt)
        lhs_var_count, rhs_var_count = SympyEnv.get_lhs_rhs_var_count(observation)
        return np.array([lhs, rhs, lhs_var_count, rhs_var_count])

    def pretty_print_feature_extractor(self) -> str:
        feature_names = ["lhs_term_count", "rhs_term_count", "lhs_var_count", "rhs_var_count"]
        return str(feature_names)
    
    def pretty_print_state(self, state) -> str:
        state_str = str(self(state))
        return state_str
    
    def pretty_print_action(self, action) -> str:
        action_str = self.env.action_space.actions[action]
        return action_str

action_space = EqRewriteActionSpace(supported_vars="x")
num_actions = len(action_space.actions)
alpha = 1e-3
maximum_step_limit = 20
num_episodes = 75000
render = True
gamma = 0.9
eps = 0.25
args = ArgumentParser()
args.add_argument("--num_episodes", type=int, default=num_episodes)
args.add_argument("--alpha", type=float, default=alpha)
args.add_argument("--maximum_step_limit", type=int, default=maximum_step_limit)
args.add_argument("--gamma", type=float, default=gamma)
args.add_argument("--eps", type=float, default=eps)
args.add_argument("--render", type=bool, default=render)
args.add_argument("--do_train", type=bool, default=True)
args.add_argument("--do_test", type=bool, default=True)
args.add_argument("--load", type=bool, default=False)
args.add_argument("--func_approx", type=str, default="nn")
args.add_argument("--algo", type=str, default="mc")
args.add_argument("--gui", type=bool, default=False)
args = args.parse_args()
num_episodes = args.num_episodes
alpha = args.alpha
maximum_step_limit = args.maximum_step_limit
gamma = args.gamma
eps = args.eps
render = args.render
load_from_file = args.load
do_train = args.do_train
do_test = args.do_test
func_approx_type = args.func_approx
algo_type = args.algo
launch_gui = args.gui
if launch_gui:
    import os
    import time
    os.environ["KIVY_NO_ARGS"] = "1"
    os.environ["KIVY_NO_CONSOLELOG"] = "1"
    from environment.sympy_env_kivy_gui import EquationApp

train_env = SympyEnv(train_eqns, maximum_step_limit=maximum_step_limit, action_space=action_space, randomize_eqn=True)
test_env = SympyEnv(test_equations, maximum_step_limit=maximum_step_limit, action_space=action_space, randomize_eqn=False)

if load_from_file:
    func_approx = BaseFuncApproximator.load("model")
else:
    if func_approx_type == "nn":
        func_approx = NeuralFuncApproximator(
            feature_extractor=OpVarCountFeatureExtractor(train_env), 
            num_features=4, 
            num_actions=num_actions, 
            learning_rate=alpha)
    else:
        func_approx = LinearFuncApproximator(
            feature_extractor=OpVarCountFeatureExtractor(train_env), 
            num_features=4, 
            num_actions=num_actions, 
            learning_rate=alpha, 
            random_init=False)
if algo_type == "td":
    algo = TDZero(num_actions, func_approx, gamma=gamma)
elif algo_type == "mc":
    algo = MonteCarlo(num_actions, func_approx, gamma=gamma)
else:
    raise ValueError(f"Invalid algorithm type: {algo_type}")
policy = EpsilonGreedyPolicy(epsilon=eps, num_actions=num_actions, func_approximator=func_approx)
train_prefix = f"train__approx_nn__td__eps"
test_prefix = f"test__approx_nn__td__gr"
gui_callback = None
app = None

def train():
    if do_train and not load_from_file:
        try:
            with train_env:
                run_policy(
                    train_env, 
                    policy, 
                    algo, 
                    episodes=num_episodes, 
                    learn=True, 
                    log=render, 
                    log_file_prefix=train_prefix, 
                    render_func_action_callback=gui_callback)
        except Exception as e:
            print(e)
            pass
    if app is not None:
        app.stop()

def test():
    if do_test:
        # now test the policy
        try:
            greedy_policy = GreedyPolicy(num_actions=num_actions, func_approximator=func_approx)
            with test_env:
                run_policy(test_env, greedy_policy, None, episodes=len(test_env.equation_strs), learn=False, log=render, log_file_prefix=test_prefix, render_func_action_callback=gui_callback)
        except Exception as e:
            print(e)
            pass
    if app is not None:
        app.stop()

def _gui_callback(env: SympyEnv, state, action, next_state, reward, done, truncated, info):
    if app is not None:
        if action is not None and next_state is not None and done is not None:
            app.gui.update_solution(f"{next_state} [{action}] [done={done}]")
        else:
            app.gui.update_equation(str(env.equation))
            app.gui.update_solution(None, reset=True)

if launch_gui:
    gui_callback = _gui_callback
    app = EquationApp(train_env)
    threading.Thread(target=train).start()
    app.run()
    app = EquationApp(test_env)
    threading.Thread(target=test).start()
    app.run()
else:
    train()
    test()
