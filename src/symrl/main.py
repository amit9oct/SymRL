from environment.sympy_env import SympyEnv, EqRewriteActionSpace
from environment.sympy_custom_eq import create_eqn
from policy.eps_greedy_policy import EpsilonGreedyPolicy
from policy.greedy_policy import GreedyPolicy
from policy.random_policy import RandomPolicy
from algorithms.td_zero import TDZero
from algorithms.mc import MonteCarlo
from func_approximator.linear_fun_approx import LinearFuncApproximator
from func_approximator.nn_fun_approx import NeuralFuncApproximator
from func_approximator.base_approx import BaseFuncApproximator
from func_approximator.op_count import OpCountFeatureExtractor
from func_approximator.op_var_count import OpVarCountFeatureExtractor
from func_approximator.op_var_rel_count import OpVarRelCountFeatureExtractor
from func_approximator.var_const_count import VarConstCountFeatureExtractor
from func_approximator.term_var_const_count import TermVarConstCountFeatureExtractor
from func_approximator.rel_term_var_const_count import RelTermVarConstCountFeatureExtractor
from func_approximator.simplified_term_var_const_count import SimpleTermVarConstCountFeatureExtractor
from func_approximator.fourier_features import FourierFeatureExtractor
from tools.eqn_generator import generate_valid_linear_equations
import threading
import os
import time
import numpy as np
from argparse import ArgumentParser
try:
    from .human_policy import HumanPolicy
except ImportError:
    from human_policy import HumanPolicy

try:
    from .run_learning import run_policy
except ImportError:
    from run_learning import run_policy

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(42)

action_space = EqRewriteActionSpace(supported_vars="x")
num_actions = len(action_space.actions)
# alpha = 3e-4
alpha = 7e-5
maximum_step_limit = 10
num_episodes = 800000
log = True
gamma = 1 #0.9
eps = 0.2
args = ArgumentParser()
args.add_argument("--num_episodes", type=int, default=num_episodes)
args.add_argument("--alpha", type=float, default=alpha)
args.add_argument("--maximum_step_limit", type=int, default=maximum_step_limit)
args.add_argument("--gamma", type=float, default=gamma)
args.add_argument("--eps", type=float, default=eps)
args.add_argument("--no_log", action="store_true")
args.add_argument("--no_train", action="store_true")
args.add_argument("--no_test", action="store_true")
args.add_argument("--do_random_test", action="store_true")
args.add_argument("--do_random_train", action="store_true")
args.add_argument("--do_human_test", action="store_true")
args.add_argument("--do_human_train", action="store_true")
args.add_argument("--sort_by_term_count", action="store_true")
args.add_argument("--load", action="store_true")
args.add_argument("--func_approx", type=str, default="lin")
args.add_argument("--algo", type=str, default="td")
args.add_argument("--gui", action="store_true")
args.add_argument("--feat_ex", type=str, default="term_var_const_gtr_cnt")
args.add_argument("--num_features", type=int, default=4)
args.add_argument("--verbose", action="store_true")
args.add_argument("--exp_prefix", type=str, default="")
args.add_argument("--train_cnt", type=int, default=100)
args.add_argument("--test_cnt", type=int, default=25)
args.add_argument("--train_max_terms", type=int, default=10)
args.add_argument("--test_max_terms", type=int, default=12)
args.add_argument("--float_prob", type=float, default=0.12)
args.add_argument("--frac_prob", type=float, default=0.2)
args.add_argument("--folder", type=str, default=None)
args = args.parse_args()
num_episodes = args.num_episodes
alpha = args.alpha
maximum_step_limit = args.maximum_step_limit
gamma = args.gamma
eps = args.eps
log = not args.no_log
load_from_file = args.load
do_train = not args.no_train
do_test = not args.no_test
do_human_test = args.do_human_test
do_human_train = args.do_human_train
func_approx_type = args.func_approx
algo_type = args.algo
launch_gui = args.gui
model_folder = args.folder
feat_ex_type = args.feat_ex
num_features = args.num_features
verbose = args.verbose
random_test = args.do_random_test
random_train = args.do_random_train
sort_by_term_count = args.sort_by_term_count
exp_prefix = args.exp_prefix
train_cnt = args.train_cnt
test_cnt = args.test_cnt
train_max_terms = args.train_max_terms
test_max_terms = args.test_max_terms
float_prob = args.float_prob
frac_prob = args.frac_prob
assert not load_from_file or model_folder is not None, "Model folder must be provided if loading from file"
train_eqns = generate_valid_linear_equations(train_cnt, train_max_terms, 0xf00d, float_prob, frac_prob)
test_equations = generate_valid_linear_equations(test_cnt, test_max_terms, 0xfead, float_prob, frac_prob)
print("Arguments:")
print(args)
for eqn in train_eqns:
    eqn = create_eqn(eqn) # Assert that the equation is valid

for eqn in test_equations:
    eqn = create_eqn(eqn) # Assert that the equation is valid
if sort_by_term_count:
    train_eqns = sorted(train_eqns, key=lambda x: len(x.split()))
    test_equations = sorted(test_equations, key=lambda x: len(x.split()))
if launch_gui:
    os.environ["KIVY_NO_ARGS"] = "1"
    os.environ["KIVY_NO_CONSOLELOG"] = "1"
    from environment.sympy_env_kivy_gui import EquationApp

train_env = SympyEnv(train_eqns, maximum_step_limit=maximum_step_limit, action_space=action_space, randomize_eqn=not sort_by_term_count)
test_env = SympyEnv(test_equations, maximum_step_limit=maximum_step_limit, action_space=action_space, randomize_eqn=False)

if load_from_file:
    func_approx = BaseFuncApproximator.load(model_folder)
else:
    if feat_ex_type == "fourier":
        feat_ex = FourierFeatureExtractor(train_env, n_features=num_features)
    elif feat_ex_type == "op_count":
        feat_ex = OpCountFeatureExtractor(train_env)
        num_features = feat_ex.num_features
    elif feat_ex_type == "op_var_count":
        feat_ex = OpVarCountFeatureExtractor(train_env)
        num_features = feat_ex.num_features
    elif feat_ex_type == "op_var_rel_count":
        feat_ex = OpVarRelCountFeatureExtractor(train_env)
        num_features = feat_ex.num_features
    elif feat_ex_type == "var_const_count":
        feat_ex = VarConstCountFeatureExtractor(train_env)
        num_features = feat_ex.num_features
    elif feat_ex_type == "term_var_const_count":
        feat_ex = TermVarConstCountFeatureExtractor(train_env)
        num_features = feat_ex.num_features
    elif feat_ex_type == "rel_term_var_const_count":
        feat_ex = RelTermVarConstCountFeatureExtractor(train_env)
        num_features = feat_ex.num_features
    elif feat_ex_type == "simpl_term_var_const_count":
        feat_ex = SimpleTermVarConstCountFeatureExtractor(train_env)
        num_features = feat_ex.num_features
    else:
        raise ValueError(f"Invalid feature extractor type: {feat_ex_type}")
    if func_approx_type == "nn":
        func_approx = NeuralFuncApproximator(
            feature_extractor=feat_ex,
            num_features=num_features,
            num_actions=num_actions, 
            learning_rate=alpha)
    elif func_approx_type == "lin":
        func_approx = LinearFuncApproximator(
            feature_extractor=feat_ex, 
            num_features=num_features,
            num_actions=num_actions, 
            learning_rate=alpha, 
            random_init=False)
    else:
        raise ValueError(f"Invalid function approximator type: {func_approx_type}")
if algo_type == "td":
    algo = TDZero(num_actions, func_approx, gamma=gamma)
elif algo_type == "mc":
    algo = MonteCarlo(num_actions, func_approx, gamma=gamma)
else:
    raise ValueError(f"Invalid algorithm type: {algo_type}")
policy = EpsilonGreedyPolicy(epsilon=eps, num_actions=num_actions, func_approximator=func_approx)
train_prefix = f"{exp_prefix}_train__{feat_ex_type}_{func_approx_type}__{algo_type}__eps"
test_prefix = f"{exp_prefix}_test__{feat_ex_type}_{func_approx_type}__{algo_type}__gr"
gui_callback = None
app = None

def test(prefix=None, time_str=None, policy_type="greedy", env=None):
    prefix = prefix if prefix is not None else test_prefix
    env = env if env is not None else test_env
    if do_test:
        # now test the policy
        try:
            if policy_type == "greedy":
                greedy_policy = GreedyPolicy(num_actions=num_actions, func_approximator=func_approx)
            elif policy_type == "eps":
                greedy_policy = EpsilonGreedyPolicy(epsilon=eps, num_actions=num_actions, func_approximator=func_approx)
            elif policy_type == "human":
                greedy_policy = HumanPolicy(num_actions=num_actions, func_approximator=func_approx)
            elif policy_type == "random":
                greedy_policy = RandomPolicy(action_space=action_space)
            else:
                raise ValueError(f"Invalid policy type: {policy_type}")
            with env:
                run_policy(
                    env, 
                    greedy_policy, 
                    None, 
                    episodes=len(env.equation_strs), 
                    learn=False, 
                    log=log, 
                    log_file_prefix=prefix, 
                    render_func_action_callback=gui_callback,
                    verbose=verbose,
                    time_str=time_str)
        except Exception as e:
            print(e)
            pass
    # if app is not None:
    #     app.gui.set_stop()

def train():
    time_str = time.strftime('%Y%m%d_%H%M%S')
    def _eval(policy):
        test(prefix=train_prefix + "_eval", time_str=time_str, policy_type="greedy", 
            env=SympyEnv(train_eqns, maximum_step_limit=maximum_step_limit, action_space=action_space, randomize_eqn=False))
        test(prefix=test_prefix + "_eval", time_str=time_str, policy_type="greedy", 
            env=SympyEnv(test_equations, maximum_step_limit=maximum_step_limit, action_space=action_space, randomize_eqn=False))
    if do_train:
        try:
            with train_env:
                run_policy(
                    train_env, 
                    policy, 
                    algo, 
                    episodes=num_episodes, 
                    learn=True, 
                    log=log, 
                    log_file_prefix=train_prefix, 
                    render_func_action_callback=gui_callback,
                    eval_func_action_callback=_eval,
                    verbose=verbose)
        except Exception as e:
            print(e)
            pass
    if app is not None:
        app.gui.set_stop()


def _gui_callback(env: SympyEnv, state, action, next_state, reward, done, truncated, info):
    if app is not None:
        if action is not None and next_state is not None and done is not None:
            app.gui.update_solution(f"{next_state} [{action}] [done={done}]")
        else:
            app.gui.update_equation(str(env.equation))
            app.gui.update_solution(None, reset=True)
        time.sleep(0.1)

if launch_gui:
    assert load_from_file, "Model must be loaded when launching GUI"
    gui_test_equations = [
        "-2*x - x -9*x/2 =  6 - 4 + 8.4*x"
    ]
    gui_callback = _gui_callback
    greedy_policy = GreedyPolicy(num_actions=num_actions, func_approximator=func_approx)
    gui_env = SympyEnv(gui_test_equations, maximum_step_limit=maximum_step_limit, action_space=action_space, randomize_eqn=False)
    app = EquationApp(gui_env, policy=greedy_policy)
    threading.Thread(target=test).start()
    app.run()
else:
    if do_human_train:
        do_test_orig = do_test
        do_test = True
        test(prefix="human_train", policy_type="human", 
            env=SympyEnv(train_eqns, maximum_step_limit=maximum_step_limit, action_space=action_space, randomize_eqn=False))
        do_test = do_test_orig
    if do_human_test:
        do_test_orig = do_test
        do_test = True
        test(prefix="human_test", policy_type="human",
            env=SympyEnv(test_equations, maximum_step_limit=maximum_step_limit, action_space=action_space, randomize_eqn=False))
        do_test = do_test_orig
    if random_train:
        do_test_orig = do_test
        do_test = True
        time_str = time.strftime('%Y%m%d_%H%M%S')
        random_runs = num_episodes // train_cnt
        for i in range(random_runs):
            test(prefix="random_train", time_str=time_str, policy_type="random",
                env=SympyEnv(train_eqns, maximum_step_limit=maximum_step_limit, action_space=action_space, randomize_eqn=False))
        do_test = do_test_orig
    if random_test:
        do_test_orig = do_test
        do_test = True
        time_str = time.strftime('%Y%m%d_%H%M%S')
        eval_times = num_episodes // 1000
        for i in range(eval_times):
            test(prefix="random_test", time_str=time_str, policy_type="random",
                env=SympyEnv(test_equations, maximum_step_limit=maximum_step_limit, action_space=action_space, randomize_eqn=False))
        do_test = do_test_orig
    train()
    test()
