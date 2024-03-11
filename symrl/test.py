from environment.sympy_env import SympyEnv, EqRewriteActionSpace
from policy.run_policy import run_policy
from policy.eps_greedy_policy import EpsilonGreedyPolicy
from algorithms.td_zero import TDZero
from func_approximator.linear_fun_approx import LinearFuncApproximator
import numpy as np
try:
    from .run_learning import run_learning_loop
except ImportError:
    from run_learning import run_learning_loop

train_eqns = [
    'x + 2 = 3',
    '3*x = 9',
    '2*x + 3 = 7',
    "2*x + 3*x - 5 = - 7",
    "3*x + 5*x - 7 = - 9",
    "4*x + 6*x - 9 = - 11",
    "5*x + 7*x - 11 = - 13",
    "6*x + 8*x - 13 = - 15",
    "7*x + 9*x - 15 = - 17",
    "8*x + 10*x - 17 = - 19",
    "9*x + 11*x - 19 = - 21",
    "10*x + 12*x - 21 = - 23",
    "11*x + 13*x - 23 = - 25"
]

test_eqns = [
    "12*x + 14*x - 25 = - 27",
    "13*x + 15*x - 27 = - 29",
    "14*x + 16*x - 29 = - 31",
    "15*x + 17*x - 31 = - 33",
    "16*x + 18*x - 33 = - 35",
    "17*x + 19*x - 35 = - 37",
    "18*x + 20*x - 37 = - 39",
    "19*x + 21*x - 39 = - 41",
    "20*x + 22*x - 41 = - 43",
    "21*x + 23*x - 43 = - 45"
]

def op_count_feature(observation):
    lhs, rhs = SympyEnv.get_lhs_rhs_op_count(observation)
    return np.array([lhs, rhs])

action_space = EqRewriteActionSpace()
num_actions = len(action_space.actions)
alpha = 0.01
func_approx = LinearFuncApproximator(
    feature_extractor=op_count_feature, 
    num_features=2, 
    num_actions=num_actions, 
    learning_rate=alpha, 
    random_init=True)
td_algo = TDZero(func_approx, gamma=0.9)
policy = EpsilonGreedyPolicy(epsilon=0.3, num_actions=num_actions, func_approximator=func_approx)

for equation in train_eqns:
    env = SympyEnv(equation)
    run_learning_loop(env, policy, td_algo, episodes=10000, max_steps_per_episode=150)


# now test the policy
for equation in test_eqns:
    env = SympyEnv(equation)
    run_policy(env, policy, episodes=10, max_steps=100)