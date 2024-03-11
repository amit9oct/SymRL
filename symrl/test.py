from environment.sympy_env import SympyEnv
from policy.random_policy import RandomPolicy
from policy.human_policy import HumanPolicy
from policy.run_policy import run_policy

# Create a sympy environment
env = SympyEnv("2*x + 3*x - 5 = - 7")

# Create a random policy
policy = RandomPolicy(env.action_space)

# Run the policy
run_policy(env, policy, episodes=50, max_steps=100)