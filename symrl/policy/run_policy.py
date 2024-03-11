try:
    from .base_policy import BasePolicy
except ImportError:
    from base_policy import BasePolicy

def run_policy(env, policy, episodes=10, max_steps=100):
    for episode in range(episodes):
        obs = env.reset()
        done = False
        step = 0
        print(f"Episode {episode + 1}:")
        while not done and step < max_steps:
            action = policy.select_action(obs)
            obs, reward, done, _ = env.step(action)
            print(f"Episode {episode + 1}, Step {step + 1}: Action: {action}, Reward: {reward}")
            env.render()
            step += 1
        if done:
            print("Equation solved!")
        else:
            print("Equation not solved within the step limit.")
        print("--------------------------------")
