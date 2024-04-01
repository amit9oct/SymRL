import gymnasium as gym
import numpy as np
try:
    from .sympy_custom_eq import CustomEq, create_eqn, get_op_count, get_var_count, get_term_count
    from .sympy_env_gui import SympyEnvGUI
except ImportError:
    from sympy_custom_eq import CustomEq, create_eqn, get_op_count, get_var_count, get_term_count
    from sympy_env_gui import SympyEnvGUI

class EqRewriteActionSpace(gym.Space):
    """
    A custom action space that handles string-encoded actions with parameters.
    """
    def __init__(self, supported_vars="abcdefghijklmnopqrstuvwxyz"):
        super(EqRewriteActionSpace, self).__init__((), np.int64)
        self.action_types = ["collect", "move_terms", "divide_by_coeff", "simplify_identity", "collect_constants", "move_constant"]
        self.action_types_short = ["CL", "MV", "DIV", "SIM", "CLC", "MVC"]
        self.action_types_map = {short: action for short, action in zip(self.action_types_short, self.action_types)}
        self.actions = [f"{action}({var})" for action in self.action_types_short[:-2] for var in supported_vars]
        self.actions += ["CL(C)", "MV(C)"]

    def sample(self):
        # Randomly sample an action
        return np.random.choice(self.actions)

    def contains(self, x):
        # Check if x is a valid action
        return x in self.actions

class SympyEnv(gym.Env):
    def __init__(self, equation_strs: list, maximum_step_limit: int = None, action_space=None, randomize_eqn=True):
        assert isinstance(equation_strs, list), "Equations should be a list of strings"
        assert all(isinstance(eqn, str) for eqn in equation_strs), "Equations should be a list of strings"
        assert action_space is None or isinstance(action_space, EqRewriteActionSpace), "Invalid action space"
        assert len(equation_strs) > 0, "At least one equation should be provided"
        self.equation_strs = equation_strs
        self.equation_str = equation_strs[0]
        self.equation = create_eqn(self.equation_str)
        self.maximum_step_limit = maximum_step_limit
        self.step_count = 0
        SympyEnv._check_eqn(self.equation)
        self.action_space = EqRewriteActionSpace(supported_vars='x') if action_space is None else action_space
        self.observation_space = None
        self.reward_range = (0, 1)
        self.gui : SympyEnvGUI = None
        self.randomize_eqn = randomize_eqn
        self.eqn_idx = -1
        self.eqn_solved_count = [0] * len(equation_strs)
        self.solved_eqns = np.zeros(len(equation_strs), dtype=bool)
        self.reset()
    
    @property
    def done(self):
        return self.is_solved()
    
    def step(self, action):
        trunction_reward = -1
        success_reward = 5
        failure_reward = -1
        if self.maximum_step_limit is not None and self.step_count >= self.maximum_step_limit:
            info = {
                "solved_eqns_count": float(np.sum(self.solved_eqns)),
            }
            return self.equation, trunction_reward, self.done, True, info
        else:
            if isinstance(action, int) or np.issubdtype(action, np.integer):
                action = self.action_space.actions[action]
            if not self.action_space.contains(action):
                raise ValueError(f"Invalid action: {action}")
            action_type, var = action.split('(')
            var = var.rstrip(')')
            # Assuming CustomEq class has these rewriter methods correctly implemented.
            try:
                full_action_type = self.action_space.action_types_map[action_type]
                self.equation = self.equation.rewrite(full_action_type, var=var)
            except AttributeError:
                raise ValueError(f"Invalid action type: {action_type}")
            done = self.is_solved()
            if done:
                self.eqn_solved_count[self.eqn_idx] += 1
                self.solved_eqns[self.eqn_idx] = True
            # Reward incentivizes solving the equation quickly and solving more equations
            reward = (success_reward/self.eqn_solved_count[self.eqn_idx] + success_reward/self.step_count) if done else failure_reward
            self.step_count += 1
            info = {
                "solved_eqns_count": float(np.sum(self.solved_eqns)),
            }
            return self.equation, reward, done, False, info

    def reset(self):
        # Select a new equation randomly
        if self.randomize_eqn:
            self.eqn_idx = np.random.randint(0, len(self.equation_strs))
            self.equation_str = self.equation_strs[self.eqn_idx]
        else:
            self.eqn_idx = (self.eqn_idx + 1) % len(self.equation_strs)
            self.equation_str = self.equation_strs[self.eqn_idx]
        self.equation = create_eqn(self.equation_str)
        self.step_count = 0
        return self.equation
    
    def is_solved(self):
        return SympyEnv._check_solved(self.equation)
    
    def render(self, info=None):
        if self.gui is not None:
            # Update the GUI with the current equation
            self.gui.latest_equation.append(str(self.equation) + str(info))
            self.gui.update_gui_with_latest_equation()
        else:
            # Fallback to console printing if GUI is not used
            print(self.equation)
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @staticmethod
    def _check_solved(eqn):
        lhs, rhs = eqn.args
        if lhs.is_Atom and rhs.is_Atom:
            if lhs.is_Symbol and rhs.is_Number:
                return lhs.name == 'x'
            elif lhs.is_Number and rhs.is_Symbol:
                return rhs.name == 'x'
            elif lhs.is_Number and rhs.is_Number:
                return lhs == rhs
            else:
                return lhs == rhs # Reflexivity
        else:
            return False

    @staticmethod
    def _check_eqn(eqn):
        lhs, rhs = eqn.args
        unknown_supported_symbols = set()
        def _math(expr, parent):
            nonlocal unknown_supported_symbols
            if expr.is_Atom and expr.is_Symbol:
                # symbol should be only be a-z
                if not expr.name.islower():
                    unknown_supported_symbols.add(expr)
            return expr
        CustomEq.postorder_traversal(lhs, _math)
        CustomEq.postorder_traversal(rhs, _math)
        if len(unknown_supported_symbols) >= 1:
            raise ValueError(f"Equation has unknown supported symbols: {unknown_supported_symbols}")
    
    @staticmethod
    def get_lhs_rhs_op_count(observation):
        lhs, rhs = get_op_count(observation)
        return lhs, rhs
    
    @staticmethod
    def get_lhs_rhs_var_count(observation):
        lhs, rhs = get_var_count(observation, var='x')
        return lhs, rhs
    
    @staticmethod
    def get_lhs_rhs_term_count(observation):
        lhs, rhs = get_term_count(observation)
        return lhs, rhs