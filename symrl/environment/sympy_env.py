import gymnasium as gym
import numpy as np
try:
    from .sympy_custom_eq import CustomEq, create_eqn, get_op_count
except ImportError:
    from sympy_custom_eq import CustomEq, create_eqn, get_op_count

class EqRewriteActionSpace(gym.Space):
    """
    A custom action space that handles string-encoded actions with parameters.
    """
    def __init__(self):
        super(EqRewriteActionSpace, self).__init__((), np.int64)
        self.action_types = ["collect", "move_terms", "divide_by_coeff", "simplify_identity", "collect_constants", "move_constant"]
        self.action_types_short = ["CL", "MV", "DIV", "SIM", "CLC", "MVC"]
        self.action_types_map = {short: action for short, action in zip(self.action_types_short, self.action_types)}
        self.actions = [f"{action}({var})" for action in self.action_types_short[:-2] for var in "abcdefghijklmnopqrstuvwxyz"]
        self.actions += ["CL(C)", "MV(C)"]

    def sample(self):
        # Randomly sample an action
        return np.random.choice(self.actions)

    def contains(self, x):
        # Check if x is a valid action
        return x in self.actions

class SympyEnv(gym.Env):
    def __init__(self, equation_str: str):
        self.equation_str = equation_str
        self.equation = create_eqn(equation_str)
        SympyEnv._check_eqn(self.equation)
        self.action_space = EqRewriteActionSpace()
        self.observation_space = None
        self.reward_range = (0, 1)
        self.reset()

    def step(self, action):
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
        reward = 1 if done else 0
        return self.equation, reward, done, {}

    def reset(self):
        self.equation = create_eqn(self.equation_str)
        return self.equation
    
    def is_solved(self):
        return SympyEnv._check_solved(self.equation)
    
    def render(self):
        print(self.equation)


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