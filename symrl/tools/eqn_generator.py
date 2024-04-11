import random

def generate_valid_linear_equations(num_equations, max_terms, seed=None, float_probability=0.12, frac_probability=0.20):
    """
    Generate valid linear equations with at least one 'x' on either LHS or RHS.
    The sum of 'x' coefficients will not be zero. Allows floating point numbers as coefficients or constants.
    
    Parameters:
    - num_equations: Number of equations to generate
    - max_terms: Maximum number of terms in an equation
    - seed: Seed for the random number generator (optional)
    - p: Probability of generating a floating point coefficient or constant term
    
    Returns:
    - A list of generated linear equations as strings
    """
    assert float_probability >= 0 and float_probability <= 1, "Probability must be between 0 and 1"
    assert frac_probability >= 0 and frac_probability <= 1, "Probability must be between 0 and 1"
    assert float_probability + frac_probability < 1, "Sum of probabilities must be less than or equal to 1"
    if seed is not None:
        random.seed(seed)
    equations = []

    def generate_coefficient():
        """Generate a coefficient or constant, with a chance 'p' of being a floating point number."""
        if random.random() < float_probability:
            return round(random.uniform(-10, 10), 2)
        elif random.random() < float_probability + frac_probability:
            num = random.randint(-10, 10)
            den = random.randint(1, 10)
            if den == 1 or num == 0:
                return num
            else:
                return f"{num}/{den}"
        else:
            return random.randint(-10, 10)

    def add_term(with_x, side_equation, skip_zero=True):
        """Generate and return a term for the equation, along with its coefficient."""
        coefficient = generate_coefficient()
        coefficient_eval = eval(str(coefficient))
        if coefficient_eval == 0 and skip_zero:
            return "", 0  # Skip zero coefficients
        sign = "+" if coefficient_eval >= 0 and side_equation and not side_equation.endswith("=") else ""
        if coefficient_eval < 0:
            sign = "-"
            coefficient_eval = -coefficient_eval
            if isinstance(coefficient, str):
                coefficient = coefficient.strip("-")
            else:
                coefficient = -coefficient
        term = f"{sign} {coefficient}" if sign else f"{coefficient}"
        if with_x:
            if isinstance(coefficient, str) and (coefficient.startswith("-1/") or coefficient.startswith("1/")):
                den = coefficient.split("/")[1]
                term = f"{sign} x/{den}" if sign else f"x/{den}"
            else:
                if coefficient_eval != 1 and coefficient_eval != -1:
                    term += "*x"
                else:
                    term = term.strip('1')
                    term += "x"
            return term, coefficient_eval if sign != "-" else -coefficient_eval
        return term, 0

    while len(equations) < num_equations:
        num_terms = random.randint(3, max_terms)
        equation = ""
        total_coefficient = 0

        # Generate terms for both sides
        for side in ('LHS', 'RHS'):
            num_terms_side = random.randint(1, num_terms - 1) if side == 'LHS' else num_terms - num_terms_side
            for _idx in range(num_terms_side):
                with_x = random.choice([True, False]) or (total_coefficient == 0 and side == 'RHS')
                term, coeff = add_term(with_x, equation, skip_zero= _idx != num_terms_side - 1 or side != 'RHS')
                equation += " " + term if term else ""
                total_coefficient += coeff if with_x else 0

            if side == 'LHS':
                equation += " ="
                if total_coefficient == 0:  # Ensure at least one 'x' term in the equation
                    equation_rhs, _ = add_term(True, "")
                    equation += " " + equation_rhs

        # Ensure non-zero sum of 'x' coefficients
        if total_coefficient != 0:
            equations.append(equation.strip())

    return equations

if __name__ == "__main__":
    num_equations = 1000
    max_terms = 50
    seed = 0xf00
    equations = generate_valid_linear_equations(num_equations, max_terms, seed)
    for eq in equations:
        print(eq)
