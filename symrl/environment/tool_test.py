import sympy as sp

# Prime factorization of 2003
factors_2003 = sp.factorint(2003)
print("Prime factors of 2003:", factors_2003)

# Define symbols
x, y = sp.symbols('x y', integer=True)

# Original expression
expr = 4*x**3 - 7*y**3 - 2003

# Let's check if there's any simplification or properties we can observe
simplified_expr = sp.simplify(expr)
print("Simplified expression:", simplified_expr)

# Additionally, let's explore the expression modulo some likely candidates that might reveal structural properties.
# Common choices are moduli that relate to the factors of 2003 or the coefficients in the equation.

# Compute and print the expression modulo various numbers
for mod in [2, 3, 7, 17]:  # Including 17 as it's a factor of 2003
    mod_expr = sp.simplify(expr % mod)
    print(f"Expression modulo {mod}:", mod_expr)
