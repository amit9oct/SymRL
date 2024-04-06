from sympy import Eq, sympify, simplify, Add
from sympy_addons import customize_rewrite

class CustomEq(Eq):
    C = sympify('C')
    def __new__(cls, lhs, rhs, simplify_identity = False, **options):
        lhs = lhs
        rhs = rhs
        if simplify_identity:
            lhs = CustomEq.simplify_identity(lhs)
            rhs = CustomEq.simplify_identity(rhs)
        return Eq.__new__(cls, lhs, rhs, **options)

    def postorder_traversal(expr, expr_op, parent=None):
        new_args = []
        if not hasattr(expr, 'args'):
            return expr
        orig_args_count = len(expr.args)
        for arg in expr.args:
            new_arg = CustomEq.postorder_traversal(arg, expr_op, parent=expr)
            if new_arg is not None:
                new_args.append(new_arg)
        if len(new_args) > 0:
            new_expr = expr.func(*new_args, evaluate=False)
        elif orig_args_count == 0:
            new_expr = expr
        else:
            new_expr = None
        if new_expr is not None:
            expr = expr_op(new_expr, parent)
        else:
            expr = None
        return expr
    
    def drop_coeff_with_var(expr, var):
        drop_counts = 0
        dropped_coeff = None
        def _match(term, parent):
            nonlocal drop_counts, dropped_coeff
            simplified_term = sympify(f"{term}", evaluate=True)
            if simplified_term.has(var) and ((expr == parent) or (parent is None and term == expr)) and drop_counts == 0:
                args = list(simplified_term.args)
                if var in args or var == simplified_term:
                    if simplified_term.is_Mul or (var == simplified_term and (parent is None or parent.is_Add)):
                        drop_counts += 1
                        dropped_coeff = term
                        return None
            return term
        rewrite_res = CustomEq.postorder_traversal(expr, _match)
        if rewrite_res is None:
            return 0, dropped_coeff
        else:
            return rewrite_res, dropped_coeff
        
    def collect_coeff_with_var(expr, var):
        collected_coeff = 0
        collected_something = False
        def _match(term, parent):
            nonlocal collected_coeff, collected_something
            simplified_term = sympify(f"{term}", evaluate=True)
            if simplified_term.has(var) and ((expr == parent) or (parent is None and term == expr)):
                args = list(simplified_term.args)
                if var in args or var == simplified_term:
                    if simplified_term.is_Mul or (var == simplified_term and (parent is None or parent.is_Add)):
                        collected_something = True
                        collected_coeff += simplified_term.coeff(var)
                        return None
            return term
        rewrite_res = CustomEq.postorder_traversal(expr, _match)
        if collected_something:
            collected_coeff = sympify(f"{collected_coeff}*{var}")
            # collected_coeff = collect(collected_coeff, var, evaluate=False)[var]*var
        if rewrite_res is None and collected_something:
            return collected_coeff
        elif not collected_something:
            return expr
        else:
            return sympify(f"{collected_coeff} + {rewrite_res}", evaluate=False)
    
    def collect_constants(expr):
        collected_constants = 0
        collected_some_constants = False
        def _match(term, parent):
            nonlocal collected_constants, collected_some_constants
            simplified_term = sympify(f"{term}", evaluate=True)
            if simplified_term.is_Number and ((expr == parent) or (parent is None and term == expr)):
                if parent is None or parent.is_Add:
                    collected_constants += term
                    collected_some_constants = True
                    return None
            return term
        rewrite_res = CustomEq.postorder_traversal(expr, _match)
        if collected_some_constants:
            collected_constants = simplify(collected_constants)
        if rewrite_res is None and collected_some_constants:
            return collected_constants
        elif not collected_some_constants:
            return expr
        else:
            return sympify(f"{rewrite_res} + {collected_constants}", evaluate=False)

    def add_term(expr, term):
        new_expr_args = Add.make_args(expr) + Add.make_args(term)
        return Add._from_args(new_expr_args)
    
    def divide_by_coeff(expr, var, coeff=None):
        coeff_assigned = coeff is not None
        coeff = 1 if coeff is None else coeff
        def _match(term, parent):
            nonlocal coeff, coeff_assigned
            simlified_term = simplify(term)
            if simlified_term.is_Mul and ((expr == parent) or (parent is None and term == expr)) and not coeff_assigned:
                args = list(simlified_term.args)
                if len(args) == 2 and var in args:
                    coeff = term.coeff(var)
                    coeff_assigned = True
            return term
        if not coeff_assigned:
            CustomEq.postorder_traversal(expr, _match)
        def _transform(term, parent):
            if ((expr == parent) or (parent is None and term == expr)):
                if parent is None or parent.is_Add:
                    return simplify(term/coeff)
            return term
        if coeff_assigned and coeff != 1 and coeff != 0:
            expr = CustomEq.postorder_traversal(expr, _transform)
        return expr, coeff
    
    def simplify_identity(expr):
        def _match(term, parent):
            if term.is_Mul:
                args = list(term.args)
                if 1 in args:
                    # Remove the 1 from the term
                    new_term = term.func(*[arg for arg in args if arg != 1], evaluate=False)
                    return new_term
                else:
                    return term
            elif term.is_Add:
                args = list(term.args)
                if 0 in args:
                    # Remove the 0 from the term
                    new_term = term.func(*[arg for arg in args if arg != 0], evaluate=False)
                    return new_term
                else:
                    return term
            else:
                return term
        rewrite_res = CustomEq.postorder_traversal(expr, _match)
        if rewrite_res is None:
            return 0
        else:
            return rewrite_res

    def move_constant(expr):
        constant = 0
        constant_found = False
        def _match(term, parent):
            nonlocal constant, constant_found
            simplified_term = simplify(term)
            if simplified_term.is_Number and ((expr == parent) or (parent is None and term == expr)) and not constant_found:
                if parent is None or parent.is_Add:
                    constant = simplified_term
                    constant_found = True
                    return None
            return term
        rewrite_res = CustomEq.postorder_traversal(expr, _match)
        if rewrite_res is None and constant_found:
            return 0, simplify(constant)
        elif not constant_found:
            return expr, 0
        else:
            return rewrite_res, simplify(constant)

    def move_terms_rewriter_rhs(*args, **kwargs):
        var_name = kwargs.get('var', None)
        assert var_name is not None, 'Variable name not provided'
        lhs, rhs = args
        if var_name == 'C':
            return CustomEq.move_constant_rewriter_rhs(*args, **kwargs)
        else:
            # Match the last coefficient of the variable in the lhs
            var = sympify(var_name, evaluate=False)
            new_lhs, dropped_term = CustomEq.drop_coeff_with_var(lhs, var)
            if dropped_term is not None:
                new_rhs = CustomEq.add_term(rhs, -dropped_term)
            else:
                new_lhs = lhs
                new_rhs = rhs
            return CustomEq(new_lhs, new_rhs, simplify_identity=True, evaluate=False)
    
    def move_terms_rewriter_lhs(*args, **kwargs):
        var_name = kwargs.get('var', None)
        assert var_name is not None, 'Variable name not provided'
        lhs, rhs = args
        if var_name == 'C':
            return CustomEq.move_constant_rewriter_lhs(*args, **kwargs)
        else:
            var = sympify(var_name, evaluate=False)
            new_rhs, dropped_term = CustomEq.drop_coeff_with_var(rhs, var)
            if dropped_term is not None:
                new_lhs = CustomEq.add_term(lhs, -dropped_term)
            else:
                new_lhs = lhs
                new_rhs = rhs
            return CustomEq(new_lhs, new_rhs, simplify_identity=True, evaluate=False)
    
    def move_constant_rewriter_rhs(*args, **kwargs):
        lhs, rhs = args
        new_lhs, constant = CustomEq.move_constant(lhs)
        if constant != 0:
            new_rhs = CustomEq.add_term(rhs, -constant)
        else:
            new_rhs = rhs
        return CustomEq(new_lhs, new_rhs, simplify_identity=True, evaluate=False)
    
    def move_constant_rewriter_lhs(*args, **kwargs):
        lhs, rhs = args
        new_rhs, constant = CustomEq.move_constant(rhs)
        if constant != 0:
            new_lhs = CustomEq.add_term(lhs, constant)
        else:
            new_lhs = lhs
        return CustomEq(new_lhs, new_rhs, simplify_identity=True, evaluate=False)

    def collect_rewriter(*args, **kwargs):
        var_name = kwargs.get('var', None)
        assert var_name is not None, 'Variable name not provided'
        if var_name == 'C':
            return CustomEq.collect_constants_rewriter(*args, **kwargs)
        else:
            var = sympify(var_name, evaluate=False)
            lhs, rhs = args
            lhs = CustomEq.collect_coeff_with_var(lhs, var)
            rhs = CustomEq.collect_coeff_with_var(rhs, var)
            return CustomEq(lhs, rhs, simplify_identity=True, evaluate=False)
    
    def divide_by_coeff_rewriter(*args, **kwargs):
        var_name = kwargs.get('var', None)
        assert var_name is not None, 'Variable name not provided'
        var = sympify(var_name, evaluate=False)
        lhs, rhs = args
        # Count the number of operators in the lhs
        lhs, coeff = CustomEq.divide_by_coeff(lhs, var)
        if coeff != 1 and coeff != 0:
            rhs, _ = CustomEq.divide_by_coeff(rhs, var, coeff=coeff)
        return CustomEq(lhs, rhs, simplify_identity=True, evaluate=False)
    
    def simplify_identity_rewriter(*args, **kwargs):
        lhs, rhs = args
        lhs = CustomEq.simplify_identity(lhs)
        rhs = CustomEq.simplify_identity(rhs)
        # Since the constant has been removed, add it back to the equation
        return CustomEq(lhs, rhs, evaluate=False)
        
    def collect_constants_rewriter(*args, **kwargs):
        lhs, rhs = args
        lhs = CustomEq.collect_constants(lhs)
        rhs = CustomEq.collect_constants(rhs)
        return CustomEq(lhs, rhs, evaluate=False)
    
    def __str__(self):
        # Remove the constant from the equation
        lhs = self.lhs
        rhs = self.rhs
        # lhs = CustomEq.constant_remove_rewriter(lhs)
        # rhs = CustomEq.constant_remove_rewriter(rhs)
        return f"{lhs} = {rhs}"
    
    def set_rewrite_rules():
        customize_rewrite(CustomEq)
        CustomEq.rewrite_manager.add_rule('collect', CustomEq.collect_rewriter)
        CustomEq.rewrite_manager.add_rule('move_terms_rhs', CustomEq.move_terms_rewriter_rhs)
        CustomEq.rewrite_manager.add_rule('move_terms_lhs', CustomEq.move_terms_rewriter_lhs)
        CustomEq.rewrite_manager.add_rule('divide_by_coeff', CustomEq.divide_by_coeff_rewriter)
        CustomEq.rewrite_manager.add_rule('simplify_identity', CustomEq.simplify_identity_rewriter)
        CustomEq.rewrite_manager.add_rule('collect_constants', CustomEq.collect_constants_rewriter)
        CustomEq.rewrite_manager.add_rule('move_constant_rhs', CustomEq.move_constant_rewriter_rhs)
        CustomEq.rewrite_manager.add_rule('move_constant_lhs', CustomEq.move_constant_rewriter_lhs)

CustomEq.set_rewrite_rules()


def create_eqn(eqn: str):
    lhs, rhs = eqn.split('=')
    # Replace all constants with
    lhs = sympify(lhs, evaluate=False)
    rhs = sympify(rhs, evaluate=False)
    return CustomEq(lhs, rhs, evaluate=False)

def get_op_count(eqn: CustomEq):
    lhs, rhs = eqn.args
    return lhs.count_ops(), rhs.count_ops()

def get_var_count(eqn: CustomEq, var: str):
    lhs, rhs = eqn.args
    var = sympify(var, evaluate=False)
    return lhs.count(var), rhs.count(var)

def get_term_count(eqn: CustomEq):
    lhs, rhs = eqn.args
    # recursively count the number of terms in the equation
    def _count_terms(expr):
        if expr.is_Atom:
            return 1
        else:
            return sum([_count_terms(arg) for arg in expr.args])
    return _count_terms(lhs), _count_terms(rhs)


if __name__ == "__main__":
    eqn = create_eqn('3*y + 2*x + 1 + 5 = 4*y + 0 + 2*(y - 4)')
    print("Created equation:", eqn)
    eqn = eqn.rewrite('move_terms_rhs', var='y')
    print("After move_terms(y):", eqn)
    eqn = eqn.rewrite('collect', var='y')
    print("After collect(y):", eqn)
    eqn = eqn.rewrite('simplify_identity')
    print("After simplify_identity:", eqn)
    eqn = eqn.rewrite('divide_by_coeff', var='x')
    print("After divide_by_coeff(x):", eqn)
    eqn = eqn.rewrite('collect_constants')
    print("After collect_constants:", eqn)
    eqn = eqn.rewrite('collect', var='y')
    print("After collect(y):", eqn)
    eqn = eqn.rewrite('collect_constants')
    print("After collect_constants:", eqn)
    eqn = eqn.rewrite('move_constant_rhs')
    print("After move_constant:", eqn)
    eqn = eqn.rewrite('collect_constants')
    print("After collect_constants:", eqn)
    eqn = eqn.rewrite('collect_constants')
    print("After collect_constants:", eqn)
    eqn = eqn.rewrite('collect', var='y')
    print("After collect(y):", eqn)
    print("Op count:", get_op_count(eqn))
    eqn = create_eqn('15*x = 12')
    print("Created equation:", eqn)
    eqn = eqn.rewrite('collect', var='x')
    print("After collect(x):", eqn)
    eqn = create_eqn('6*x + 1 = -10 - 7')
    print("Created equation:", eqn)
    eqn = eqn.rewrite('collect', var='x')
    print("After collect(x):", eqn)
    eqn = create_eqn('5 - 4*x = -6*x - 6 - 4')
    print("Var count:", get_var_count(eqn, 'x'))
    print("Created equation:", eqn)
    eqn = eqn.rewrite('move_terms_rhs', var='x')
    print("After move_terms(x):", eqn)
    eqn = create_eqn('5 - 4*x + 3*x = -6*x - 6 - 4')
    print("Var count:", get_var_count(eqn, 'x'))
    print("Term count:", get_term_count(create_eqn('5 - 4*x + 3*x = -6*x - 6 - 4')))
    print("Created equation:", eqn)
    eqn = eqn.rewrite('move_terms_rhs', var='x')
    print("After move_terms(x):", eqn)
    print("Divide by coeff test")
    eqn = create_eqn('x = 3 - 2')
    print("Created equation:", eqn)
    eqn = eqn.rewrite('divide_by_coeff', var='x')
    print("After divide_by_coeff(x):", eqn)
    eqn = create_eqn('3*x = 3')
    print("Created equation:", eqn)
    eqn = eqn.rewrite('move_terms_rhs', var='x')
    print("After move_terms(x):", eqn)
    eqn = eqn.rewrite('move_terms_lhs', var='x')
    print("After move_terms(x):", eqn)
    eqn = eqn.rewrite('simplify_identity')
    print("After simplify_identity:", eqn)