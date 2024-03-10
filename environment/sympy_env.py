from sympy import Eq, sympify
from sympy_addons import customize_rewrite

class CustomEq(Eq):
    C = sympify('C')
    def __new__(cls, lhs, rhs, fresh=True, **options):
        if fresh:
            lhs = CustomEq.constant_rewriter(lhs)
            rhs = CustomEq.constant_rewriter(rhs)
        else:
            lhs = lhs
            rhs = rhs
        return Eq.__new__(cls, lhs, rhs, **options)

    def postorder_traversal(expr, expr_op, parent=None):
        new_args = []
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
    
    def constant_rewriter(expr):
        def _match(term, parent):
            if term.is_Number and ((expr == parent and parent.is_Add) or (parent is None and term == expr)):
                new_term = sympify(f"{term}*{CustomEq.C}", evaluate=False)
                return new_term
            else:
                return term
        rewrite_res = CustomEq.postorder_traversal(expr, _match)
        if rewrite_res is None:
            return 0
        else:
            return rewrite_res
    
    def constant_remove_rewriter(expr):
        def _match(term, parent):
            if term.is_Mul and ((expr == parent and parent.is_Add) or (parent is None and term == expr)):
                args = list(term.args)
                if CustomEq.C in args:
                    return term/CustomEq.C
                else:
                    return term
            else:
                return term
        rewrite_res = CustomEq.postorder_traversal(expr, _match)
        if rewrite_res is None:
            return 0
        else:
            return rewrite_res
    
    def drop_coeff_with_var(expr, var):
        drop_counts = 0
        dropped_coeff = None
        def _match(term, parent):
            nonlocal drop_counts, dropped_coeff
            if term.is_Mul and ((expr == parent and parent.is_Add) or (parent is None and term == expr)) and drop_counts == 0:
                args = list(term.args)
                if var in args:
                    drop_counts += 1
                    dropped_coeff = term
                    return None
                else:
                    return term
            else:
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
            if term.is_Mul and ((expr == parent and parent.is_Add) or (parent is None and term == expr)):
                args = list(term.args)
                if var in args:
                    collected_coeff += term
                    collected_something = True
                    return None
                else:
                    return term
            else:
                return term
        rewrite_res = CustomEq.postorder_traversal(expr, _match)
        if rewrite_res is None and collected_something:
            return collected_coeff
        elif not collected_something:
            return expr
        else:
            return sympify(f"{collected_coeff} + {rewrite_res}", evaluate=False)
    
    def add_term(expr, term):
        expr = sympify(f"{expr} + {term}", evaluate=False)
        return expr
    
    def divide_by_coeff(expr, var, coeff=None):
        coeff_assigned = coeff is not None
        coeff = 1 if coeff is None else coeff
        def _match(term, parent):
            nonlocal coeff, coeff_assigned
            if term.is_Mul and ((expr == parent and parent.is_Add) or (parent is None and term == expr)) and not coeff_assigned:
                args = list(term.args)
                if var in args:
                    coeff = term.coeff(var)
                    coeff_assigned = True
            return term
        if not coeff_assigned:
            CustomEq.postorder_traversal(expr, _match)
        def _transform(term, parent):
            if ((expr == parent and parent.is_Add) or (parent is None and term == expr)):
                return term/coeff
            else:
                return term
        if coeff_assigned and coeff != 1 and coeff != 0:
            expr = CustomEq.postorder_traversal(expr, _transform)
        return expr, coeff
    
    def simplify_identity(expr, var):
        def _match(term, parent):
            if term.is_Mul and ((expr == parent and parent.is_Add) or (parent is None and term == expr)):
                args = list(term.args)
                if var in args and 1 in args:
                    # Remove the 1 from the term
                    args.remove(1)
                    return term.func(*args)
                else:
                    return term
            elif term.is_Add and ((expr == parent and parent.is_Add) or (parent is None and term == expr)):
                args = list(term.args)
                if var in args and 0 in args:
                    # Remove the 0 from the term
                    args.remove(0)
                    return term.func(*args)
                else:
                    return term
            else:
                return term
        rewrite_res = CustomEq.postorder_traversal(expr, _match)
        if rewrite_res is None:
            return 0
        else:
            return rewrite_res

    def move_terms_rewriter(*args, **kwargs):
        var_name = kwargs.get('var', None)
        assert var_name is not None, 'Variable name not provided'
        lhs, rhs = args
        # Match the last coefficient of the variable in the lhs
        var = sympify(var_name, evaluate=False)
        new_lhs, dropped_term = CustomEq.drop_coeff_with_var(lhs, var)
        new_rhs = CustomEq.add_term(rhs, -dropped_term)
        return CustomEq(new_lhs, new_rhs, fresh=False, evaluate=False)

    def collect_rewriter(*args, **kwargs):
        var_name = kwargs.get('var', None)
        assert var_name is not None, 'Variable name not provided'
        var = sympify(var_name, evaluate=False)
        lhs, rhs = args
        lhs = CustomEq.collect_coeff_with_var(lhs, var)
        rhs = CustomEq.collect_coeff_with_var(rhs, var)
        return CustomEq(lhs, rhs, fresh=False, evaluate=False)
    
    def divide_by_coeff_rewriter(*args, **kwargs):
        var_name = kwargs.get('var', None)
        assert var_name is not None, 'Variable name not provided'
        var = sympify(var_name, evaluate=False)
        lhs, rhs = args
        # Count the number of operators in the lhs
        lhs, coeff = CustomEq.divide_by_coeff(lhs, var)
        if coeff != 1 and coeff != 0:
            rhs, _ = CustomEq.divide_by_coeff(rhs, var, coeff=coeff)
        return CustomEq(lhs, rhs, fresh=False, evaluate=False)
    
    def simplify_identity_rewriter(*args, **kwargs):
        var_name = kwargs.get('var', None)
        assert var_name is not None, 'Variable name not provided'
        var = sympify(var_name, evaluate=False)
        lhs, rhs = args
        # Remove the constant from the equation
        lhs = CustomEq.constant_remove_rewriter(lhs)
        rhs = CustomEq.constant_remove_rewriter(rhs)
        lhs = CustomEq.simplify_identity(lhs, var)
        rhs = CustomEq.simplify_identity(rhs, var)
        # Since the constant has been removed, add it back to the equation
        return CustomEq(lhs, rhs, fresh=True, evaluate=False)
        
    def __str__(self):
        # Remove the constant from the equation
        lhs = self.lhs
        rhs = self.rhs
        lhs = CustomEq.constant_remove_rewriter(lhs)
        rhs = CustomEq.constant_remove_rewriter(rhs)
        return f"{lhs} = {rhs}"
    
    def set_rewrite_rules():
        customize_rewrite(CustomEq)
        CustomEq.rewrite_manager.add_rule('collect', CustomEq.collect_rewriter)
        CustomEq.rewrite_manager.add_rule('move_terms', CustomEq.move_terms_rewriter)
        CustomEq.rewrite_manager.add_rule('divide_by_coeff', CustomEq.divide_by_coeff_rewriter)
        CustomEq.rewrite_manager.add_rule('simplify_identity', CustomEq.simplify_identity_rewriter)

CustomEq.set_rewrite_rules()


def create_eqn(eqn: str):
    lhs, rhs = eqn.split('=')
    # Replace all constants with
    lhs = sympify(lhs, evaluate=False)
    rhs = sympify(rhs, evaluate=False)
    return CustomEq(lhs, rhs, evaluate=False)

eqn = create_eqn('3*y + 2*x + 1 + 5 = 4*y + 0')
print(eqn)
eqn = eqn.rewrite('move_terms', var='y')
print(eqn)
eqn = eqn.rewrite('collect', var='y')
print(eqn)
eqn = eqn.rewrite('simplify_identity', var='y')
print(eqn)
eqn = eqn.rewrite('divide_by_coeff', var='x')
print(eqn)