import re

from ipet.parsing.Solver import SCIPSolver

PRESOLVE_TIME_NAME = "presolve_time"

ROWS_NAME = "rows"
COLUMNS_NAME = "columns"
INT_COLUMNS_NAME = "int_columns"
CONT_COLUMNS_NAME = "cont_columns"
NON_ZEROS_NAME = "nonzeros"

RED_ROWS_NAME = "reduced_rows"
RED_COLUMNS_NAME = "reduced_columns"
RED_INT_COLUMNS_NAME = "reduced_int_columns"
RED_CONT_COLUMNS_NAME = "reduced_cont_columns"
RED_NON_ZEROS_NAME = "reduced_nonzeros"

PRESOLVING_ROUNDS_NAME = "presolv_rounds"

FAST_ROUNDS_NAME = "fast_rounds"
MEDIUM_ROUNDS_NAME = "medium_rounds"
EXHAUSTIVE_ROUNDS_NAME = "exhaustive_rounds"

COLUMNS_DELETED_NAME = "cols_del"
ROWS_DELETED_NAME = "rows_del"
BOUND_CHANGES_NAME = "chg_bound"
CHANGES_SIDES_NAME = "chg_sides"
CHANGED_COEFFICIENTS_NAME = "chg_coeff"
TSX_APPLIED_NAME = "tsx_applied"
TSX_CONFLICTS_NAME = "tsx_conflicts"

SOLVER_column_singleton = "colsingleton"
SOLVER_coeff_tightening = "coefftightening"
SOLVER_propagation = "propagation"
SOLVER_simple_probing = "simpleprobing"
SOLVER_stuffing = "stuffing"
SOLVER_dual_fix = "dualfix"
SOLVER_fix_continuous = "fixcontinuous"
SOLVER_parallel_rows = "parallelrows"
SOLVER_parallel_columns = "parallelcols"
SOLVER_doubletoneq = "doubletoneq"
SOLVER_simplify_inequality = "simplifyineq"
SOLVER_dual_infer = "dualinfer"
SOLVER_substitution = "substitution"
SOLVER_probing = "probing"
SOLVER_dom_colums = "domcol"
SOLVER_sparsify = "sparsify"

TSX_RATE_column_singleton = "tsx%colsingleton"
TSX_RATE_coeff_tightening = "tsx%coefftightening"
TSX_RATE_propagation = "tsx%propagation"
TSX_RATE_simple_probing = "tsx%simpleprobing"
TSX_RATE_stuffing = "tsx%stuffing"
TSX_RATE_dual_fix = "tsx%dualfix"
TSX_RATE_fix_continuous = "tsx%fixcontinuous"
TSX_RATE_parallel_rows = "tsx%parallelrows"
TSX_RATE_parallel_columns = "tsx%parallelcols"
TSX_RATE_doubletoneq = "tsx%doubletoneq"
TSX_RATE_simplify_inequality = "tsx%simplifyineq"
TSX_RATE_dual_infer = "tsx%dualinfer"
TSX_RATE_substitution = "tsx%substitution"
TSX_RATE_probing = "tsx%probing"
TSX_RATE_dom_colums = "tsx%domcol"
TSX_RATE_sparsify = "tsx%sparsify"

TIME_column_singleton = "time%colsingleton"
TIME_coeff_tightening = "time%coefftightening"
TIME_propagation = "time%propagation"
TIME_simple_probing = "time%simpleprobing"
TIME_stuffing = "time%stuffing"
TIME_dual_fix = "time%dualfix"
TIME_fix_continuous = "time%fixcontinuous"
TIME_parallel_rows = "time%parallelrows"
TIME_parallel_columns = "time%parallelcols"
TIME_doubletoneq = "time%doubletoneq"
TIME_simplify_inequality = "time%simplifyineq"
TIME_dual_infer = "time%dualinfer"
TIME_substitution = "time%substitution"
TIME_probing = "time%probing"
TIME_dom_colums = "time%domcol"
TIME_sparsify = "time%sparsify"

DEFAULT_VALUE = -1.0


class PapiloSolver(SCIPSolver):
    solverId = "PaPILO"
    # used to identify the solver
    recognition_expr = re.compile("starting presolve of problem")
    version_expr = re.compile("PaPILO version (\S+)")

    presolving_time_expr = re.compile("presolving finished after\s+(\S+)")
    presolving_time_inf_expr = re.compile("presolving detected infeasible problem after\s+(\S+)")
    presolving_time_unb_expr = re.compile("presolving detected unbounded problem after\s+(\S+)")
    presolving_time_unb_inf_expr = re.compile("presolving detected unbounded or infeasible problem after\s+(\S+)")

    floating_point_expr = "[-+]?[0-9]*\.?[0-9]*"

    rows_expr = re.compile("\s+rows:\s+(\S+)")
    columns_expr = re.compile("\s+columns:\s+(\S+)")
    int_columns_expr = re.compile("\s+int. columns:\s+(\S+)")
    cont_columns_expr = re.compile("\s+cont. columns:\s+(\S+)")
    non_zeros_expr = re.compile("\s+nonzeros:\s+(\S+)")

    fast_rounds_expr = re.compile("^(.*?)Fast\s+")
    medium_rounds_expr = re.compile("^(.*?)Medium\s+")
    exhaustive_rounds_expr = re.compile("^(.*?)Exhaustive")
    final_rounds_expr = re.compile("^(.*?)Final\s+")
    trivial_rounds_expr = re.compile("^(.*?)Trivial\s+")
    unchanged_rounds_expr = re.compile("^(.*?)Unchanged\s+")

    red_rows_expr = re.compile("\s+reduced rows:\s+(\S+)")
    red_columns_expr = re.compile("\s+reduced columns:\s+(\S+)")
    red_int_columns_expr = re.compile("\s+reduced int. columns:\s+(\S+)")
    red_cont_columns_expr = re.compile("\s+reduced cont. columns:\s+(\S+)")
    red_non_zeros_expr = re.compile("\s+reduced nonzeros:\s+(\S+)")

    presolving_rounds = re.compile("presolved\s+(\S+)")
    columns_deleted = re.compile("presolved \d+ rounds:\s+(\S+)")
    rows_deleted = re.compile("presolved \d+ rounds:\s+\d+ del cols,\s+(\S+)")
    bound_changes = re.compile("presolved \d+ rounds:\s+\d+ del cols,\s+\d+ del rows,\s+(\S+)")
    changed_sides = re.compile("presolved \d+ rounds:\s+\d+ del cols,\s+\d+ del rows,\s+\d+ chg bounds,\s+(\S+)")
    changed_coefficients = re.compile(
        "presolved \d+ rounds:\s+\d+ del cols,\s+\d+ del rows,\s+\d+ chg bounds,\s+\d+ chg sides,\s+(\S+)")
    transactions_applied = re.compile(
        "presolved \d+ rounds:\s+\d+ del cols,\s+\d+ del rows,\s+\d+ chg bounds,\s+\d+ chg sides,\s+\d+ chg coeffs,\s+(\S+)")
    transactions_conflicts = re.compile(
        "presolved \d+ rounds:\s+\d+ del cols,\s+\d+ del rows,\s+\d+ chg bounds,\s+\d+ chg sides,\s+\d+ chg coeffs,\s+\d+ tsx applied,\s+(\S+)")

    fast_rounds = 0
    medium_rounds = 0
    exhaustive_rounds = 0

    def __init__(self, **kw):
        super(PapiloSolver, self).__init__(**kw)

    def reset(self):
        super(PapiloSolver, self).reset()
        self.addData(PRESOLVE_TIME_NAME, DEFAULT_VALUE)

        self.addData(ROWS_NAME, DEFAULT_VALUE)
        self.addData(COLUMNS_NAME, DEFAULT_VALUE)
        self.addData(INT_COLUMNS_NAME, DEFAULT_VALUE)
        self.addData(CONT_COLUMNS_NAME, DEFAULT_VALUE)
        self.addData(NON_ZEROS_NAME, DEFAULT_VALUE)

        self.addData(RED_ROWS_NAME, DEFAULT_VALUE)
        self.addData(RED_COLUMNS_NAME, DEFAULT_VALUE)
        self.addData(RED_INT_COLUMNS_NAME, DEFAULT_VALUE)
        self.addData(RED_CONT_COLUMNS_NAME, DEFAULT_VALUE)
        self.addData(RED_NON_ZEROS_NAME, DEFAULT_VALUE)

        self.addData(PRESOLVING_ROUNDS_NAME, DEFAULT_VALUE)
        self.addData(COLUMNS_DELETED_NAME, DEFAULT_VALUE)
        self.addData(ROWS_DELETED_NAME, DEFAULT_VALUE)
        self.addData(CHANGES_SIDES_NAME, DEFAULT_VALUE)
        self.addData(CHANGED_COEFFICIENTS_NAME, DEFAULT_VALUE)
        self.addData(BOUND_CHANGES_NAME, DEFAULT_VALUE)
        self.addData(TSX_APPLIED_NAME, DEFAULT_VALUE)
        self.addData(TSX_CONFLICTS_NAME, DEFAULT_VALUE)
        fast_rounds = 0
        medium_rounds = 0
        exhaustive_rounds = 0

    def extractPrimalboundHistory(self, line):
        pass

    def extractDualboundHistory(self, line):
        pass

    def extractHistory(self, line):
        pass

    def extractOptionalInformation(self, line: str):
        if line.startswith("presolving finished"):
            self.extractByExpression(line, self.presolving_time_expr, PRESOLVE_TIME_NAME)
        elif line.startswith("presolving detected infeasible problem"):
            self.extractByExpression(line, self.presolving_time_inf_expr, PRESOLVE_TIME_NAME)
        elif line.startswith("presolving detected unbounded or infeasible problem"):
            self.extractByExpression(line, self.presolving_time_unb_inf_expr, PRESOLVE_TIME_NAME)
        elif line.startswith("presolving detected unbounded problem"):
            self.extractByExpression(line, self.presolving_time_unb_expr, PRESOLVE_TIME_NAME)

        self.extractByExpression(line, self.rows_expr, ROWS_NAME)
        self.extractByExpression(line, self.columns_expr, COLUMNS_NAME)
        self.extractByExpression(line, self.int_columns_expr, INT_COLUMNS_NAME)
        self.extractByExpression(line, self.cont_columns_expr, CONT_COLUMNS_NAME)
        self.extractByExpression(line, self.non_zeros_expr, NON_ZEROS_NAME)

        self.extractByExpression(line, self.red_rows_expr, RED_ROWS_NAME)
        self.extractByExpression(line, self.red_columns_expr, RED_COLUMNS_NAME)
        self.extractByExpression(line, self.red_int_columns_expr, RED_INT_COLUMNS_NAME)
        self.extractByExpression(line, self.red_cont_columns_expr, RED_CONT_COLUMNS_NAME)
        self.extractByExpression(line, self.red_non_zeros_expr, RED_NON_ZEROS_NAME)

        self.extractByExpression(line, self.presolving_rounds, PRESOLVING_ROUNDS_NAME)
        self.extractByExpression(line, self.columns_deleted, COLUMNS_DELETED_NAME)
        self.extractByExpression(line, self.rows_deleted, ROWS_DELETED_NAME)
        self.extractByExpression(line, self.bound_changes, BOUND_CHANGES_NAME)
        self.extractByExpression(line, self.changed_sides, CHANGES_SIDES_NAME)
        self.extractByExpression(line, self.changed_coefficients, CHANGED_COEFFICIENTS_NAME)
        self.extractByExpression(line, self.transactions_applied, TSX_APPLIED_NAME)
        self.extractByExpression(line, self.transactions_conflicts, TSX_CONFLICTS_NAME)

        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_column_singleton),
                                 TIME_column_singleton)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_coeff_tightening),
                                 TIME_coeff_tightening)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_probing), TIME_probing)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_stuffing), TIME_stuffing)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_sparsify), TIME_sparsify)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_substitution),
                                 TIME_substitution, )
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_dual_fix), TIME_dual_fix)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_dual_infer), TIME_dual_infer)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_dom_colums), TIME_dom_colums)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_fix_continuous),
                                 TIME_fix_continuous)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_simplify_inequality),
                                 TIME_simplify_inequality)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_doubletoneq), TIME_doubletoneq)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_parallel_columns),
                                 TIME_parallel_columns)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_parallel_rows),
                                 TIME_parallel_rows)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_propagation), TIME_propagation)
        self.extractByExpression(line, self.setup_time_expr_for_solver(SOLVER_simple_probing),
                                 TIME_simple_probing)

        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_column_singleton),
                                 TSX_RATE_column_singleton)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_coeff_tightening),
                                 TSX_RATE_coeff_tightening)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_probing), TSX_RATE_probing)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_stuffing), TSX_RATE_stuffing)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_sparsify), TSX_RATE_sparsify)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_substitution),
                                 TSX_RATE_substitution, )
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_dual_fix), TSX_RATE_dual_fix)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_dual_infer), TSX_RATE_dual_infer)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_dom_colums), TSX_RATE_dom_colums)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_fix_continuous),
                                 TSX_RATE_fix_continuous)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_simplify_inequality),
                                 TSX_RATE_simplify_inequality)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_doubletoneq), TSX_RATE_doubletoneq)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_parallel_columns),
                                 TSX_RATE_parallel_columns)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_parallel_rows),
                                 TSX_RATE_parallel_rows)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_propagation), TSX_RATE_propagation)
        self.extractByExpression(line, self.setup_tsx_rate_expr_for_solver(SOLVER_simple_probing),
                                 TSX_RATE_simple_probing)
        if self.trivial_rounds_expr.match(line) is not None:
            self.fast_rounds = 0
            self.medium_rounds = 0
            self.exhaustive_rounds = 0
        if self.fast_rounds_expr.match(line) is not None:
            self.fast_rounds = self.fast_rounds + 1
        if self.medium_rounds_expr.match(line) is not None:
            self.medium_rounds = self.medium_rounds + 1
        if self.exhaustive_rounds_expr.match(line) is not None:
            if self.unchanged_rounds_expr.match(line) is None:
                self.exhaustive_rounds = self.exhaustive_rounds + 1

        if self.final_rounds_expr.match(line) is not None:
            self.addData(FAST_ROUNDS_NAME, self.fast_rounds)
            self.addData(MEDIUM_ROUNDS_NAME, self.medium_rounds)
            self.addData(EXHAUSTIVE_ROUNDS_NAME, self.exhaustive_rounds)

    def setup_tsx_rate_expr_for_solver(self, name):
        return re.compile("\s+" + name + "\s+\d+\s+" + self.floating_point_expr + "\s+\d+\s+(\S+)")

    def setup_time_expr_for_solver(self, name):
        return re.compile(
            "\s+" + name + "\s+\d+\s+" + self.floating_point_expr + "\s+\d+\s+" + self.floating_point_expr + "\s+(\S+)")
