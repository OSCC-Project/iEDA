from dataclasses import dataclass


@dataclass
class Lock:
    index: int
    presolver: str


@dataclass
class Information:
    lock: bool
    conflict: bool
    conflicting_presolver: str
    representation: str
    modified_rows: list
    modified_columns: list
    bounds_locked: list
    inactive: bool
    inactive_columns: list
    redundant_rows: list


col_reduction = {
    -1: "NONE",
    -2: "OBJECTIVE",
    -3: "LOWER_BOUND",
    -4: "UPPER_BOUND",
    -5: "FIXED",
    -6: "LOCKED",
    -7: "LOCKED_STRONG",
    -8: "SUBSTITUTE",
    -9: "BOUNDS_LOCKED",
    -10: "REPLACE",
    -11: "SUBSTITUTE_OBJ",
    -12: "PARALLEL",
    -13: "IMPL_INT",
    -14: "FIXED_INFINITY",
}

row_reduction = {
    -1: "NONE",
    -2: "RHS",
    -3: "LHS",
    -4: "REDUNDANT",
    -5: "LOCKED",
    -6: "LOCKED_STRONG",
    -7: "RHS_INF",
    -8: "LHS_INF",
    -9: "SPARSIFY",
    -10: "RHS_LESS_RESTRICTIVE",
    -11: "LHS_LESS_RESTRICTIVE",
    -12: "REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE",
    -13: "SAVE_ROW"
}


def get_index(lock: Lock):
    return lock.index


amount_expecting_nones_col = 0
amount_expecting_nones_rows = 0


def check_if_row_is_locked(row: int, modified_rows: list, redundant_rows: list):
    if row in map(get_index, modified_rows):
        conflict = [x for x in modified_rows if x.index == row][0]
        return Information(True, True, conflict.presolver,
                           "DETECTED CONFLICT for row " + row.__str__() + " p " + conflict.presolver,
                           list(), list(), list(), False, list(), list())
    if row in map(get_index, redundant_rows):
        conflict = [x for x in redundant_rows if x.index == row][0]
        return Information(True, True, conflict.presolver,
                           "DETECTED CONFLICT for row " + row.__str__() + " p " + conflict.presolver,
                           list(), list(), list(), False, list(), list())
    else:
        return Information(True, False, "", "LOCKED row " + str(row), list(), list(), list(), False, list(), list())


def check_if_colum_is_locked(col: int, modified_columns: list):
    if col in map(get_index, modified_columns):
        conflict = [x for x in modified_columns if x.index == col][0]
        return Information(True, True, conflict.presolver,
                           "DETECTED CONFLICT for locking column " + col.__str__() + " presolver " + conflict.presolver,
                           list(), list(), list(), False, list(), list())
    else:
        return Information(True, False, "", "LOCKED column " + str(col.__str__()), list(), list(), list(), False,
                           list(), list())


def check_if_bounds_is_locked(col: int, bounds_locked: list):
    if col in map(get_index, bounds_locked):
        conflict = [x for x in bounds_locked if x.index == col][0]
        return Information(True, True, conflict.presolver,
                           "DETECTED CONFLICT for bounds column " + col.__str__() + " presolver " + conflict.presolver,
                           list(), list(), list(), False, list(), list())
    else:
        return Information(True, False, "", "BOUNDS LOCKED column " + str(col.__str__()), list(), list(), list(), False,
                           list(), list())


def evaluate(row: int, col: int, val: float, presolver: str, modified_columns: list, modified_rows: list,
             modified_var_bounds: list, inactive_columns: list, redundant_rows: list, conflict_detected: bool):
    global amount_expecting_nones_col
    global amount_expecting_nones_rows
    if (amount_expecting_nones_rows > 0 and not "NONE" == col_reduction.get(row)) or (
            amount_expecting_nones_col > 0 and not "NONE" == row_reduction.get(col)):
        print("NONE expected")

    conflicting: bool
    conflicting_presolver: None
    representation: ""
    modified_rows_in_tsx = []
    modified_columns_in_tsx = []
    modified_var_bounds_in_tsx = []
    deleted_columns_in_tsx = []
    redundant_rows_in_tsx = []

    conflict_presolver = ""
    redundant = False

    if row < 0:
        if "LOCKED" == col_reduction.get(row):
            return check_if_colum_is_locked(col, modified_columns)
        elif "BOUNDS_LOCKED" == col_reduction.get(row):
            return check_if_bounds_is_locked(col, modified_var_bounds)

        representation = ""
        if col in map(get_index, inactive_columns):
            s = [x for x in inactive_columns if x.index == col][0]
            conflict_presolver = s.presolver
            # exclude cases where doubletoneq generates conflicts with itself
            if presolver != "doubletoneq" and conflict_presolver != "doubletoneq":
                redundant = True
                representation = "REDUNDANT: "

        if "NONE" == col_reduction.get(row):
            redundant = False
            modified_columns_in_tsx.append(col)
            amount_expecting_nones_rows = 0
            representation = "REPLACE continue: * " + str(col) + " + " + str(val)
        elif "OBJECTIVE" == col_reduction.get(row):
            print("not implemented OBJECTIVE")
            print([row, col, val])
            print([row, col, val])
        elif "LOWER_BOUND" == col_reduction.get(row):
            modified_var_bounds_in_tsx.append(Lock(col, presolver))
            if not conflict_detected:
                redundant = False
                representation = "LOWER_BOUND of column " + col.__str__() + " to " + val.__str__()
            else:
                representation += "LOWER_BOUND of column " + col.__str__() + " to " + val.__str__()
        elif "UPPER_BOUND" == col_reduction.get(row):
            modified_var_bounds_in_tsx.append(Lock(col, presolver))
            if not conflict_detected:
                redundant = False
                representation = "UPPER_BOUND of column " + col.__str__() + " to " + val.__str__()
            else:
                representation += "UPPER_BOUND of column " + col.__str__() + " to " + val.__str__()
        elif "FIXED" == col_reduction.get(row):
            modified_var_bounds_in_tsx.append(Lock(col, presolver))
            deleted_columns_in_tsx.append(Lock(col, presolver))
            # TODO: currently double fixes are allowed
            if not conflict_detected:
                redundant = False
                representation = "FIXED column " + str(col) + " to " + str(val)
            else:
                representation += "FIXED column " + str(col) + " to " + str(val)
        elif "SUBSTITUTE" == col_reduction.get(row):
            if val in map(get_index, redundant_rows):
                conflict = [x for x in redundant_rows if x.index == val][0]
                conflict_presolver = conflict.presolver
                redundant = True
                representation = "REDUNDANT: SUBSTITUTE column " + col.__str__() + " with red. row " + val.__str__()
            else:
                deleted_columns_in_tsx.append(Lock(col, presolver))
                representation += "SUBSTITUTE column " + col.__str__() + " with row " + val.__str__()
        elif "SUBSTITUTE_OBJ" == col_reduction.get(row):
            deleted_columns_in_tsx.append(Lock(col, presolver))
            representation += "SUBSTITUTE_OBJ column " + col.__str__() + " with row " + val.__str__()
        elif "REPLACE" == col_reduction.get(row):
            #TODO:
            redundant = False
            modified_columns_in_tsx.append(col)
            amount_expecting_nones_rows = 1
            representation = "REPLACE variables " + str(col) + " with " + str(val)
        elif "PARALLEL" == col_reduction.get(row):
            if int(col) in map(get_index, inactive_columns):
                s = [x for x in inactive_columns if x.index == int(val)][0]
                conflict_presolver = s.presolver
                redundant = True
                representation = "REDUNDANT: PARALLEL colums " + str(int(val)) + "/" + str(col) + " p " + s.presolver
            else:
                modified_var_bounds_in_tsx.append(Lock(col, presolver))
                modified_var_bounds_in_tsx.append(Lock(int(val), presolver))
                # the column val in this case is substitued in this case there is no need to make any changes here
                # TODO: check if it is really the val
                deleted_columns_in_tsx.append(Lock(int(col), presolver))
                representation += "PARALLEL columns " + col.__str__() + " and " + val.__str__()
        elif "IMPL_INT" == col_reduction.get(row):
            representation += "IMPLIED INTEGER for column + " + col.__str__()
        elif "FIXED_INFINITY" == col_reduction.get(row):
            modified_var_bounds_in_tsx.append(Lock(col, presolver))
            representation += "FIXED column " + col.__str__() + " to infinity" + val.__str__()
        else:
            representation += "unknown case " + col_reduction.get(row) + " for col " + str(col.__str__())
    elif col < 0:
        if "LOCKED" == row_reduction.get(col):
            return check_if_row_is_locked(row, modified_rows, redundant_rows)
        elif "LOCKED_STRONG" == row_reduction.get(col):
            return check_if_row_is_locked(row, modified_rows, redundant_rows)

        representation = ""
        if row in map(get_index, redundant_rows):
            conflict = [x for x in redundant_rows if x.index == row][0]
            conflict_presolver = conflict.presolver
            redundant = True
            representation = "REDUNDANT: "

        if "RHS" == row_reduction.get(col):
            modified_rows_in_tsx.append(Lock(row, presolver))
            representation += "RHS " + row.__str__() + " to " + val.__str__()
        elif "LHS" == row_reduction.get(col):
            modified_rows_in_tsx.append(Lock(row, presolver))
            representation += "LHS " + row.__str__() + " to " + val.__str__()
        elif "RHS_LESS_RESTRICTIVE" == row_reduction.get(col):
            modified_rows_in_tsx.append(Lock(row, presolver))
            representation += "RHS_LESS_RESTRICTIVE " + row.__str__() + " to " + val.__str__()
        elif "LHS_LESS_RESTRICTIVE" == row_reduction.get(col):
            modified_rows_in_tsx.append(Lock(row, presolver))
            representation += "LHS_LESS_RESTRICTIVE " + row.__str__() + " to " + val.__str__()
        elif "RHS_INF" == row_reduction.get(col):
            modified_rows_in_tsx.append(Lock(row, presolver))
            representation += "RHS " + row.__str__() + " to infinity"
        elif "LHS_INF" == row_reduction.get(col):
            modified_rows_in_tsx.append(Lock(row, presolver))
            representation += "LHS " + row.__str__() + " to negative infinity"
        elif "REDUNDANT" == row_reduction.get(col):
            # TODO: actually it needs to lock the bounds
            redundant_rows_in_tsx.append(Lock(row, presolver))
            representation += "REDUNDANT row " + row.__str__()
        elif "REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE" == row_reduction.get(col):
            representation += "REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE"
        elif "SAVE_ROW" == row_reduction.get(col):
            representation += "SAVE_ROW"
        elif "SPARSIFY" == row_reduction.get(col):
            amount_expecting_nones_col = val
            representation += "SPARSIFY for row " + row.__str__()
        elif "NONE" == row_reduction.get(col):
            if amount_expecting_nones_col < 1:
                print("ERROR: unexpected none")
            else:
                redundant = False
                amount_expecting_nones_col = amount_expecting_nones_col - 1
                representation = "SPARSIFY uses row " + row.__str__() + " with factor " + val.__str__()
        else:
            representation = "unknown case " + col_reduction.get(row) + " for col " + str(col)
    else:
        modified_rows_in_tsx.append(Lock(row, presolver))
        modified_columns_in_tsx.append(Lock(col, presolver))
        representation = "CHANGE ENTRY (" + row.__str__() + "," + col.__str__() + ") to " + val.__str__()
    return Information(False, False, conflict_presolver, representation, modified_rows_in_tsx,
                       modified_columns_in_tsx,
                       modified_var_bounds_in_tsx, redundant, deleted_columns_in_tsx, redundant_rows_in_tsx)
