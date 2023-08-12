import re
from dataclasses import dataclass

import numpy as np

from lock import Lock
from lock import evaluate


@dataclass
class Stats:
    presolver: str
    tsx_found: int
    application_rate: float


presolver = {
    0: "coefftightening",
    1: "colsingleton",
    2: "domcol",
    3: "doubletoneq",
    4: "dualfix",
    5: "dualinfer",
    6: "fixcontinuous",
    7: "parallelcols",
    8: "parallelrows",
    9: "probing",
    10: "simpleprobing",
    11: "simplifyineq",
    12: "implint",
    13: "propagation",
    14: "sparsify",
    15: "stuffing",
    16: "substitution"
}

# Fast =0, Medium = 1, Exhaustive = 2
presolver_to_velocity = {
    "coefftightening": 0,
    "colsingleton": 0,
    "domcol": 2,
    "doubletoneq": 1,
    "dualfix": 1,
    "dualinfer": 2,
    "fixcontinuous": 1,
    "parallelcols": 1,
    "parallelrows": 1,
    "probing": 2,
    "simpleprobing": 1,
    "simplifyineq": 1,
    "implint": 2,
    "propagation": 0,
    "sparsify": 2,
    "stuffing": 1,
    "substitution": 2
}

presolver_to_index = {
    "coefftightening": 0,
    "colsingleton": 1,
    "domcol": 2,
    "doubletoneq": 3,
    "dualfix": 4,
    "dualinfer": 5,
    "fixcontinuous": 6,
    "parallelcols": 7,
    "parallelrows": 8,
    "probing": 9,
    "simpleprobing": 10,
    "simplifyineq": 11,
    "implint": 12,
    "propagation": 13,
    "sparsify": 14,
    "stuffing": 15,
    "substitution": 16
}

modified_rows = []
modified_columns = []
inactive_columns = []
redundant_rows = []
bounds_locked = []

tsx_modified_rows = []
tsx_modified_columns = []
tsx_inactive_columns = []
tsx_redundant_rows = []
tsx_bounds_locked = []

current_velocity = -1
rounds_per_presolver = np.zeros([len(presolver)])
same_appeareances_per_presolver = np.zeros([len(presolver), len(presolver)])
conflict_appearances_per_presolver = np.zeros([len(presolver), len(presolver)])

# auxiliary variable for rounds
presolver_this_rounds = np.zeros([len(presolver)])
conflicts_this_rounds = np.zeros([len(presolver), len(presolver)])

# for verifying
amount_expecting_nones_col = 0
amount_expecting_nones_rows = 0


def clear_locks(copy: bool):
    global current_velocity, modified_rows, modified_columns, bounds_locked, presolver_this_rounds, \
        same_appeareances_per_presolver, conflicts_this_rounds, conflict_appearances_per_presolver, \
        rounds_per_presolver
    modified_rows.clear()
    modified_columns.clear()
    bounds_locked.clear()
    inactive_columns.clear()
    redundant_rows.clear()
    current_velocity = -1

    if copy:
        for j in range(0, len(presolver_this_rounds)):
            if presolver_this_rounds[j] == 1:
                rounds_per_presolver[j] = rounds_per_presolver[j] + 1
        copy_appearances_to_matrix(presolver_this_rounds, same_appeareances_per_presolver)
        for i in range(0, len(conflicts_this_rounds)):
            for j in range(0, len(conflicts_this_rounds)):
                if conflicts_this_rounds[i][j] == 1:
                    conflict_appearances_per_presolver[i][j] = conflict_appearances_per_presolver[i][j] + 1
    presolver_this_rounds = np.zeros([len(presolver)])
    conflicts_this_rounds = np.zeros([len(presolver), len(presolver)])


def copy_appearances_to_matrix(array, matrix):
    for i in range(len(array)):
        if array[i] == 1:
            for j in range(i + 1, len(array)):
                if array[j] == 1:
                    matrix[i][j] = matrix[i][j] + 1
                    matrix[j][i] = matrix[j][i] + 1


def clear_tsx_locks():
    tsx_modified_columns.clear()
    tsx_bounds_locked.clear()
    tsx_modified_rows.clear()
    tsx_inactive_columns.clear()
    tsx_redundant_rows.clear()


def copy_tsx_locks_to_all_locks():
    for i in tsx_modified_columns:
        modified_columns.append(i)
    for i in tsx_modified_rows:
        modified_rows.append(i)
    for i in tsx_bounds_locked:
        bounds_locked.append(i)
    for i in tsx_inactive_columns:
        inactive_columns.append(i)
    for i in tsx_redundant_rows:
        redundant_rows.append(i)


def parse_string(expr, line: str):
    result = expr.match(line).groups()[0].split(",")
    del result[-1]
    return list(map(int, result))


def is_statistics(line: str):
    for p in presolver.values():
        if line.__contains__(p):
            return True
    return False


def verify(expected_stats, applied_tsx, canceled_tsx, dependency_matrix, inactive_matrix):
    conflict_tsx = dependency_matrix.sum(1)
    inactive_tsx = inactive_matrix.sum(1)
    successful = True
    for expected_stat in expected_stats:
        index = presolver_to_index.get(expected_stat.presolver)
        upperbound_application_rate = (expected_stat.application_rate + 0.1) * expected_stat.tsx_found / 100
        lowerbound_application_rate = (expected_stat.application_rate - 0.1) * expected_stat.tsx_found / 100
        found_tsx = applied_tsx[index] + canceled_tsx[index] + conflict_tsx[index] + inactive_tsx[index]

        if expected_stat.tsx_found != found_tsx:
            print("ERROR: TSX DIFFERENCE " + str(expected_stat.tsx_found) + " vs " + str(found_tsx) +
                  " for " + expected_stat.presolver)
            successful = False
        elif lowerbound_application_rate > applied_tsx[index] or upperbound_application_rate < applied_tsx[index]:
            print("ERROR BY APPLICATION RATE for " + expected_stat.presolver +
                  " " + str(applied_tsx[index] / found_tsx * 100) +
                  " instead of " + str(expected_stat.application_rate) + "")
            successful = False
    return successful


def run_analysis_for_file(filename: str, noisy=False):
    global current_velocity, rounds_per_presolver, conflict_appearances_per_presolver, same_appeareances_per_presolver
    with open(filename) as f:
        round_expr = re.compile("round\s+(\S+)")
        conflict_expr = re.compile("CONFLICT\s+(\S+)")
        reduction_expr = re.compile("row\s+(\S+)")
        reduction2_expr = re.compile("row [+-]?\d+ col\s+(\S+)")
        reduction3_expr = re.compile("row [+-]?\d+ col [+-]?\d+ val\s+(\S+)")
        presolver_expr = re.compile("Presolver\s+(\S+)")
        modified_rows_expr = re.compile("modified rows:\s+(\S+)")
        modified_columns_expr = re.compile("modified columns:\s+(\S+)")
        tsx_expr = re.compile("tsx")
        canceled_expr = re.compile("canceled")

        # stats
        dependency_matrix = np.zeros([len(presolver), len(presolver)])
        inactive_matrix = np.zeros([len(presolver), len(presolver)])
        applied_tsx = np.zeros([len(presolver)])
        canceled_tsx = np.zeros([len(presolver)])
        rounds = np.zeros([3])

        rounds_per_presolver = np.zeros([len(presolver)])
        same_appeareances_per_presolver = np.zeros([len(presolver), len(presolver)])
        conflict_appearances_per_presolver = np.zeros([len(presolver), len(presolver)])

        expected_stats = []
        conflict_expected = False
        conflict_detected = False
        inactive = False
        current_presolver = ""
        presolver_found_transaction_in_this_round = False
        errors = 0

        reductions = 0
        inactive_reductions = 0
        index_conflict_presolver = 0
        index_inactive_presolver = 0

        tsx_can_be_applied_if_not_canceled = False

        for line in f.readlines():
            reduction_match = reduction_expr.match(line)
            round_match = round_expr.match(line)

            if tsx_can_be_applied_if_not_canceled:
                presolver_index = presolver_to_index.get(current_presolver)
                if canceled_expr.match(line):
                    if noisy:
                        print("  last TSX canceled")
                    canceled_tsx[presolver_index] = canceled_tsx[presolver_index] + 1
                else:
                    if not presolver_found_transaction_in_this_round:
                        presolver_found_transaction_in_this_round = True
                        presolver_this_rounds[presolver_index] = 1
                    presolver_index = presolver_to_index.get(current_presolver)
                    if reductions == inactive_reductions:
                        inactive = True
                    if conflict_expected and (conflict_expected != (conflict_detected or inactive)):
                        print("ERROR: conflict in " + current_presolver + " for row " + str(row) + " col " + str(
                            col) + " val " + str(val))
                        errors = errors + 1
                    elif inactive:
                        inactive_matrix[presolver_index][index_inactive_presolver] \
                            = 1 + inactive_matrix[presolver_index][index_inactive_presolver]
                        if conflicts_this_rounds[presolver_index][index_conflict_presolver] == 0:
                            conflicts_this_rounds[presolver_index][index_inactive_presolver] = 1
                    elif conflict_detected:
                        dependency_matrix[presolver_index][index_conflict_presolver] \
                            = 1 + dependency_matrix[presolver_index][index_conflict_presolver]
                        if conflicts_this_rounds[presolver_index][index_conflict_presolver] == 0:
                            conflicts_this_rounds[presolver_index][index_conflict_presolver] = 1
                    elif not conflict_expected and not inactive:
                        copy_tsx_locks_to_all_locks()
                        applied_tsx[presolver_index] = applied_tsx[presolver_index] + 1

                clear_tsx_locks()
                conflict_expected = False
                inactive = False
                conflict_detected = False
                tsx_can_be_applied_if_not_canceled = False
                reductions = 0
                inactive_reductions = 0
                index_conflict_presolver = 0
                index_inactive_presolver = 0

            if round_match:
                current_round = int(round_match.groups()[0])
                if noisy:
                    print("-------------------------------")
                    print("-------------------------------")
                    if "Fast" in line:
                        print("current round (Fast) " + current_round.__str__())
                    elif "Medium" in line:
                        print("current round (Medium) " + current_round.__str__())
                    elif "Exhaustive" in line:
                        print("current round (Exhaustive) " + current_round.__str__())
                    elif "Trivial" in line:
                        print("current round (Trivial) " + current_round.__str__())
                    print("-------------------------------")
                update_rounds(line, rounds)
                clear_locks(True)
            elif conflict_expr.match(line):
                conflict_expected = True
                if noisy:
                    print("CONFLICT OCCURRED")
            elif reduction_match:
                row = int(reduction_match.groups()[0])
                col = int(reduction2_expr.match(line).groups()[0])
                val = float(reduction3_expr.match(line).groups()[0])
                information = evaluate(row, col, val, current_presolver, modified_columns, modified_rows, bounds_locked,
                                       inactive_columns, redundant_rows, conflict_detected)
                for i in information.modified_columns:
                    tsx_modified_columns.append(i)
                for i in information.modified_rows:
                    tsx_modified_rows.append(i)
                for i in information.bounds_locked:
                    tsx_bounds_locked.append(i)
                for i in information.inactive_columns:
                    tsx_inactive_columns.append(i)
                for i in information.redundant_rows:
                    tsx_redundant_rows.append(i)
                if noisy:
                    print("   " + str(information.representation))

                if information.conflict and not conflict_detected:
                    conflict_detected = True
                    index_conflict_presolver = presolver_to_index.get(information.conflicting_presolver)

                if not information.lock:
                    reductions = reductions + 1
                    if information.inactive:
                        inactive_reductions = inactive_reductions + 1
                        if not inactive:
                            index_inactive_presolver = presolver_to_index.get(information.conflicting_presolver)

            elif presolver_expr.match(line):
                current_presolver = line.split()[1]
                velocity = presolver_to_velocity.get(current_presolver)
                if current_velocity == -1:
                    current_velocity = velocity
                elif not current_velocity == velocity:
                    clear_locks(False)
                    current_velocity = velocity
                    if noisy:
                        print("   TO NEXT ROUND")
                if noisy:
                    print(current_presolver)
                presolver_found_transaction_in_this_round = False
            elif tsx_expr.match(line):
                tsx_can_be_applied_if_not_canceled = True
                if noisy:
                    print("  -")
            elif modified_columns_expr.match(line):
                if not conflict_detected:
                    splitted_values = parse_string(modified_columns_expr, line)
                    for i in splitted_values:
                        modified_columns.append(Lock(i, current_presolver))
            elif modified_rows_expr.match(line):
                if not conflict_detected:
                    splitted_values = parse_string(modified_rows_expr, line)
                    for i in splitted_values:
                        modified_rows.append(Lock(i, current_presolver))

            elif is_statistics(line):
                stats_by_presolver = list(filter(None, line.split(" ")))
                # expected_stats.append(
                #     Stats(stats_by_presolver[0], int(stats_by_presolver[3]), float(stats_by_presolver[4])))

        verification_successful = verify(expected_stats, applied_tsx, canceled_tsx, dependency_matrix, inactive_matrix)
        if not verification_successful:
            errors = errors + 1
        clear_locks(True)
        if applied_tsx.max() > 0:  # and noisy:
            print("\nTransactions applied:")
            for i in range(len(applied_tsx)):
                if applied_tsx[i] > 0:
                    print(" -" + presolver.get(i), applied_tsx[i])

        if dependency_matrix.max() > 0:  # and noisy:
            print("\nDependencies:")
            for idx, x in np.ndenumerate(dependency_matrix):
                if x > 0:
                    print(" -" + presolver.get(idx[0]), presolver.get(idx[1]), x)

        if canceled_tsx.max() > 0 and noisy:
            print("\nTransactions canceled:")
            for i in range(len(canceled_tsx)):
                if canceled_tsx[i] > 0:
                    print(" -" + presolver.get(i), canceled_tsx[i])

        if inactive_matrix.max() > 0 and noisy:
            print("\nConflicts because of inactivity:")
            for idx, x in np.ndenumerate(inactive_matrix):
                if x > 0:
                    print(" -" + presolver.get(idx[0]), presolver.get(idx[1]), x)

        if noisy:
            print("\nFAST: " + str(rounds[0]) + "\nMEDIUM: " + str(rounds[1]) + "\nEXHAUSTIVE: " + str(rounds[2]))
            print("\nRounds: ")
            for idx, x in np.ndenumerate(rounds_per_presolver):
                if x > 0:
                    print(" -" + presolver.get(idx[0]) + " " + str(x))

            print("\nSuccessful Calls: ")
            for i in range(len(rounds_per_presolver)):
                if rounds_per_presolver[i] > 0:
                    print(" -" + presolver.get(i), rounds_per_presolver[i])

            print("\nSame Appearances: ")
            for idx, x in np.ndenumerate(same_appeareances_per_presolver):
                if x > 0:
                    print(" -" + presolver.get(idx[0]), presolver.get(idx[1]), x)

            print("\nConflict Appearances: ")
            for idx, x in np.ndenumerate(conflict_appearances_per_presolver):
                if x > 0:
                    print(" -" + presolver.get(idx[0]), presolver.get(idx[1]), x)

        if errors > 0:
            print("potential errors " + errors.__str__())
        clear_locks(False)
        clear_tsx_locks()
        # print(same_appeareances_per_presolver)
        # print(conflict_appearances_per_presolver)
        return applied_tsx, dependency_matrix, canceled_tsx, inactive_matrix, errors, rounds, rounds_per_presolver, \
               same_appeareances_per_presolver, conflict_appearances_per_presolver


def update_rounds(line, rounds):
    if "Fast" in line:
        rounds[0] = rounds[0] + 1
    elif "Medium" in line:
        rounds[1] = rounds[1] + 1
    elif "Exhaustive" in line:
        rounds[2] = rounds[2] + 1
