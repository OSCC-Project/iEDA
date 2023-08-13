import os
import re
from dataclasses import dataclass

import numpy as np


@dataclass
class InstanceConflict:
    name: str
    error: bool
    transactions: np.array
    conflicts: np.array
    conflicts_because_of_inactivity: np.array
    canceled_transactions: np.array
    fast_rounds: int
    medium_rounds: int
    exhaustive_rounds: int
    calls = np.array
    same_appearances = np.array
    conflict_appearances = np.array


presolver = {
    0: "colsingleton",
    1: "coefftightening",
    2: "propagation",

    3: "simpleprobing",
    4: "parallelrows",
    5: "parallelcols",
    6: "stuffing",
    7: "dualfix",
    8: "fixcontinuous",
    9: "simplifyineq",
    10: "doubletoneq",

    11: "implint",
    12: "domcol",
    13: "dualinfer",
    14: "probing",
    15: "substitution",
    16: "sparsify"
}

abbrevations = {
    0: "cs",
    1: "co",
    2: "cp",

    3: "sp",
    4: "pr",
    5: "pc",
    6: "st",
    7: "d",
    8: "f",
    9: "si",
    10: "dt",

    11: "ii",
    12: "dc",
    13: "di",
    14: "po",
    15: "su",
    16: "sp"
}

presolver_to_index = {
    "colsingleton": 0,
    "coefftightening": 1,
    "propagation": 2,

    "simpleprobing": 3,
    "parallelrows": 4,
    "parallelcols": 5,
    "stuffing": 6,
    "dualfix": 7,
    "fixcontinuous": 8,
    "simplifyineq": 9,
    "doubletoneq": 10,

    "implint": 11,
    "domcol": 12,
    "dualinfer": 13,
    "probing": 14,
    "substitution": 15,
    "sparsify": 16,
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


def round_to_string(x):
    if isinstance(x, int) or x.is_integer():
        return str(int(x))
    return "{:.2f}".format(x)


def percentage(quotient, divisor):
    if divisor == 0:
        return "NaN"
    return round_to_string(quotient / divisor * 100)


def parse_lock_information(directory: str):
    global root, files, filename, f, status, transactions, canceled_transactions, conflicts, conflicts_because_of_inactivity, split, errors, name
    parsed_results = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            print(filename)
            with open(directory + filename) as f:
                tsx_expr = re.compile("Transactions applied:")
                inactive_expr = re.compile("Transactions inactive:")
                dependency_expr = re.compile("Dependencies:")
                canceled_expr = re.compile("Transactions canceled:")
                successful_calls_expr = re.compile("Successful Calls")
                same_appearances_expr = re.compile("Same Appearances:")
                conflict_appearances_expr = re.compile("Conflict Appearances:")
                fast_expr = re.compile("FAST:")
                medium_expr = re.compile("MEDIUM:")
                exhaustive_expr = re.compile("EXHAUSTIVE:")
                information_expr = re.compile(" -")
                status = 0

                transactions = np.zeros(17)
                canceled_transactions = np.zeros(17, float)
                conflicts = np.zeros((17, 17), float)
                conflicts_because_of_inactivity = np.zeros((17, 17), float)
                fast_rounds = 0
                medium_rounds = 0
                exhaustive_rounds = 0
                calls = np.zeros(17)
                same_appearances = np.zeros((17, 17), float)
                conflict_appearances = np.zeros((17, 17), float)

                for line in f.readlines():
                    tsx_match = tsx_expr.match(line)
                    dependency_match = dependency_expr.match(line)
                    canceled_match = canceled_expr.match(line)
                    information_match = information_expr.match(line)
                    inactive_match = inactive_expr.match(line)
                    successful_calls_match = successful_calls_expr.match(line)
                    same_appearances_match = same_appearances_expr.match(line)
                    conflict_appearances_match = conflict_appearances_expr.match(line)
                    fast_match = fast_expr.match(line)
                    medium_match = medium_expr.match(line)
                    exhaustive_match = exhaustive_expr.match(line)
                    if tsx_match:
                        status = 1
                    elif dependency_match:
                        status = 2
                    elif canceled_match:
                        status = 3
                    elif inactive_match:
                        status = 4
                    elif successful_calls_match:
                        status = 5
                    elif same_appearances_match:
                        status = 6
                    elif conflict_appearances_match:
                        status = 7
                    elif fast_match:
                        fast_rounds = (float(line.replace('\n', '').split(' ')[1]))
                    elif medium_match:
                        medium_rounds = (float(line.replace('\n', '').split(' ')[1]))
                    elif exhaustive_match:
                        exhaustive_rounds = (float(line.replace('\n', '').split(' ')[1]))
                    elif information_match:
                        if status == 1:
                            split = line.replace(" -", "").replace("\n", "").split(" ")
                            transactions[presolver_to_index.get(split[0])] = float(split[1])
                        elif status == 3:
                            split = line.replace(" -", "").replace("\n", "").split(" ")
                            canceled_transactions[presolver_to_index.get(split[0])] = float(split[1])
                        elif status == 2:
                            split = line.replace(" -", "").replace("\n", "").split(" ")
                            conflicts[presolver_to_index.get(split[0]), presolver_to_index.get(split[1])] \
                                = float(split[2])
                        elif status == 4:
                            split = line.replace(" -", "").replace("\n", "").split(" ")
                            conflicts_because_of_inactivity[
                                presolver_to_index.get(split[0]), presolver_to_index.get(split[1])] \
                                = float(split[2])
                        elif status == 5:
                            split = line.replace(" -", "").replace("\n", "").split(" ")
                            calls[presolver_to_index.get(split[0])] = float(split[1])
                        elif status == 6:
                            split = line.replace(" -", "").replace("\n", "").split(" ")
                            same_appearances[
                                presolver_to_index.get(split[0]), presolver_to_index.get(split[1])] \
                                = float(split[2])
                        elif status == 7:
                            split = line.replace(" -", "").replace("\n", "").split(" ")
                            conflict_appearances[
                                presolver_to_index.get(split[0]), presolver_to_index.get(split[1])] \
                                = float(split[2])
                    elif bool(re.search(r'\d+', line)):
                        errors = int(line)
                        name = filename.split('.')[2].split("_")[1]
                        instance_conflict = InstanceConflict(name, errors > 0, transactions, conflicts,
                                                             conflicts_because_of_inactivity, canceled_transactions,
                                                             fast_rounds,
                                                             medium_rounds, exhaustive_rounds)
                        instance_conflict.calls = calls
                        instance_conflict.same_appearances = same_appearances
                        instance_conflict.conflict_appearances = conflict_appearances
                        parsed_results.append(instance_conflict)
                        # , calls, same_appearances, conflict_appearances)

                # if np.sum(transactions) == 0:
                #     #chromaticindex1024-7, cod105comp07-2idx, neos5, neos-1582420, sorrell3
                #     print("Instance didn't found a single reduction")
    return parsed_results


def calc_shifted_geometric_mean(list_of_numbers, shift_by=10.0):
    """ Return the shifted geometric mean of a list of numbers, where the additional shift defaults to
    10.0 and can be set via shiftby
    """
    geometric_mean = 1.0
    nitems = 0
    for number in list_of_numbers:
        nitems = nitems + 1
        nextnumber = number + shift_by
        geometric_mean = pow(geometric_mean, (nitems - 1) / float(nitems)) * pow(nextnumber, 1 / float(nitems))
    return geometric_mean - shift_by


def generate_table_rounds(velocity):
    r = ""
    for i in range(0, len(presolver)):
        res = ""
        if presolver_to_velocity.get(presolver.get(i)) != velocity:
            continue
        matches = list(filter(lambda x: x.calls[i] > 0, information))
        val = list(map(lambda x: x.calls[i], matches))
        res = res + (str(presolver.get(i)) + " (" + abbrevations.get(i) + ")" + " & " + round_to_string(sum(val)))
        for j in range(0, len(presolver)):
            if presolver_to_velocity.get(presolver.get(j)) != velocity:
                continue
            res = res + " & "
            if j > i:
                res = res + "- "
                continue
            matches_same = list(filter(lambda x: x.same_appearances[i][j] > 0, information))
            matches_con = list(filter(lambda x: x.conflict_appearances[i][j] > 0, information))
            if len(matches_con) == 0 and i == j:
                assert (len(matches_same) == 0)
                res = res + "0/" + round_to_string(sum(val))
            elif len(matches_con) > 0 and i == j:
                assert (len(matches_same) == 0)
                res = res + round_to_string(
                    sum(list(map(lambda x: x.conflict_appearances[i][j], matches_con)))) + "/" + round_to_string(sum(val))
            elif len(matches_same) == 0:
                assert (len(matches_con) == 0)
                res = res + " 0/0 "
            else:
                if len(matches_con) != 0:
                    res = res + round_to_string(
                        sum(list(map(lambda x: x.conflict_appearances[i][j], matches_con))))
                else:
                    res = res + "0"
                res = res + "/" + round_to_string(sum(list(map(lambda x: x.same_appearances[i][j], matches_same))))
        r = r + res + "\\\\\n"
    return r


def get_rounds(x, velocity):
    if velocity == 0:
        return x.fast_rounds
    if velocity == 1:
        return x.medium_rounds
    if velocity == 2:
        return x.exhaustive_rounds


def get_all_transactions(x, p):
    r = x.transactions[p]
    for j in range(0, len(presolver)):
        r = r + x.conflicts[p][j] + x.conflicts_because_of_inactivity[p][j]
    return r


if __name__ == '__main__':
    information = parse_lock_information("locks/")

    files = len(information)
    print()
    no_conflict = list(filter(lambda x: np.sum(x.conflicts) == 0, information))
    inact_and_no_c = list(
        filter(lambda x: np.sum(x.conflicts_because_of_inactivity) == 0 and np.sum(x.conflicts) == 0, information))
    sum_transaction = sum(np.sum(v.transactions) for v in information)
    sum_conflicts = sum(np.sum(v.conflicts) for v in information)
    sum_inactivites = sum(np.sum(v.conflicts_because_of_inactivity) for v in information)
    medium_rounds = sum(np.sum(v.medium_rounds) for v in information)
    exhaustive_rounds = sum(np.sum(v.exhaustive_rounds) for v in information)
    fast_rounds = sum(np.sum(v.fast_rounds) for v in information)

    print("instances:" + str(files))
    print("instances with no conflicts: " + str(len(inact_and_no_c)))
    print("instances with no real conflict: " + str(len(no_conflict)))
    print("conflict per tsx: " + str(percentage(sum_conflicts, sum_transaction)))
    print("inactivities per tsx: " + str(percentage(sum_inactivites, sum_transaction)))
    print("medium rounds " + str(medium_rounds))
    print("exhaustive rounds " + str(exhaustive_rounds))

    print("\n-----------------------------------------------------------------------")
    print("Plausibility checked failed for the following instances: ")
    for instance in information:
        for i in range(0, len(presolver)):
            for j in range(0, len(presolver)):
                if i == j:
                    if instance.same_appearances[i][j] > instance.calls[i]:
                        print(instance.name)
                elif instance.same_appearances[i][j] < instance.conflict_appearances[i][j]:
                    print(instance.name)

    print()
    print("fast Rounds: " + round_to_string(fast_rounds))
    print("-----------------------------------------------------------------------")
    print(generate_table_rounds(0))

    print()
    print("medium Rounds: " + round_to_string(medium_rounds))
    print("-----------------------------------------------------------------------")
    print(generate_table_rounds(1))

    print()
    print("exhaustive Rounds: " + round_to_string(exhaustive_rounds))
    print("-----------------------------------------------------------------------")
    print(generate_table_rounds(2))

    print()
    print("-----------------------------------------------------------------------")
    res = ""
    for i in range(0, len(presolver)):
        for j in range(0, len(presolver)):
            matches = list(
                filter(lambda x: x.conflicts[i][j] > 0 or x.conflicts_because_of_inactivity[i][j] > 0, information))
            if len(matches) == 0:
                continue
            all_tsx = sum(list(map(lambda x: get_all_transactions(x, i), information)))
            all_tsx_files = list(filter(lambda x: get_all_transactions(x, i) > 0, information))

            val = list(map(lambda x: x.conflicts[i][j] + x.conflicts_because_of_inactivity[i][j], matches))
            conflict_rate = list(
                map(lambda x: (x.conflicts[i][j] + x.conflicts_because_of_inactivity[i][j]) / get_all_transactions(x,
                                                                                                                   i),
                    matches))

            val_conflict = list(map(lambda x: x.conflicts[i][j], matches))
            val_inactive = list(map(lambda x: x.conflicts_because_of_inactivity[i][j], matches))
            avg_inactive = list(map(lambda x: x.conflicts_because_of_inactivity[i][j], matches))
            amount_of_conflicts = sum(val_conflict) + sum(val_inactive)

            res = res + str(presolver.get(i)) + "-" + str(presolver.get(j)) + " & "

            res = res + str(int(all_tsx)) + " & "

            res = res + str(int(amount_of_conflicts)) + " & "
            res = res + percentage(sum(conflict_rate), len(all_tsx_files)) + "\% & "
            # percentage
            # res = res + percentage(sum(val_conflict), amount_of_conflicts) + " & "
            res = res + percentage(sum(val_inactive), amount_of_conflicts) + "\% "
            res = res + "\\\\\n"
    print(res)
