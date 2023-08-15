import os

import numpy as np

from papilo_parser import presolver
from papilo_parser import run_analysis_for_file

files = {
    "p0548": "results/alexander.short.10_p0548.papilo.alexander-Latitude-5310.default.out",
    "rgn": "results/alexander.short.11_rgn.papilo.alexander-Latitude-5310.default.out",
    "bell5": "results/alexander.short.1_bell5.papilo.alexander-Latitude-5310.default.out",
    "blend2": "results/alexander.short.2_blend2.papilo.alexander-Latitude-5310.default.out",
    "dcmulti": "results/alexander.short.3_dcmulti.papilo.alexander-Latitude-5310.default.out",
    "egout": "results/alexander.short.4_egout.papilo.alexander-Latitude-5310.default.out",
    "enigma": "results/alexander.short.5_enigma.papilo.alexander-Latitude-5310.default.out",
    "flugpl": "results/alexander.short.6_flugpl.papilo.alexander-Latitude-5310.default.out",
    "gt2": "results/alexander.short.7_gt2.papilo.alexander-Latitude-5310.default.out",
    "lseu": "results/alexander.short.8_lseu.papilo.alexander-Latitude-5310.default.out",
    "misc03": "results/alexander.short.9_misc03.papilo.alexander-Latitude-5310.default.out"}


def get_filename(name: str, dir: str):
    for root, dirs, files in os.walk(dir):
        for filename in files:
            if filename.endswith(".out") and filename.__contains__(name):
                return dir + "/" + filename


def to_string(matrix, name):
    string = "\n" + name
    for idx, x in np.ndenumerate(matrix):
        if x > 0:
            string = string + "\n -" + presolver.get(idx[0]) + " " + presolver.get(idx[1]) + " " + str(x)
    return string


def to_string_list(vector: list, name: str):
    string = "\n" + name
    for i in range(len(vector)):
        if vector[i] > 0:
            string = string + "\n -" + presolver.get(i).__str__() + " " + str(vector[i])
    return string


def rounds_to_string(rounds):
    return "\nFAST: " + str(rounds[0]) + "\nMEDIUM: " + str(rounds[1]) + "\nEXHAUSTIVE: " + str(rounds[2])


if __name__ == '__main__':
    # (applied_tsx, dependency_matrix, canceled_tsx, inactive_tsx, errors, rounds, rounds_per_presolver,
    #  same_appeareances_per_presolver, conflict_appearances_per_presolver) = run_analysis_for_file("test.out", True)

    main_dependency_matrix = np.zeros([17, 17])
    files_with_errors = []
    source_folder = "results"
    destination_folder = "locks/"
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            if filename.endswith(".out") and not filename.startswith("check"):
                print("\n" + filename)
                if os.path.exists(destination_folder + filename.replace(".out", ".txt")):
                    print("existing")
                    continue
                try:
                    (applied_tsx, dependency_matrix, canceled_tsx, inactive_matrix, errors, rounds,
                     rounds_per_presolver, same_appeareances_per_presolver, conflict_appearances_per_presolver) = \
                        run_analysis_for_file(root + "/" + filename, False)
                    f = open(destination_folder + filename.replace(".out", ".txt"), "w")
                    main_dependency_matrix = np.add(main_dependency_matrix, dependency_matrix)
                    if dependency_matrix.max() > 0:
                        print(to_string(dependency_matrix, "Dependencies:"))
                    if errors > 0:
                        files_with_errors.append(filename)
                    f.write(to_string_list(applied_tsx, "Transactions applied:") + "\n" +
                            to_string(dependency_matrix, "Dependencies:") + "\n" +
                            to_string_list(canceled_tsx, "Transactions canceled:") + "\n" +
                            to_string(inactive_matrix, "Transactions inactive:") + "\n" +
                            rounds_to_string(rounds) + "\n" +
                            to_string_list(rounds_per_presolver, "Successful Calls:") + "\n" +
                            to_string(same_appeareances_per_presolver, "Same Appearances:") + "\n" +
                            to_string(conflict_appearances_per_presolver, "Conflict Appearances:") + "\n" +
                            "\n" + str(errors))
                except RuntimeError:
                    files_with_errors.append(filename)
                    print("\n Runtime error occurred")
    if main_dependency_matrix.max() > 0:
        print("Dependencies:")
        for idx, x in np.ndenumerate(main_dependency_matrix):
            if x > 0:
                print(" -" + presolver.get(idx[0]), presolver.get(idx[1]), x)

    print("Errors")
    print(files_with_errors)
