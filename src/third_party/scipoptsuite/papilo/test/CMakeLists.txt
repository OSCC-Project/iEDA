find_package(Boost REQUIRED OPTIONAL_COMPONENTS serialization program_options iostreams)

if (TARGET papilolib)
    set(PAPILOLIB_TESTS
            "papilolib"
            "papilolib-no-rows"
            )
    set(PAPILOLIB_TEST_FILE PapiloLib.cpp)
    set(PAPILOLIB_TARGET papilolib)
else ()
    set(PAPILOLIB_TESTS "")
    set(PAPILOLIB_TEST_FILE "")
    set(PAPILOLIB_TARGET "")
endif ()

if (Boost_IOSTREAMS_FOUND AND Boost_SERIALIZATION_FOUND AND Boost_PROGRAM_OPTIONS_FOUND)
#    configure_file(resources/dual_fix_neg_inf.postsolve resources/dual_fix_neg_inf.postsolve COPYONLY)
#    configure_file(resources/dual_fix_pos_inf.postsolve resources/dual_fix_pos_inf.postsolve COPYONLY)
    configure_file(instances/dual_fix_neg_inf.mps resources/dual_fix_neg_inf.mps COPYONLY)
    set(BOOST_REQUIRED_TESTS
#            "finding-the-right-value-in-postsolve-for-a-column-fixed-pos-inf"
#            "finding-the-right-value-in-postsolve-for-a-column-fixed-neg-inf"
            "mps-parser-loading-simple-problem"
            )
    set(BOOST_REQUIRED_TEST_FILES
#            papilo/core/PostsolveTest.cpp
            papilo/io/MpsParserTest.cpp
            )
else ()
    set(BOOST_REQUIRED_TESTS "")
    set(BOOST_REQUIRED_TEST_FILES)
endif ()

add_executable(unit_test TestMain.cpp

        papilo/core/MatrixBufferTest.cpp
        papilo/core/SparseStorageTest.cpp
        papilo/core/PresolveTest.cpp
        papilo/core/ProblemUpdateTest.cpp
        papilo/misc/VectorUtilsTest.cpp

        papilo/presolve/CoefficientStrengtheningTest.cpp
        papilo/presolve/ConstraintPropagationTest.cpp
        papilo/presolve/DualFixTest.cpp
        papilo/presolve/DominatedColsTest.cpp
        papilo/presolve/FixContinuousTest.cpp
        papilo/presolve/FreeVarSubstitutionTest.cpp
        papilo/presolve/ImplIntDetectionTest.cpp
        papilo/presolve/ParallelRowDetectionTest.cpp
        papilo/presolve/ParallelColDetectionTest.cpp
        papilo/presolve/ProbingTest.cpp
        papilo/presolve/SingletonColsTest.cpp
        papilo/presolve/SingletonStuffingTest.cpp
        papilo/presolve/SimpleProbingTest.cpp
        papilo/presolve/SimpleSubstitutionTest.cpp
        papilo/presolve/SimplifyInequalitiesTest.cpp
        papilo/presolve/SparsifyTest.cpp

        ${BOOST_REQUIRED_TEST_FILES}
        ${PAPILOLIB_TEST_FILE})

target_link_libraries(unit_test papilo ${PAPILOLIB_TARGET} ${Boost_LIBRARIES})
if(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    target_link_libraries(unit_test log)
endif()

set(unit_tests
        "accurate-numerical-statistics"

        "matrix-buffer"
        "vector-comparisons"
        "matrix-comparisons"

        "replacing-variables-is-postponed-by-flag"
        "happy-path-replace-variable"
        "happy-path-substitute-matrix-coefficient-into-objective"
        "happy-path-aggregate-free-column"
        "presolve-activity-is-updated-correctly-huge-values"


        #ProblemUpdate
        "trivial-presolve-singleton-row"
        "trivial-presolve-singleton-row-pt-2"

        "problem-comparisons"

        #Coefficient-strengthening
        "happy-path-coefficient-strengthening"

        #ConstraintPropagation
        "constraint-propagation-happy-path"
        "constraint-propagation-no-tightening-for-lp"

        #DomCol
        "domcol-happy-path"
        "domcol-parallel-columns"
        "domcol-multiple-parallel-cols-generate_redundant-reductions"
        "domcol-multiple-column"

        #DualFix
        "dual-fix-happy-path"
        "dual-fix-trivial-column-presolve-finds-reduction"
        "dual-fix-no-dual-substitution-for-lp"
        "dual-fix-dual-substitution"
        "dual-fix-dual-substitution-rounding"
        "dual-fix-dual-substitution-unbounded-variables"
        "dual-fix-dual-substitution-equation"
        "dual-fix-infinity"

        # Fix Continuous
        "happy-path-presolve-fix-continuous"
        "happy-path-no-presolve-fix-continuous"

        #FreeVarSubstitution
        "happy-path-test-free-variable-detection"

        #Implied Integer
        "happy-path-implied-integer-detection"

        #Parallel Row Detection
        "parallel-row-unchanged"
        "parallel-row-two-equations-infeasible-second-row-dominant"
        "parallel-row-two-equations-infeasible-first-row-dominant"
        "parallel-row-two-equations-feasible-second-row-dominant"
        "parallel-row-two-equations-feasible-first-row-dominant"
        "parallel-row-two-inequalities-redundant-row-second-row-dominant"
        "parallel-row-two-inequalities-redundant-row-first-row-dominant"
        "parallel-row-two-inequalities-tighten-lower-bound-second-row-dominant"
        "parallel-row-two-inequalities-tighten-lower-bound-first-row-dominant"
        "parallel-row-two-inequalities-tighten-upper-bound-second-row-dominant"
        "parallel-row-two-inequalities-tighten-upper-bound-first-row-dominant"
        "parallel-row-two-inequalities-infeasible-first-row-dominant"
        "parallel-row-two-inequalities-infeasible-second-row-dominant"
        "parallel-row-two-inequalities-tighten-upper-bound-first-row-neg"
        "parallel-row-overwrite-inf-first-row-rhs-inf"
        "parallel-row-overwrite-inf-first-row-lhs-inf"
        "parallel-row-overwrite-inf-first-row-lhs-inf-neg-factor"
        "parallel-row-mixed-infeasible-first-row-equation"
        "parallel-row-best-bound-is-used-for-rhs-coeff-not-1"
        "parallel-row-best-bound-is-used-for-rhs-coeff"
        "parallel-row-mixed-second-row-equation"
        "parallel-row-mixed-infeasible-second-row-equation"
        "parallel-row-multiple-parallel-rows"

        #parallel Column Detection
        "parallel_col_detection_2_integer_columns"
        "parallel_col_detection_2_continuous_columns"
        "parallel_col_detection_int_cont_merge_possible"
        "parallel_col_detection_cont_int_merge_possible"
        "parallel_col_detection_cont_int_merge_failed"
        "parallel_col_detection_int_cont_merge_failed"
        "parallel_col_detection_int_merge_failed_hole"
        "parallel_col_detection_obj_not_parallel"
        "parallel_col_detection_multiple_parallel_columns"
        "parallel_col_detection_objective_zero"

        #Probing
        "happy-path-probing"
        "failed-path-probing-on-not-binary-variables"

        #Singleton Column
        "happy-path-singleton-column"
        "happy-path-singleton-column-equation"
        "happy-path-singleton-column-implied-bounds-negative-coeff-pos-bounds"
        "happy-path-singleton-column-implied-bounds-negative-coeff-neg-bounds"
        "happy-path-singleton-column-implied-bounds-positive-coeff-pos-bounds"
        "happy-path-singleton-column-implied-bounds-positive-coeff-neg-bounds"

        #Singleton Stuffing
        "singleton-stuffing-make-sure-to-first-set-bounds-to-infinity"

        #Simple Probing
        "happy-path-simple-probing"
        "happy-path-simple-probing-only-negative-coeff"
        "happy-path-simple-probing-only-binary-negative-coefficient"
        "happy-path-simple-probing-only-binary-positive-coefficient"

        #Simple Substitution
        "simple-substitution-happy-path-for-2-int"
        "simple-substitution-happy-path-for-2-continuous"
        "simple-substitution-happy-path-for-continuous-and-integer"
        "simple-substitution-happy-path-for-int-continuous-coeff"
        "simple-substitution-should_return_infeasible_if_gcd_of_coeff_is_in_rhs"
        "simple-substitution-should_return_feasible_if_gcd_of_coeff_is_in_rhs"
        "simple-substitution-failed-path-for-2-int"
        "simple-substitution-2-negative-integer"
        "simple-substitution-feasible-gcd"
        "simple-substitution-violated-gcd"

        #Simplify Inequality
        "happy-path-simplify-inequalities-only-greatest-divisor"
        "simplify_inequ_doesnt_lock_more_rows"
        "simplify_inequ_doesnt_apply_lb_and_ub_on_one_row"

        #Sparsify
        "happy-path-sparsify"
        "happy-path-sparsify-two-equalities"
        "failed-path-sparsify"

        "integration-test-for-flugpl"
        ${PAPILOLIB_TESTS}
        ${BOOST_REQUIRED_TESTS}
        )

# add a test to build the SCIP binary that all further tests depend on
add_test(NAME unit-test-build
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target unit_test
        )

set_tests_properties(unit-test-build
        PROPERTIES
        RESOURCE_LOCK unittestbin)

foreach (test ${unit_tests})
    add_test(NAME unit-test-${test} COMMAND unit_test ${test})

    set_tests_properties(unit-test-${test}
            PROPERTIES
            DEPENDS unit-test-build)
endforeach ()

if (TARGET papilo-executable)

    # MIP instances
    set(instances_MIP
            "MIP/bell5.mps\;8966406.49152"
            "MIP/blend2.mps\;7.598985"
            "MIP/dcmulti.mps\;188182"
            "MIP/egout.mps\;568.1007"
            "MIP/enigma.mps\;0"
            "MIP/flugpl.mps\;1201500"
            "MIP/gt2.mps\;21166"
            "MIP/lseu.mps\;1120"
            "MIP/misc03.mps\;3360"
            "MIP/p0548.mps\;8691"
            "MIP/rgn.mps\;82.19999924"
            )

    set(instances_gurobi_MIP
            "MIP/bell5.mps\;8966406.49152"
            "MIP/blend2.mps\;7.598985"
            #"MIP/dcmulti.mps\;188182" gurobi per default does not close the gap entirely
            "MIP/egout.mps\;568.1007"
            "MIP/enigma.mps\;0"
            "MIP/flugpl.mps\;1201500"
            "MIP/gt2.mps\;21166"
            "MIP/lseu.mps\;1120"
            "MIP/misc03.mps\;3360"
            "MIP/p0548.mps\;8691"
            "MIP/rgn.mps\;82.19999924"
            )


    # LP instances
    set(instances_LP
            "LP/adlittle.mps\;0.22549496316238038228101176621492e6"
            "LP/afiro.mps\;-0.46475314285714285714285714285714e3"
            "LP/agg.mps\;-0.35991767286576506712640824319636e8"
            "LP/beaconfd.mps\;0.335924858072e5"
            #TODO: MPS error while RHS identifier is missing
            #"LP/blend.mps\;-0.30812149845828220173774356124984e2"
            "LP/bore3d.mps\;0.13730803942084927215581987251301e4"
            "LP/brandy.mps\;0.15185098964881283835426751550618e4"
            "LP/capri.mps\;0.26900129137681610087717280693754e4"
            #TODO: objective value differs slightly
            #"LP/etamacro.mps\;-0.7557152333749133350792583667773e3"
            "LP/finnis.mps\;0.17279106559561159432297900375543e6"
            "LP/israel.mps\;-0.89664482186304572966200464196045e6"
            "LP/kb2.mps\;-0.17499001299062057129526866493726e4"
            "LP/lotfi.mps\;-0.2526470606188e2"
            "LP/recipe.mps\;-0.266616e3"
            "LP/sc105.mps\;-0.52202061211707248062628010857689e2"
            "LP/sc205.mps\;-0.52202061211707248062628010857689e2"
            "LP/sc50a.mps\;-0.64575077058564509026860413914575e2"
            "LP/sc50b.mps\;-0.7e2"
            "LP/scagr7.mps\;-0.2331389824330984e7"
            "LP/scagr25.mps\;-0.14753433060768523167790925075974e8"
            "LP/scfxm1.mps\;0.18416759028348943683579089143655e5"
            "LP/scorpion.mps\;0.18781248227381066296479411763586e4"
            "LP/scrs8.mps\;0.90429695380079143579923107948844e3"
            "LP/scsd1.mps\;0.86666666743333647292533502995263e1"
            "LP/seba.mps\;0.157116e5"
            "LP/share1b.mps\;-0.7658931857918568112797274346007e5"
            "LP/share2b.mps\;-0.41573224074141948654519910873841e3"
            "LP/shell.mps\;0.1208825346e10"
            "LP/vtp-base.mps\;0.1298314624613613657395984384889e6"
            )

    set(infeas_LP
            "LP/bgetam.mps"
            "LP/box1.mps"
            "LP/galenet.mps"
            "LP/gams10am.mps"
            "LP/refinery.mps"
            "LP/woodinfe.mps"
            "LP/klein1.mps"
            "LP/ex72a.mps"
            "LP/forest6.mps"
            )

    set(numtypes
            "d"
            "q"
            "r"
            )
    set(double_numtypes
            "d"
            )

    # add a test to build the SCIP binary that all further tests depend on
    add_test(NAME papilo-build
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target papilo-executable
            )
    # avoid that several build jobs try to concurrently build the binaries
    set_tests_properties(papilo-build
            PROPERTIES
            RESOURCE_LOCK papilobin)

    # macro to split an instance into its relevant information
    # - path
    # - optval
    # - basename
    macro(split_instance instance)
        list(GET instance 0 path)
        list(GET instance 1 optval)
        list(GET instance 1 optval)
        get_filename_component(basename ${path} NAME)
    endmacro(split_instance)

    ##solving these instances returns the defined optimal value for every numtype
    macro(add_instancetests instances numtypes settings)
        foreach (instance ${${instances}})
            split_instance(instance)
            file(TO_NATIVE_PATH "${PROJECT_SOURCE_DIR}/check/instances/${path}" instance_file)
            file(TO_NATIVE_PATH "${PROJECT_SOURCE_DIR}/settings/${settings}" settings_file)
            foreach (numtype ${${numtypes}})
                add_test(NAME ${numtype}-solve-${basename}-${settings}
                        COMMAND $<TARGET_FILE:papilo-executable> solve -a ${numtype} -f ${instance_file} -o ${optval} -p ${settings_file}
                        )
                set_tests_properties(${numtype}-solve-${basename}-${settings}
                        PROPERTIES
                        PASS_REGULAR_EXPRESSION "validation: SUCCESS"
                        DEPENDS papilo-build
                        )
            endforeach (numtype)
        endforeach (instance)
    endmacro(add_instancetests)

    ##solving returns infeasible for these instances and every potential numtype
    macro(add_infeasible_instancetests instances numtypes settings)
        foreach (instance ${${instances}})
            file(TO_NATIVE_PATH "${PROJECT_SOURCE_DIR}/check/instances/${instance}" instance_file)
            file(TO_NATIVE_PATH "${PROJECT_SOURCE_DIR}/settings/${settings}" settings_file)

            foreach (numtype ${${numtypes}})
                add_test(NAME ${numtype}-infeas-${instance}
                        COMMAND $<TARGET_FILE:papilo-executable> solve -a ${numtype} -f ${instance_file}  -p ${settings_file}
                        )
                set_tests_properties(${numtype}-infeas-${instance}
                        PROPERTIES
                        PASS_REGULAR_EXPRESSION "(presolving|solving) detected infeasible problem after [0-9].[0-9][0-9][0-9] seconds"
                        DEPENDS papilo-build
                        )
            endforeach (numtype)
        endforeach (instance)
    endmacro(add_infeasible_instancetests)


    # comparing the objective value for MIP instances and check if primal postsolve works
    if(PAPILO_HAVE_SCIP)
        add_instancetests(instances_MIP numtypes "default.set")
    elseif(PAPILO_HAVE_GUROBI)
        add_instancetests(instances_gurobi_MIP double_numtypes "default.set")
    endif()
    # comparing the objective value for LP instances and check if dual postsolve works (with/out basis calculation)
    if(PAPILO_HAVE_SOPLEX OR PAPILO_HAVE_HIGHS)
        add_instancetests(instances_LP numtypes "lp_presolvers_with_basis.set")
        add_instancetests(instances_LP numtypes "lp_presolvers_without_basis.set")
        add_infeasible_instancetests(infeas_LP numtypes "lp_presolvers_with_basis.set")
    elseif(PAPILO_HAVE_GLOP)
        add_instancetests(instances_LP double_numtypes "lp_presolvers_without_basis_glop.set")
        add_infeasible_instancetests(infeas_LP double_numtypes "lp_presolvers_with_basis.set")
    endif()
endif ()



set(filename_for_solutions "solutions.test")
# the following code block checks if a solution is contained in a reduced problem.
# The input for solutions.test file can be done via a file with the following format:
# PATH_TO_MPS_FILE_1,PATH_TO_SOL_FILE_1
# PATH_TO_MPS_FILE_2,PATH_TO_SOL_FILE_2
# ...
if (TARGET papilo-executable AND EXISTS "${PROJECT_SOURCE_DIR}/${filename_for_solutions}")

    FILE(READ "${PROJECT_SOURCE_DIR}/${filename_for_solutions}" content)
    string(REPLACE "\n" ";" instances ${content})

    macro(split_solution solution)
        string(REPLACE "," ";" sol ${solution})
        list(GET sol 0 path)
        list(GET sol 1 path_solution)
        get_filename_component(basename ${path} NAME)
    endmacro(split_solution)

    #solution is still contained after presolving
    macro(validate_solution instance settings)
        split_solution(${instance})
        file(TO_NATIVE_PATH "${path}" instance_file)
        file(TO_NATIVE_PATH "${path_solution}" solution_file)
        file(TO_NATIVE_PATH "${PROJECT_SOURCE_DIR}/settings/${settings}" settings_file)
        #TODO find better testname -> solution file can be occur multiple times
        add_test(NAME validate-solution-${settings}-${solution_file}
                COMMAND $<TARGET_FILE:papilo-executable> presolve -f ${instance_file} -b ${solution_file} -p ${settings_file}
                )
        set_tests_properties(validate-solution-${settings}-${solution_file}
                PROPERTIES
                PASS_REGULAR_EXPRESSION "validation: SUCCESS\n"
                DEPENDS papilo-build
                )
    endmacro(validate_solution)

    foreach (instance ${instances})
        validate_solution(${instance}, "no_strong_dual_reductions.set")
#        validate_solution(${instance}, "scip_no_strong_dual_reductions.set")
    endforeach (instance)
endif ()
