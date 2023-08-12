include(CTest)

add_dependencies(check
   gcg_check)

add_custom_target(gcg_check
   COMMAND ${CMAKE_CTEST_COMMAND} -R "-default" --output-on-failure --schedule-random
   DEPENDS gcg)

file(RELATIVE_PATH GCG_REL_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gcg)
add_custom_target(gcg_cluster
   COMMAND ${CMAKE_COMMAND} -E env "GCG_BINARY=${GCG_REL_PATH}" ${PROJECT_SOURCE_DIR}/check/cmake_wrapper.sh cluster
   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/check/
   DEPENDS gcg)

add_custom_target(visu
   COMMAND ${CMAKE_COMMAND} -E env "GCG_BINARY=${GCG_REL_PATH}" ${PROJECT_SOURCE_DIR}/check/cmake_wrapper.sh visu
   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/check/
   DEPENDS gcg)

#
# define the instance sets and settings.
#
# semicolon '\;' is used to split an instance and its optimal objective value
# For infeasible instances, '+infinity' is used (or '-infinity' in case of maximization)

# In order to run a sub-suite, use -R. `make check` just runs anything matching default.*.
#

#
# MIPLIB instances
#

set(settings_MIPLIB
    "default"
)

set(instances_MIPLIB
   "instances/miplib/noswot.mps\;-41"
   )

#
# BPP instances
#

set(settings_BPP
    "default"
)

set(instances_BPP
   "instances/bpp/N1C1W4_M.BPP.lp\;41"
   "instances/bpp/N1C2W2_O.BPP.lp\;29"
   "instances/bpp/N1C3W1_A.lp\;16"
   )

#
# CPMP instances
#

set(settings_CPMP
    "default"
)

set(instances_CPMP
   "instances/cpmp/p1250-2.lp\;412"
   "instances/cpmp/p1650-2.txt.lp\;336"
   "instances/cpmp/p2050-1.txt.lp\;266"
   )

#
# CS instances
#

set(settings_CS
    "default"
)

set(instances_CS
   "instances/cs/TEST0055.lp\;11"
   "instances/cs/TEST0059.lp\;11"
   )

#
# GAP instances
#

set(settings_GAP
    "default"
)

set(instances_GAP
   "instances/gap/gap4_2.txt.lp\;644"
   "instances/gap/gap8_4.txt.lp\;1117"
   )

#
# MKP instances
#

set(settings_MKP
    "default"
)

set(instances_MKP
   "instances/mkp/strong_d_45_15_12.lp\;-8703"
   "instances/mkp/strong_s_75_15_14.lp\;-7158"
   "instances/mkp/strong_s_75_15_18.lp\;-8360"
   )

#
# add a test to build the GCG binary that all further tests depend on
#
add_test(NAME gcg-build
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target gcg
        )

#
# avoid that several build jobs try to concurrently build the GCG library
# note that this ressource lock name is not the actual libgcg target
#
set_tests_properties(gcg-build
                    PROPERTIES
                        RESOURCE_LOCK libgcg
                    )


#
# macro to split an instance into its relevant information
# - path
# - optval
# - basename
#
macro(split_instance instance)
    list(GET instance 0 path)
    list(GET instance 1 optval)
    get_filename_component(basename ${path} NAME)
endmacro(split_instance)

macro(add_instancetests instances settings prefix)
#
# loop over the instances
#
    foreach(instance ${${instances}})
        split_instance(instance)
        #
        # loop over all settings
        #
        foreach(setting ${${settings}})
            #
            # treat the instance as a tuple (list) of two values
            #
            add_test(NAME ${prefix}-${setting}-${basename}
                    COMMAND $<TARGET_FILE:gcg> -f ${PROJECT_SOURCE_DIR}/check/${path} -s ${PROJECT_SOURCE_DIR}/settings/${setting}.set -o ${optval} ${optval}
                    )
            set_tests_properties(${prefix}-${setting}-${basename}
                                PROPERTIES
                                    PASS_REGULAR_EXPRESSION "Validation         : Success"
                                    DEPENDS gcg-build
                                    RESOURCE_LOCK libscip
                                )
        endforeach(setting)
    endforeach(instance)
endmacro(add_instancetests)

add_instancetests(instances_MIPLIB settings_MIP "MIPLIB")
add_instancetests(instances_BPP settings_BPP "BPP")
add_instancetests(instances_CPMP settings_CPMP "CPMP")
add_instancetests(instances_CS settings_CS "CS")
add_instancetests(instances_GAP settings_GAP "GAP")
add_instancetests(instances_MKP settings_MKP "MKP")

#this test cannot work, since gcg's main.c does not handle SIGTERMS
#
## add a test for handling of the SIGTERM signal. The test uses the timeout command that
## is only available on Linux, that is available on MAC OS as "gtimeout" after installing
## the 'coreutils' package
##
#if (UNIX)
#    if (APPLE)
#        set(timeoutcommand "gtimeout")
#    else ()
#        set(timeoutcommand "timeout")
#    endif (APPLE)
#    add_test(NAME signal-handling-sigterm-blend2
#            COMMAND ${timeoutcommand} -sSIGTERM 1 $<TARGET_FILE:gcg> -f ${CMAKE_CURRENT_SOURCE_DIR}/instances/miplib/noswot.mps
#            )
#    set_tests_properties(signal-handling-sigterm-blend2
#                        PROPERTIES
#                            DEPENDS gcg-build
#                            PASS_REGULAR_EXPRESSION "termination signal received"
#                            #
#                            # I assume that this test takes longer than 2 seconds
#                            #
#                            )
#endif (UNIX)

