include(CTest)

#
# define the set of feasible instances
#
set(successInstances
    "adlittle\;0.22549496316238038228101176621492e6"
    "afiro\;-0.46475314285714285714285714285714e3"
    "agg\;-0.35991767286576506712640824319636e8"
    "beaconfd\;0.335924858072e5"
    "blend\;-0.30812149845828220173774356124984e2"
    "bore3d\;0.13730803942084927215581987251301e4"
    "brandy\;0.15185098964881283835426751550618e4"
    "capri\;0.26900129137681610087717280693754e4"
    "etamacro\;-0.7557152333749133350792583667773e3"
    "finnis\;0.17279106559561159432297900375543e6"
    "grow7\;-0.47787811814711502616766956242865e8"
    "israel\;-0.89664482186304572966200464196045e6"
    "kb2\;-0.17499001299062057129526866493726e4"
    "lotfi\;-0.2526470606188e2"
    "recipe\;-0.266616e3"
    "sc105\;-0.52202061211707248062628010857689e2"
    "sc205\;-0.52202061211707248062628010857689e2"
    "sc50a\;-0.64575077058564509026860413914575e2"
    "sc50b\;-0.7e2"
    "scagr25\;-0.14753433060768523167790925075974e8"
    "scagr7\;-0.2331389824330984e7"
    "scfxm1\;0.18416759028348943683579089143655e5"
    "scorpion\;0.18781248227381066296479411763586e4"
    "scrs8\;0.90429695380079143579923107948844e3"
    "scsd1\;0.86666666743333647292533502995263e1"
    "seba\;0.157116e5"
    "share1b\;-0.7658931857918568112797274346007e5"
    "bgetam\;+infinity"
    "box1\;+infinity"
    "ex72a\;+infinity"
    "forest6\;+infinity"
    "galenet\;+infinity"
    "gams10am\;+infinity"
    "klein1\;+infinity"
    "refinery\;+infinity"
    "woodinfe\;+infinity"
    "gas11\;-infinity"
    )

set(failInstances
    "share2b\;0.123456789e5"
    "shell\;+infinity"
    "vtp-base\;-infinity"
    )

#
# define the set of settings
#
set(settings
    default
    default-col
    default-row
    devex
    internalsimplifier
    papilosimplifier
    polish1
    polish2
    steep
    )

# use GREATER to work with old cmake versions
if(BOOST)
   #need boost to run exact
   set(settings
       ${settings}
       exact)
if(${Boost_VERSION_MACRO} GREATER "106999")
    set(settings
        ${settings}
        mpf)
   if(SOPLEX_HAVE_FLOAT128)
    set(settings
        ${settings}
        quad)
   endif()
endif()
endif()

#
# define a macro to add tests
#
# add_instancetests takes an instance group and a status
# that SoPlex should report as arguments
#
macro(add_instancetests instances solutionstatus)
#
# loop over the instances
#
foreach(instance ${${instances}})
    #
    # loop over all settings
    #
    foreach(setting ${settings})
        #
        # treat the instance as a tuple (list) of two values
        #
        list(GET instance 0 name)
        list(GET instance 1 optval)
        #
        # specify the instance and the settings load command for soplex
        #
        set(inst "${PROJECT_SOURCE_DIR}/check/instances/${name}.mps")
        set(optvalCommand "--extsol=${optval}")
        if(${setting} STREQUAL "default")
            set(loadCommand "")
        else()
            set(loadCommand "--loadset=${PROJECT_SOURCE_DIR}/settings/${setting}.set")
        endif()

        #
        # add a test and test properties
        #
        add_test(NAME ${name}-${setting} COMMAND $<TARGET_FILE:soplex> ${optvalCommand} ${loadCommand} ${inst})
        set_tests_properties (${name}-${setting} PROPERTIES PASS_REGULAR_EXPRESSION "Validation          : ${solutionstatus}")
    endforeach(setting)
endforeach(instance)
endmacro(add_instancetests)

#
# add tests for success and fail instances
#
add_instancetests( successInstances "Success" )
add_instancetests( failInstances "Fail" )