include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# interface function for setting common library properties
#
function(setLibProperties targetname outputname)
    set_target_properties(${targetname} PROPERTIES
        OUTPUT_NAME ${outputname}
        MACOSX_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endfunction(setLibProperties)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED on)

set(gcgsources
    benders_gcg.c
    bendersplugins.c
    branch_empty.c
    branch_bpstrong.c
    branch_generic.c
    branch_orig.c
    branch_relpsprob.c
    branch_ryanfoster.c
    class_conspartition.cpp
    class_indexpartition.cpp
    miscvisualization.cpp
    class_pricingcontroller.cpp
    class_pricingtype.cpp
    class_partialdecomp.cpp
    class_detprobdata.cpp
    class_stabilization.cpp
    class_varpartition.cpp
    clscons_miplibconstypes.cpp
    clscons_nnonzeros.cpp
    clscons_consnamenonumbers.cpp
    clscons_consnamelevenshtein.cpp
    clscons_scipconstypes.cpp
    clscons_gamsdomain.cpp
    clscons_gamssymbol.cpp
    clsvar_gamsdomain.cpp
    clsvar_gamssymbol.cpp
    clsvar_objvalues.cpp
    clsvar_scipvartypes.cpp
    clsvar_objvaluesigns.cpp
    colpool.c
    cons_decomp.cpp
    cons_integralorig.c
    cons_masterbranch.c
    cons_origbranch.c
    dec_compgreedily.cpp
    dec_connected_noNewLinkingVars.cpp
    dec_connectedbase.cpp
    dec_consclass.cpp
    dec_constype.cpp
    dec_dbscan.cpp
    dec_densemasterconss.cpp
    dec_generalmastersetcover.cpp
    dec_generalmastersetpack.cpp
    dec_generalmastersetpart.cpp
    dec_hcgpartition.cpp
    dec_hrcgpartition.cpp
    dec_hrgpartition.cpp
    dec_mastersetcover.cpp
    dec_mastersetpack.cpp
    dec_mastersetpart.cpp
    dec_mst.cpp
    dec_neighborhoodmaster.cpp
    dec_postprocess.cpp
    dec_staircase_lsp.cpp
    dec_stairheur.cpp
    dec_varclass.cpp
    decomp.c
    dialog_explore.cpp
    dialog_gcg.c
    dialog_graph.cpp
    dialog_master.c
    disp_gcg.c
    disp_master.c
    event_bestsol.c
    event_display.c
    event_mastersol.c
    event_relaxsol.c
    event_solvingstats.c
    gcgcol.c
    gcg_general.c
    gcggithash.c
    gcgheur.c
    gcgsepa.c
    gcgplugins.c
    gcgpqueue.c
    gcgsort.c
    gcgvar.c
    graph/graph_gcg.cpp
    graph/graph_tclique.cpp
    graph/inst.cpp
    graph/weights.cpp
    heur_gcgcoefdiving.c
    heur_gcgdins.c
    heur_gcgfeaspump.c
    heur_gcgfracdiving.c
    heur_gcgguideddiving.c
    heur_gcglinesdiving.c
    heur_gcgpscostdiving.c
    heur_gcgrens.c
    heur_gcgrins.c
    heur_gcgrounding.c
    heur_gcgshifting.c
    heur_gcgsimplerounding.c
    heur_gcgveclendiving.c
    heur_gcgzirounding.c
    heur_greedycolsel.c
    heur_mastercoefdiving.c
    heur_masterdiving.c
    heur_masterfracdiving.c
    heur_masterlinesdiving.c
    heur_mastervecldiving.c
    heur_origdiving.c
    heur_relaxcolsel.c
    heur_restmaster.c
    heur_setcover.c
    heur_xpcrossover.c
    heur_xprins.c
    masterplugins.c
    misc.c
    nodesel_master.c
    objdialog.cpp
    params_visu.c
    presol_roundbound.c
    pricer_gcg.cpp
    pricestore_gcg.c
    pricingjob.c
    pricingprob.c
    reader_blk.cpp
    reader_cls.cpp
    reader_dec.cpp
    reader_gp.cpp
    reader_ref.c
    reader_tex.cpp
    relax_gcg.c
    scip_misc.c
    scoretype.c
    sepa_basis.c
    sepa_master.c
    solver.c
    solver_knapsack.c
    solver_mip.c
    stat.c
)

set(gcgheaders
    branch_empty.h
    branch_bpstrong.h
    branch_generic.h
    branch_orig.h
    branch_relpsprob.h
    branch_ryanfoster.h
    class_conspartition.h
    class_indexpartition.h
    miscvisualization.h
    class_pricingcontroller.h
    class_pricingtype.h
    class_partialdecomp.h
    class_detprobdata.h
    class_stabilization.h
    class_varpartition.h
    clscons_consnamelevenshtein.h
    clscons_consnamenonumbers.h
    clscons_gamsdomain.h
    clscons_gamssymbol.h
    clscons_miplibconstypes.h
    clscons_nnonzeros.h
    clscons_scipconstypes.h
    clsvar_gamsdomain.h
    clsvar_gamssymbol.h
    clsvar_objvalues.h
    clsvar_objvaluesigns.h
    clsvar_scipvartypes.h
    colpool.h
    cons_decomp.h
    cons_decomp.hpp
    cons_integralorig.h
    cons_masterbranch.h
    cons_origbranch.h
    dec_compgreedily.h
    dec_connected_noNewLinkingVars.h
    dec_connectedbase.h
    dec_consclass.h
    dec_constype.h
    dec_dbscan.h
    dec_densemasterconss.h
    dec_generalmastersetcover.h
    dec_generalmastersetpack.h
    dec_generalmastersetpart.h
    dec_hcgpartition.h
    dec_hrcgpartition.h
    dec_hrgpartition.h
    dec_mastersetcover.h
    dec_mastersetpack.h
    dec_mastersetpart.h
    dec_mst.h
    dec_neighborhoodmaster.h
    dec_postprocess.h
    dec_staircase_lsp.h
    dec_stairheur.h
    dec_varclass.h
    decomp.h
    def.h
    dialog_gcg.h
    dialog_graph.h
    dialog_master.h
    disp_gcg.h
    disp_master.h
    event_bestsol.h
    event_display.h
    event_mastersol.h
    event_relaxsol.h
    event_solvingstats.h
    gcg.h
    gcgcol.h
    gcg_general.h
    gcggithash.h
    gcgplugins.h
    gcgpqueue.h
    gcgsort.h
    heur_gcgcoefdiving.h
    heur_gcgdins.h
    heur_gcgfeaspump.h
    heur_gcgfracdiving.h
    heur_gcgguideddiving.h
    heur_gcglinesdiving.h
    heur_gcgpscostdiving.h
    heur_gcgrens.h
    heur_gcgrins.h
    heur_gcgrounding.h
    heur_gcgshifting.h
    heur_gcgsimplerounding.h
    heur_gcgveclendiving.h
    heur_gcgzirounding.h
    heur_greedycolsel.h
    heur_mastercoefdiving.h
    heur_masterdiving.h
    heur_masterfracdiving.h
    heur_masterlinesdiving.h
    heur_mastervecldiving.h
    heur_origdiving.h
    heur_relaxcolsel.h
    heur_restmaster.h
    heur_setcover.h
    heur_xpcrossover.h
    heur_xprins.h
    masterplugins.h
    nodesel_master.h
    objdialog.h
    objpricer_gcg.h
    params_visu.h
    presol_roundbound.h
    pricer_gcg.h
    pricestore_gcg.h
    pricingjob.h
    pricingprob.h
    pub_bliss.h
    pub_colpool.h
    pub_decomp.h
    pub_gcgcol.h
    pub_gcgheur.h
    pub_gcgsepa.h
    pub_gcgpqueue.h
    pub_gcgvar.h
    pub_pricingjob.h
    pub_pricingprob.h
    pub_solver.h
    reader_blk.h
    reader_cls.h
    reader_dec.h
    reader_gp.h
    reader_ref.h
    reader_tex.h
    relax_gcg.h
    scip_misc.h
    scoretype.h
    sepa_basis.h
    sepa_master.h
    solver.h
    solver_cliquer.h
    solver_knapsack.h
    solver_mip.h
    solver_xyz.h
    stat.h
    struct_branchgcg.h
    struct_colpool.h
    struct_decomp.h
    struct_detector.h
    struct_gcgcol.h
    struct_gcgpqueue.h
    struct_pricestore_gcg.h
    struct_pricingjob.h
    struct_pricingprob.h
    struct_solver.h
    struct_vardata.h
    type_branchgcg.h
    type_classifier.h
    type_colpool.h
    type_consclassifier.h
    type_decomp.h
    type_detector.h
    type_gcgcol.h
    type_gcgpqueue.h
    type_masterdiving.h
    type_origdiving.h
    type_parameter.h
    type_pricestore_gcg.h
    type_pricingjob.h
    type_pricingprob.h
    type_pricingstatus.h
    type_scoretype.h
    type_solver.h
    type_varclassifier.h
    wrapper_partialdecomp.h
)

set(gcggraphheaders
    graph/bipartitegraph.h
    graph/bipartitegraph_def.h
    graph/bridge.h
    graph/columngraph.h
    graph/columngraph_def.h
    graph/graph.h
    graph/graph_def.h
    graph/graph_gcg.h
    graph/graph_interface.h
    graph/graph_tclique.h
    graph/graphalgorithms.h
    graph/graphalgorithms_def.h
    graph/hypercolgraph.h
    graph/hypercolgraph_def.h
    graph/hypergraph.h
    graph/hypergraph_def.h
    graph/hyperrowcolgraph.h
    graph/hyperrowcolgraph_def.h
    graph/hyperrowgraph.h
    graph/hyperrowgraph_def.h
    graph/matrixgraph.h
    graph/matrixgraph_def.h
    graph/rowgraph.h
    graph/rowgraph_def.h
    graph/weights.h
   )

if(SYM STREQUAL "bliss")
   set(gcgsources ${gcgsources}
      bliss.cpp
      bliss_automorph.cpp
      dec_isomorph.cpp
   )
   set(gcgheaders ${gcgheaders} 
      bliss_automorph.h
      bliss_automorph.hpp
      dec_isomorph.h
   )
endif()

if(CLIQUER_FOUND)
   set(gcgsources ${gcgsources}
      solver_cliquer.c
   )
endif()

if(CPLEX_FOUND)
   set(gcgsources ${gcgsources}
      solver_cplex.c
   )
   set(gcgheaders ${gcgheaders}
      solver_cplex.h
   )
endif()

if(HMETIS_FOUND)
   include_directories(${CMAKE_CURRENT_BINARY_DIR})
   add_definitions(-DHMETIS_HEADER)
   configure_file(hmetis.h.in hmetis.h)
endif()

# all source files should be compiled with a c++ compiler
if(CXXONLY)
    set_source_files_properties(main.c ${gcgsources} PROPERTIES LANGUAGE CXX)

    # for the clang compiler this suppresses the warnings about treating 'c' input as 'c++' when CXXONLY is enabled
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
       add_compile_options(-x c++)
    endif()
endif()

add_library(libgcg ${gcgsources})
setLibProperties(libgcg "gcg")

target_link_libraries(libgcg PRIVATE ${SCIP_LIBRARIES} ${SYM_LIBRARIES})

if(SHARED)
    set(SCIP_NEEDED 0)
    target_link_libraries(libgcg PRIVATE ${SCIP_PIC_LIBRARIES} ${SYM_PIC_LIBRARIES} ${CLIQUER_PIC_LIBRARIES} ${CPLEX_PIC_LIBRARIES} ${GSL_PIC_LIBRARIES})
    add_executable(gcg main.c ${gcgsources})
else()
    set(SCIP_NEEDED ${SCIP_FOUND})
    target_link_libraries(libgcg PRIVATE ${SCIP_LIBRARIES} ${SYM_LIBRARIES} ${CLIQUER_LIBRARIES} ${CPLEX_LIBRARIES} ${GSL_LIBRARIES})
    add_executable(gcg main.c)
    target_link_libraries(gcg libgcg)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    find_package(Sanitizers)
    add_sanitizers(gcg)
endif()

target_compile_definitions(gcg PRIVATE EXTERN=extern)

target_link_libraries(gcg ${SCIP_LIBRARIES} ${SYM_LIBRARIES} ${CLIQUER_LIBRARIES} ${CPLEX_LIBRARIES} ${GSL_LIBRARIES})

add_dependencies(libgcg gcg_update_githash)
add_dependencies(gcg gcg_update_githash)

set_target_properties(libgcg PROPERTIES
    VERSION ${GCG_VERSION_MAJOR}.${GCG_VERSION_MINOR}.${GCG_VERSION_PATCH}.${GCG_VERSION_SUB}
    SOVERSION ${GCG_VERSION_MAJOR}.${GCG_VERSION_MINOR}
    INSTALL_RPATH_USE_LINK_PATH TRUE)

# set the install rpath to the installed destination
set_target_properties(gcg PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE)

# install the header files of gcg
install(FILES ${gcgheaders} DESTINATION include/gcg)
# separate installation of graph headers to preserve the folder structure
# see e.g. here: https://stackoverflow.com/a/11097012
install(FILES ${gcggraphheaders} DESTINATION include/gcg/graph)
if(HMETIS_FOUND)
   install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hmetis.h DESTINATION include/gcg)
endif()

# install the binary and the library to appropriate locations and add them to an export group
install(TARGETS gcg libgcg EXPORT gcg-targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

# Add all targets to the build-tree export set
export(TARGETS gcg libgcg
  FILE "${CMAKE_BINARY_DIR}/gcg-targets.cmake")

#make scip dir absolute for the config file
if(SCIP_FOUND)
    get_filename_component(CONF_SCIP_DIR ${SCIP_DIR} REALPATH BASE_DIR ${CMAKE_SOURCE_DIR})
endif()

#configure the config file for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src")
configure_file(${PROJECT_SOURCE_DIR}/gcg-config.cmake.in
  "${CMAKE_BINARY_DIR}/gcg-config.cmake" @ONLY)

#configure the config file for the install
set(CONF_INCLUDE_DIRS "\${CMAKE_CURRENT_LIST_DIR}/../../../include")
configure_file(${PROJECT_SOURCE_DIR}/gcg-config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gcg-config.cmake" @ONLY)

# install the targets of the gcg export group and the config file so that other projects
# can link easily against gcg
install(EXPORT gcg-targets FILE gcg-targets.cmake DESTINATION lib/cmake/gcg)
install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gcg-config.cmake" DESTINATION lib/cmake/gcg)

