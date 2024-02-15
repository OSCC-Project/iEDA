set(RefinementSources
        fm/fm_commons.cpp
        fm/multitry_kway_fm.cpp
        fm/localized_kway_fm_core.cpp
        fm/global_rollback.cpp
        fm/sequential_twoway_fm_refiner.cpp
        label_propagation/label_propagation_refiner.cpp
        rebalancing/simple_rebalancer.cpp
        rebalancing/advanced_rebalancer.cpp
        deterministic/deterministic_label_propagation.cpp
        flows/refiner_adapter.cpp
        flows/problem_construction.cpp
        flows/scheduler.cpp
        flows/quotient_graph.cpp
        flows/flow_refiner.cpp
        flows/sequential_construction.cpp
        flows/parallel_construction.cpp
        flows/flow_hypergraph_builder.cpp
        )

set(Km1Sources
        gains/km1/km1_gain_cache.cpp)

set(SoedSources
        gains/soed/soed_gain_cache.cpp)

set(CutSources
        gains/cut/cut_gain_cache.cpp)

set(CutGraphSources
        gains/cut_for_graphs/cut_gain_cache_for_graphs.cpp)

set(SteinerTreeSources
        gains/steiner_tree/steiner_tree_gain_cache.cpp
        gains/steiner_tree/steiner_tree_flow_network_construction.cpp)

set(SteinerTreeGraphSources
        gains/steiner_tree_for_graphs/steiner_tree_gain_cache_for_graphs.cpp
        gains/steiner_tree_for_graphs/steiner_tree_flow_network_construction_for_graphs.cpp)

foreach(modtarget IN LISTS PARTITIONING_SUITE_TARGETS)
    target_sources(${modtarget} PRIVATE ${RefinementSources})
    target_sources(${modtarget} PRIVATE ${Km1Sources})
    target_sources(${modtarget} PRIVATE ${CutSources})

    if ( KAHYPAR_ENABLE_SOED_METRIC )
      target_sources(${modtarget} PRIVATE ${SoedSources})
    else ()
      target_sources(mtkahypar PRIVATE ${SoedSources})
      target_sources(mtkahypar_python PRIVATE ${SoedSources})
    endif()

    if ( KAHYPAR_ENABLE_STEINER_TREE_METRIC )
      target_sources(${modtarget} PRIVATE ${SteinerTreeSources})
    else ()
      target_sources(mtkahypar PRIVATE ${SteinerTreeSources})
      target_sources(mtkahypar_python PRIVATE ${SteinerTreeSources})
    endif()

    if ( KAHYPAR_ENABLE_GRAPH_PARTITIONING_FEATURES )
      target_sources(${modtarget} PRIVATE ${CutGraphSources})
      if ( KAHYPAR_ENABLE_STEINER_TREE_METRIC )
        target_sources(${modtarget} PRIVATE ${SteinerTreeGraphSources})
      else ()
        target_sources(mtkahypar PRIVATE ${SteinerTreeGraphSources})
        target_sources(mtkahypar_python PRIVATE ${SteinerTreeGraphSources})
      endif()
    else ()
      target_sources(mtkahypar PRIVATE ${CutGraphSources})
      target_sources(mtkahypar_python PRIVATE ${CutGraphSources})
      target_sources(mtkahypar PRIVATE ${SteinerTreeGraphSources})
      target_sources(mtkahypar_python PRIVATE ${SteinerTreeGraphSources})
    endif()
endforeach()
