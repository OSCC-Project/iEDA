set(MultilevelDatastructureSources
      static_hypergraph_factory.cpp
      static_hypergraph.cpp
      graph.cpp
      fixed_vertex_support.cpp)

set(MultilevelGraphDatastructureSources
      static_graph_factory.cpp
      static_graph.cpp
      graph.cpp
      fixed_vertex_support.cpp)

set(NLevelDatastructureSources
      contraction_tree.cpp
      dynamic_hypergraph.cpp
      dynamic_hypergraph_factory.cpp
      graph.cpp
      incident_net_array.cpp
      fixed_vertex_support.cpp)

set(NLevelGraphDatastructureSources
      contraction_tree.cpp
      dynamic_graph.cpp
      dynamic_graph_factory.cpp
      graph.cpp
      dynamic_adjacency_array.cpp
      fixed_vertex_support.cpp)


foreach(modtarget IN LISTS PARTITIONING_SUITE_TARGETS)
  target_sources(${modtarget} PRIVATE ${MultilevelGraphDatastructureSources})
  target_sources(${modtarget} PRIVATE ${NLevelGraphDatastructureSources})
  target_sources(${modtarget} PRIVATE ${MultilevelDatastructureSources})
  target_sources(${modtarget} PRIVATE ${NLevelDatastructureSources})
endforeach()

set(ToolsDatastructureSources
      static_hypergraph_factory.cpp
      static_hypergraph.cpp
      static_graph_factory.cpp
      static_graph.cpp
      contraction_tree.cpp
      dynamic_hypergraph.cpp
      dynamic_hypergraph_factory.cpp
      incident_net_array.cpp
      contraction_tree.cpp
      dynamic_graph.cpp
      dynamic_graph_factory.cpp
      dynamic_adjacency_array.cpp
      fixed_vertex_support.cpp)

foreach(modtarget IN LISTS TOOLS_TARGETS)
  target_sources(${modtarget} PRIVATE ${ToolsDatastructureSources})
endforeach()