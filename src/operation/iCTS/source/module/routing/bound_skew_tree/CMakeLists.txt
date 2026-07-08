add_library(
  icts_source_module_routing_bst
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/BSTRouter.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/algorithm/BinaryTopology.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/algorithm/BottomUpMergeBalance.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/algorithm/BottomUpMergeInfeasibility.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/algorithm/BottomUpMergeJoining.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/algorithm/BoundSkewTree.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/algorithm/BoundSkewTreeImpl.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/algorithm/BstPipeline.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/algorithm/TopDownEmbedding.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/clock_tree_conversion/BSTRouterBinaryTopology.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/clock_tree_conversion/BSTRouterExport.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/component/Components.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/geometry/GeomCalc.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/geometry/GeomCalcLine.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/geometry/GeomCalcPointRegion.cc
  ${ICTS_MODULE_ROUTING}/bound_skew_tree/geometry/GeomCalcTransformedRect.cc
)

target_link_libraries(
  icts_source_module_routing_bst
  PRIVATE
  icts_source_database_routing
  icts_source_database_spatial
  icts_source_utils_geometry
  log
)

target_include_directories(
  icts_source_module_routing_bst
  PUBLIC
  ${ICTS_MODULE_ROUTING}
)

if(DEBUG_ICTS_SOURCE_MODULE_ROUTING_BST)
  if(MSVC)
    target_compile_options(icts_source_module_routing_bst PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_module_routing_bst PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_module_routing_bst PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
