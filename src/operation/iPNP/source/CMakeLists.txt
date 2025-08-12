add_subdirectory(config)
add_subdirectory(data_manager)
add_subdirectory(module)

add_library(pnp iPNP.cpp)

target_link_libraries(
  pnp
  pnp-cmd
  optimizer
  evaluator
  synthesis
  data_manager
  config
  idb
  idm
  tcl
  log
  usage
  pthread
  stdc++fs
  IdbBuilder
  def_service
  lef_service)
