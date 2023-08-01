set(IDRC_HOME ${HOME_OPERATION}/iDRC)

include_directories(
  ##iDRC
  ${IDRC_HOME}
  ${IDRC_HOME}/src
  ${IDRC_HOME}/src/config
  ${IDRC_HOME}/src/util
  ${IDRC_HOME}/src/database
  ${IDRC_HOME}/src/module/RegionQuery
  ${IDRC_HOME}/src/module/EnclosedAreaCheck
  ${IDRC_HOME}/src/module/RoutingAreaCheck
  ${IDRC_HOME}/src/module/RoutingWidthCheck
  ${IDRC_HOME}/src/module/RoutingSpacingCheck
  ${IDRC_HOME}/src/module/SpotParser 
  ${IDRC_HOME}/src/module/MultiPatterning/ConnectedComponentFinder
  ${IDRC_HOME}/src/module/MultiPatterning/OddCycleFinder
  ${IDRC_HOME}/src/module/MultiPatterning/ColorableChecker
  ${IDRC_HOME}/src/module/MultiPatterning
)

link_libraries(DRC)