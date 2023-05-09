/**
 * @file StaBuildExceptionTag.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief Build exception tag of multicycle path, false path, max/min delay.
 * @version 0.1
 * @date 2022-07-20
 */

#include "StaBuildExceptionTag.hh"
#include "sdc/SdcException.hh"

namespace ista {

unsigned StaBuildExceptionTag::operator()(StaVertex* the_vertex) { return 1; }

unsigned StaBuildExceptionTag::operator()(StaGraph* the_graph) {
  LOG_INFO << "build exception tag start";
  auto* sdc_exception = get_sdc_exception();
  LOG_FATAL_IF(!sdc_exception) << "sdc exception not exist.";

  auto& prop_froms = sdc_exception->get_prop_froms();
  auto& prop_tos = sdc_exception->get_prop_tos();
  auto& prop_throughs_list = sdc_exception->get_prop_throughs();

  unsigned is_ok =
      buildTagGraph(the_graph, prop_froms, prop_tos, prop_throughs_list);

  LOG_INFO << "build exception tag end";

  return is_ok;
}

}  // namespace ista