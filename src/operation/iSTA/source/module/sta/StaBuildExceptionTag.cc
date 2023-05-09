// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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