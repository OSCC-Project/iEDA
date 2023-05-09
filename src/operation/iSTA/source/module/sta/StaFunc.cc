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
 * @file StaFunc.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of sta functor.
 * @version 0.1
 * @date 2021-02-17
 */

#include "StaFunc.hh"

#include "Sta.hh"
#include "StaDump.hh"
#include "time/Time.hh"

namespace ista {

StaFunc::StaFunc() {
  Sta* ista = Sta::getOrCreateSta();
  _ista = ista;
}

StaFunc::~StaFunc() = default;

unsigned StaFunc::operator()(StaGraph* /* the_graph */) {
  LOG_FATAL << "The func is not implemented";
  return 0;
}
unsigned StaFunc::operator()(StaVertex* /* the_vertex */) {
  LOG_FATAL << "The func is not implemented";
  return 0;
}

unsigned StaFunc::operator()(StaArc* /* the_arc */) {
  LOG_FATAL << "The func is not implemented";
  return 0;
}

unsigned StaFunc::operator()(StaClock* /* the_clock */) {
  LOG_FATAL << "The func is not implemented";
  return 0;
}

AnalysisMode StaFunc::get_analysis_mode() { return _ista->get_analysis_mode(); }
unsigned StaFunc::getNumThreads() { return _ista->get_num_threads(); }

/**
 * @brief Print the timing path record for debug.
 *
 */
void StaFunc::PrintTraceRecord() {
  StaDumpYaml dump_yaml;
  auto& path_stack = _trace_path_record;

  StaVertex* last_vertex = nullptr;
  while (!path_stack.empty()) {
    auto* the_vertex = path_stack.top();

    the_vertex->exec(dump_yaml);

    if (last_vertex) {
      auto src_arcs = last_vertex->getSrcArc(the_vertex);
      auto snk_arcs = last_vertex->getSnkArc(the_vertex);
      auto* the_arc = src_arcs.empty()
                          ? (snk_arcs.empty() ? nullptr : snk_arcs.front())
                          : src_arcs.front();
      if (the_arc) {
        the_arc->exec(dump_yaml);
      }
    }

    last_vertex = the_vertex;

    path_stack.pop();
  }

  std::string now_time = Time::getNowWallTime();
  std::string tmp = Str::replace(now_time, ":", "_");
  const char* text_file_name = Str::printf("path_%s.txt", tmp.c_str());

  dump_yaml.printText(text_file_name);
}

}  // namespace ista
