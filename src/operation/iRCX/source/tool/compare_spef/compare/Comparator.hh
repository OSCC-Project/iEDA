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
#pragma once

#include <string>

#include "compare/CouplingCapComparator.hh"
#include "compare/NetSelector.hh"
#include "compare/PathPairGenerator.hh"
#include "compare/ResistanceSolver.hh"
#include "compare/ResultSorter.hh"
#include "config/CompareSpefConfig.hh"
#include "data/CompareSpefData.hh"

namespace ircx {
namespace compare_spef {

class Comparator
{
 public:
  explicit Comparator(const Config& config);

  auto compare(const Data& test, const Data& reference) const -> Result;

 private:
  void initializeSummary(const Data& test, const Data& reference, Result& result) const;
  void compareMatchedNets(const Data& test, const Data& reference, Result& result) const;
  void collectTestOnlyNets(const Data& test, const Data& reference, Result& result) const;
  void finishSummary(const Data& test, const Data& reference, Result& result) const;

  const Config& _config;
  NetSelector _net_selector;
  CouplingCapComparator _coupling_cap_comparator;
  PathPairGenerator _path_pair_generator;
  ResistanceSolver _resistance_solver;
  ResultSorter _result_sorter;
  bool _compare_capacitance = false;
  bool _compare_resistance = false;
};

}  // namespace compare_spef
}  // namespace ircx
