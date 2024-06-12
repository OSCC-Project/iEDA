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
/**
 * @file		route_builder.h
 * @date		13/05/2024
 * @version		0.1
 * @description


        build feature data
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "feature_irt.h"

namespace idb {
class IdbPin;
class IdbVia;
}  // namespace idb
using namespace idb;

namespace ieda_feature {

class RouteDataBuilder
{
 public:
  RouteDataBuilder(RouteAnalyseData* data = nullptr) { _data = data == nullptr ? new RouteAnalyseData() : data ; }
  ~RouteDataBuilder() = default;

  // builder
  bool buildRouteData();

 private:
  RouteAnalyseData* _data = nullptr;

  bool is_pa(IdbPin* pin, IdbVia* via);

  void add_term_pa(TermPA& term_pa, DbPinAccess pin_access);
  TermPA& find_cell_term_pa(std::string cell_master_name, std::string term_name);
};

}  // namespace ieda_feature
