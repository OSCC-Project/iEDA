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
 * @project		iDB
 * @file		IdbObs.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Obstrction information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "../../../basic/geometry/IdbLayerShape.h"
#include "../db_layout/IdbLayer.h"

namespace idb {

using std::map;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class IdbObsLayer
{
 public:
  IdbObsLayer();
  ~IdbObsLayer();

  // getter
  IdbLayerShape* get_shape() { return _layer_shape; }

  // setter
  void set_shape(IdbLayerShape* layer_shape) { _layer_shape = layer_shape; }

  // operator

 private:
  IdbLayerShape* _layer_shape;
};

class IdbObs
{
 public:
  IdbObs();
  ~IdbObs();

  // getter
  uint32_t get_obs_layer_num() { return _obs_layer_list.size(); }
  std::vector<IdbObsLayer*>& get_obs_layer_list() { return _obs_layer_list; }

  // setter
  IdbObsLayer* add_obs_layer(IdbObsLayer* obs_layer = nullptr);

  // operator

 private:
  std::vector<IdbObsLayer*> _obs_layer_list;
};

}  // namespace idb
