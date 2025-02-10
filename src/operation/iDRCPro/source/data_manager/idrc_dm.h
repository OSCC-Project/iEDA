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

#include <map>
#include <vector>

namespace irt {
class BaseRegion;
class BaseShape;
}  // namespace irt

namespace idb {
class IdbRegularWireSegment;
class IdbLayerShape;
}  // namespace idb

namespace idrc {

class DrcDataManager
{
 public:
  DrcDataManager();
  ~DrcDataManager();

  std::map<int, std::vector<idb::IdbRegularWireSegment*>>* get_routing_data() { return _routing_data; }
  void set_routing_data(std::map<int, std::vector<idb::IdbRegularWireSegment*>>* routing_data) { _routing_data = routing_data; }

  std::map<int, std::vector<idb::IdbLayerShape*>>* get_pin_data() { return _pin_data; }
  void set_pin_data(std::map<int, std::vector<idb::IdbLayerShape*>>* pin_data) { _pin_data = pin_data; }

  std::vector<idb::IdbLayerShape*>* get_env_shapes() { return _env_shapes; }
  void set_env_shapes(std::vector<idb::IdbLayerShape*>* env_shapes) { _env_shapes = env_shapes; }

 private:
  // data from rt
  std::map<int, std::vector<idb::IdbRegularWireSegment*>>* _routing_data;
  std::map<int, std::vector<idb::IdbLayerShape*>>* _pin_data;
  std::vector<idb::IdbLayerShape*>* _env_shapes;
};

}  // namespace idrc