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
#include <string>
#include <vector>

#include "ieco_data_via.h"

namespace idb {
class IdbVia;
}

namespace ieco {

class EcoData
{
 public:
  EcoData();
  ~EcoData();
  std::map<std::string, EcoDataViaLayer>& get_via_layers() { return _via_layers; }
  EcoDataViaLayer& get_via_layer(const std::string layer);
  EcoDataViaLayer& get_via_layer(idb::IdbVia* idb_via);

 private:
  std::map<std::string, EcoDataViaLayer> _via_layers;
};

}  // namespace ieco