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
#include "ieco.h"

#include "ieco_via.h"

namespace ieco {

ECOManager::ECOManager()
{
  _data_manager = new EcoDataManager();
}

ECOManager::~ECOManager()
{
  if (_data_manager != nullptr) {
    delete _data_manager;
    _data_manager = nullptr;
  }
}

void ECOManager::ecoVia(std::string type)
{
  ECOVia eco_via(_data_manager);
  eco_via.init();
  eco_via.repair(type);
}

}  // namespace ieco