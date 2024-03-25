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

#include "Config.hpp"
#include "DataManager.hpp"
#include "Database.hpp"

namespace irt {

#define LA_INST (irt::LayerAssigner::getInst())

class LayerAssigner
{
 public:
  static void initInst();
  static LayerAssigner& getInst();
  static void destroyInst();
  // function
  void assign();

 private:
  // self
  static LayerAssigner* _la_instance;

  LayerAssigner() = default;
  LayerAssigner(const LayerAssigner& other) = delete;
  LayerAssigner(LayerAssigner&& other) = delete;
  ~LayerAssigner() = default;
  LayerAssigner& operator=(const LayerAssigner& other) = delete;
  LayerAssigner& operator=(LayerAssigner&& other) = delete;
  // function
};

}  // namespace irt
