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

#include "EXTLayerRect.hpp"
#include "RTU.hpp"
#include "RoutingLayer.hpp"
#include "ViaMaster.hpp"

namespace irt {

class Blockage : public EXTLayerRect
{
 public:
  Blockage() = default;
  ~Blockage() = default;
  // getter
  // setter
  void set_is_artificial(const bool is_artificial) { _is_artificial = is_artificial; }
  // function
  bool isArtificial() const { return _is_artificial; }

 private:
  bool _is_artificial = false;
};

}  // namespace irt
