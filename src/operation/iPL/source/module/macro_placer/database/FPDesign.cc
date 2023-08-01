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

#include "FPDesign.hh"

namespace ipl::imp {

FPDesign::~FPDesign()
{
  for (FPInst* std_cell : _std_cell_list) {
    if (std_cell != nullptr) {
      delete std_cell;
      std_cell = nullptr;
    }
  }
  _std_cell_list.clear();

  for (FPInst* macro : _macro_list) {
    if (macro != nullptr) {
      delete macro;
      macro = nullptr;
    }
  }
  _macro_list.clear();

  for (FPNet* net : _net_list) {
    if (net != nullptr) {
      delete net;
      net = nullptr;
    }
  }
  _net_list.clear();

  for (FPPin* pin : _pin_list) {
    if (pin != nullptr) {
      delete pin;
      pin = nullptr;
    }
  }
  _pin_list.clear();
}

}  // namespace ipl::imp