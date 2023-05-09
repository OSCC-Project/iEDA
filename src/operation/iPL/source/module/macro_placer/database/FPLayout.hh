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

#include "FPRect.hh"

namespace ipl::imp {

class FPLayout
{
 public:
  FPLayout(){};
  ~FPLayout(){};

  // getter
  FPRect* get_die_shape() const { return _die_shape; }
  FPRect* get_core_shape() const { return _core_shape; }

  // setter
  void set_die_shape(FPRect* die) { _die_shape = die; }
  void set_core_shape(FPRect* core) { _core_shape = core; }

 private:
  FPRect* _die_shape;
  FPRect* _core_shape;
};

}  // namespace ipl::imp
