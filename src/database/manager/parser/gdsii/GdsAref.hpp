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

#include "GdsElement.hpp"
#include "GdsStrans.hpp"

namespace idb {

class GdsAref : public GdsElemBase
{
 public:
  GdsAref() : GdsElemBase(GdsElemType::kAref), sname(), strans(), col(1), row(1) {}

  GdsAref& operator=(const GdsAref& rhs)
  {
    GdsElemBase::operator=(rhs);
    sname = rhs.sname;
    strans = rhs.strans;
    col = rhs.col;
    row = rhs.row;

    return *this;
  }

  void reset() override
  {
    reset_base();
    sname.clear();
    strans.reset();
    col = 1;
    row = 1;
  }

  // members
  std::string sname;
  GdsStrans strans;
  int16_t col;
  int16_t row;
};

}  // namespace idb
