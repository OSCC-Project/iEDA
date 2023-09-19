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
#include "LGRow.hh"

namespace ipl {

LGSite::LGSite(std::string name) : _name(name), _width(0), _height(0)
{
}

LGSite::~LGSite()
{
}

LGRow::LGRow(std::string row_name, LGSite* site, int32_t site_num) : _index(-1), _name(row_name), _site(site), _site_num(site_num)
{
}

LGRow::~LGRow()
{
  delete _site;
  _site = nullptr;
}

}  // namespace ipl