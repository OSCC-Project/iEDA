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
#include "Block.hh"

namespace imp {
Block::Block(std::string name, std::shared_ptr<Netlist> netlist, std::shared_ptr<Object> parent)
    : Object::Object(name, parent), _netlist(netlist)
{
}

size_t Block::level() const
{
  return !_parent.lock() ? 1 : 1 + std::static_pointer_cast<Block, Object>(_parent.lock())->level();
}

bool Block::is_leaf()
{
  return !_netlist || _netlist->empty();
}

Netlist& Block::netlist()
{
  return *_netlist;
}

const Netlist& Block::netlist() const
{
  return *_netlist;
}

void Block::set_netlist(std::shared_ptr<Netlist> netlist)
{
  _netlist = netlist;
}
}  // namespace imp
