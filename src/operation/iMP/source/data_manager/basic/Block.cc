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

bool Block::is_leaf() const
{
  return !_netlist || _netlist->empty();
}

bool Block::has_netlist() const
{
  return _netlist != nullptr;
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

static void preorder_get_instances(Block& blk, std::set<std::shared_ptr<imp::Instance>>& instances, CELL_TYPE cell_type = CELL_TYPE::kNone)
{
  if (!blk.has_netlist()) {
    return;
  }
  for (auto&& i : blk.netlist().vRange()) {
    auto sub_obj = i.property();
    if (sub_obj->isInstance()) {
      auto sub_inst = std::static_pointer_cast<Instance, Object>(sub_obj);
      if (cell_type == CELL_TYPE::kNone || cell_type == sub_inst->get_cell_master().get_cell_type()) {
        instances.insert(sub_inst);
      }
    } else {
      auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
      preorder_get_instances(*sub_block, instances);
    }
  }
}

std::set<std::shared_ptr<imp::Instance>> Block::get_instances()
{
  std::set<std::shared_ptr<imp::Instance>> instances;
  preorder_get_instances(*this, instances, CELL_TYPE::kNone);
  return instances;
}

std::set<std::shared_ptr<imp::Instance>> Block::get_macros()
{
  std::set<std::shared_ptr<imp::Instance>> macros;
  preorder_get_instances(*this, macros, CELL_TYPE::kMacro);
  return macros;
}

std::set<std::shared_ptr<imp::Instance>> Block::get_io_instances()
{
  std::set<std::shared_ptr<imp::Instance>> macros;
  preorder_get_instances(*this, macros, CELL_TYPE::kIOCell);
  return macros;
}

}  // namespace imp
