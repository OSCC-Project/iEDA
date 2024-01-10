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
#include "Netlist.hh"
#include "Layout.hh"
#include "Net.hh"
#include "Object.hh"
#include "Pin.hh"
namespace imp {
std::shared_ptr<Layout> get_layout(Netlist& netlist)
{
  return netlist.property();
}
size_t add_object(Netlist& netlist, std::shared_ptr<Object> obj)
{
  return netlist.add_vertex(obj);
}
size_t add_net(Netlist& netlist, const std::vector<size_t>& inst_pos, const std::vector<std::shared_ptr<Pin>>& pins,
               std::shared_ptr<Net> net)
{
  return netlist.add_hyper_edge(inst_pos, pins, net);
}

}  // namespace imp
