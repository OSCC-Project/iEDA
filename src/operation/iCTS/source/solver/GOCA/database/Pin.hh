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
/**
 * @file Pin.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
namespace icts {
class Inst;
class Net;
}  // namespace icts
#include "Enum.hh"
#include "Inst.hh"
#include "Net.hh"
#include "Node.hh"
namespace icts {

class Pin : public Node
{
  friend class Inst;

 public:
  virtual ~Pin() override
  {
    _inst = nullptr;
    _net = nullptr;
  }
  // get
  virtual std::string get_cell_master() const override;
  const PinType& get_pin_type() const { return _pin_type; }
  Inst* get_inst() const { return _inst; }
  Net* get_net() const { return _net; }

  // set
  void set_pin_type(const PinType& pin_type) { _pin_type = pin_type; }
  void set_inst(Inst* inst) { _inst = inst; }
  void set_net(Net* net) { _net = net; }
  // bool
  virtual bool isDriver() const override { return _pin_type == PinType::kDriver; }
  virtual bool isLoad() const override { return _pin_type == PinType::kLoad; }

 private:
  Pin(const std::string& name, const Point& location) : Node(name, location) {}
  Pin(Inst* inst, const Point& location, const std::string& pin_name, const PinType& pin_type = PinType::kLoad)
      : Node(pin_name, location), _pin_type(pin_type), _inst(inst)
  {
    init();
  }
  Pin(Node&& node) : Node(std::move(node))
  {
    std::ranges::for_each(get_children(), [&](Node* child) { child->set_parent(this); });
  }

  void init();
  PinType _pin_type = PinType::kLoad;
  Inst* _inst = nullptr;
  Net* _net = nullptr;
};
}  // namespace icts