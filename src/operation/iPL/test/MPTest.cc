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
#include <string>

#include "../source/module/macro_placer/MacroPlacer.hh"
#include "PlacerDB.hh"
#include "gtest/gtest.h"
#include "iPL_API.hh"
#include "idm.h"

namespace ipl {
class MPTest : public testing::Test
{
};

TEST_F(MPTest, testMoudle)
{
  imp::FPInst* inst_1 = new imp::FPInst();
  imp::FPInst* inst_2 = new imp::FPInst();
  imp::FPInst* inst_3 = new imp::FPInst();
  imp::FPInst* inst_4 = new imp::FPInst();
  imp::FPInst* inst_5 = new imp::FPInst();
  inst_2->set_type(imp::InstType::STD);
  inst_1->set_type(imp::InstType::STD);
  inst_3->set_type(imp::InstType::STD);
  inst_4->set_type(imp::InstType::MACRO);
  inst_5->set_type(imp::InstType::MACRO);
  inst_1->set_name("us/km/uu/__01");
  inst_2->set_name("us/__01");
  inst_3->set_name("us/km/uu/__02");
  inst_4->set_name("us/ku/__01");
  inst_5->set_name("us/ku/uu/__01");
  std::vector<imp::FPInst*> stdcell_list;
  std::vector<imp::FPInst*> macro_list;
  stdcell_list.emplace_back(inst_1);
  stdcell_list.emplace_back(inst_2);
  stdcell_list.emplace_back(inst_3);
  macro_list.emplace_back(inst_4);
  macro_list.emplace_back(inst_5);

  imp::Module* module = new imp::Module();

  for (imp::FPInst* stdcell : stdcell_list) {
    module->add_inst(stdcell);
  }
  for (imp::FPInst* macro : macro_list) {
    module->add_inst(macro);
  }
  EXPECT_EQ(module->get_name(), "top");
  EXPECT_EQ(module->get_layer(), 0);
  imp::Module* child_1_module = module->findChildMoudle("top/us");
  EXPECT_EQ(child_1_module->get_name(), "top/us");
  std::vector<imp::Module*> child_2_module_list = child_1_module->get_child_module_list();
  imp::Module* child_2_module_1 = child_2_module_list[0];
  imp::Module* child_2_module_2 = child_2_module_list[1];
  EXPECT_EQ(child_2_module_1->get_name(), "top/us/km");
  EXPECT_EQ(child_1_module->get_stdcell_list()[0], inst_2);
  EXPECT_EQ(child_2_module_1->get_child_module_list()[0]->get_stdcell_list()[1], inst_3);
  EXPECT_EQ(child_2_module_2->get_macro_list()[0], inst_4);
}

TEST_F(MPTest, testBuildNewNet) {
  imp::FPInst* inst_1 = new imp::FPInst();
  imp::FPInst* inst_2 = new imp::FPInst();
  imp::FPInst* inst_3 = new imp::FPInst();
  imp::FPInst* inst_4 = new imp::FPInst();
  imp::FPInst* inst_5 = new imp::FPInst();
  imp::FPInst* inst_6 = new imp::FPInst();
  inst_1->set_name("inst_1");
  inst_2->set_name("inst_2");
  inst_3->set_name("inst_3");
  inst_4->set_name("inst_4");
  inst_5->set_name("inst_5");
  inst_6->set_name("inst_6");
  inst_2->set_type(imp::InstType::STD);
  inst_1->set_type(imp::InstType::STD);
  inst_3->set_type(imp::InstType::STD);
  inst_4->set_type(imp::InstType::MACRO);
  inst_5->set_type(imp::InstType::MACRO);
  inst_6->set_type(imp::InstType::NEWMACRO);
  imp::FPPin* pin_1_1 = new imp::FPPin();
  imp::FPPin* pin_1_2 = new imp::FPPin();
  imp::FPPin* pin_2_1 = new imp::FPPin();
  imp::FPPin* pin_2_2 = new imp::FPPin();
  imp::FPPin* pin_3_1 = new imp::FPPin();
  imp::FPPin* pin_4_1 = new imp::FPPin();
  imp::FPPin* pin_5_1 = new imp::FPPin();
  imp::FPPin* pin_io = new imp::FPPin();
  pin_io->set_io_pin();
  pin_1_1->set_name("pin_1_1");
  pin_1_2->set_name("pin_1_2");
  pin_2_1->set_name("pin_2_1");
  pin_2_2->set_name("pin_2_2");
  pin_3_1->set_name("pin_3_1");
  pin_4_1->set_name("pin_4_1");
  pin_5_1->set_name("pin_5_1");
  pin_io->set_name("pin_io");
  pin_1_1->set_instance(inst_1);
  pin_1_2->set_instance(inst_1);
  pin_2_1->set_instance(inst_2); 
  pin_2_2->set_instance(inst_2); 
  pin_3_1->set_instance(inst_3); 
  pin_4_1->set_instance(inst_4);
  pin_5_1->set_instance(inst_5);
  inst_1->add_pin(pin_1_1);
  inst_1->add_pin(pin_1_2);
  inst_2->add_pin(pin_2_1);
  inst_2->add_pin(pin_2_2);
  inst_3->add_pin(pin_3_1);
  inst_4->add_pin(pin_4_1);
  inst_5->add_pin(pin_5_1);
  imp::FPNet* net_1 = new imp::FPNet();
  imp::FPNet* net_2 = new imp::FPNet();
  imp::FPNet* net_3 = new imp::FPNet();
  net_1->set_name("net_1");
  net_2->set_name("net_2");
  net_3->set_name("net_3");
  pin_1_1->set_net(net_1);
  pin_4_1->set_net(net_1);
  pin_1_2->set_net(net_2);
  pin_2_1->set_net(net_2);
  pin_5_1->set_net(net_2);
  pin_io->set_net(net_2);
  pin_2_2->set_net(net_3);
  pin_3_1->set_net(net_3);
  net_1->add_pin(pin_1_1);
  net_1->add_pin(pin_4_1);
  net_2->add_pin(pin_1_2);
  net_2->add_pin(pin_2_1);
  net_2->add_pin(pin_5_1);
  net_2->add_pin(pin_io);
  net_3->add_pin(pin_2_2);
  net_3->add_pin(pin_3_1);
  imp::FPDesign* design = new imp::FPDesign();
  imp::FPLayout* layout = new imp::FPLayout();
  design->add_std_cell(inst_1);
  design->add_std_cell(inst_2);
  design->add_std_cell(inst_3);
  design->add_macro(inst_4);
  design->add_macro(inst_5);
  design->add_net(net_1);
  design->add_net(net_2);
  design->add_net(net_3); 
  vector<imp::FPNet*> old_net_list;
  old_net_list.emplace_back(net_1);
  old_net_list.emplace_back(net_2);
  old_net_list.emplace_back(net_3);
  vector<imp::FPNet*> new_net_list;
  map<imp::FPInst*, imp::FPInst*> _inst_to_new_macro_map;
  _inst_to_new_macro_map.emplace(inst_1, inst_6);
  _inst_to_new_macro_map.emplace(inst_2, inst_6);
  _inst_to_new_macro_map.emplace(inst_3, inst_6);

  // test code
  int count = -1;
  for (imp::FPNet* net : old_net_list) {
    ++count;
    vector<imp::FPPin*> pin_list = net->get_pin_list();
    if (pin_list.size() > 20 || pin_list.size() == 0) {
      continue;
    }
    imp::FPNet* new_net = new imp::FPNet();
    set<imp::FPInst*> net_macro_set;

    // set name
    new_net->set_name(net->get_name());

    // read instance pin
    for (imp::FPPin* old_pin : pin_list) {
      if (old_pin->is_io_pin()) {
        new_net->add_pin(old_pin);
        continue;
      }
      imp::FPInst* old_inst = old_pin->get_instance();
      if (nullptr == old_inst) {
        continue;
      }
      if (old_inst->isMacro()) {
        net_macro_set.insert(old_inst);
      }
      else {
        imp::FPInst* new_macro = nullptr;
        auto new_macro_iter = _inst_to_new_macro_map.find(old_inst);
        if (new_macro_iter != _inst_to_new_macro_map.end()) {
          new_macro = (*new_macro_iter).second;
        }
        if (nullptr == new_macro) {
          continue;
        }
        net_macro_set.insert(new_macro);
      }
    }
    if (net_macro_set.size() < 2) {
      continue;
    }
    for (set<imp::FPInst*>::iterator it = net_macro_set.begin(); it!=net_macro_set.end();++it) {
      imp::FPPin* new_pin = new imp::FPPin();
      new_pin->set_instance(*it);
      (*it)->add_pin(new_pin);
      new_pin->set_x(0);
      new_pin->set_y(0);
      new_pin->set_net(new_net);
      new_net->add_pin(new_pin);
    }
    new_net_list.emplace_back(new_net);
  }
  // end test code

  std::cout << new_net_list[0]->get_pin_list()[0]->get_instance()->get_name() << std::endl;
  std::cout << new_net_list[0]->get_pin_list()[1]->get_instance()->get_name() << std::endl;

  std::cout << new_net_list[1]->get_pin_list()[0]->get_name() << std::endl;
  std::cout << new_net_list[1]->get_pin_list()[1]->get_instance()->get_name() << std::endl;
  std::cout << new_net_list[1]->get_pin_list()[2]->get_instance()->get_name() << std::endl;

  // EXPECT_EQ(new_net_list[0]->get_pin_list()[0]->get_instance()->get_name(), inst_4->get_name());
  // EXPECT_EQ(new_net_list[0]->get_pin_list()[1]->get_instance()->get_name(), inst_6->get_name());
  // EXPECT_EQ(new_net_list[1]->get_pin_list()[0]->get_instance()->get_name(), inst_6->get_name());
  // EXPECT_EQ(new_net_list[1]->get_pin_list()[1]->get_instance()->get_name(), inst_5->get_name());
  // EXPECT_EQ(new_net_list[1]->get_pin_list()[2]->get_name(), pin_io->get_name());
}
}  // namespace ipl