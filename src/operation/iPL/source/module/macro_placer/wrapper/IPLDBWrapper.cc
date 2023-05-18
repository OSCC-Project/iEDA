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
#include "IPLDBWrapper.hh"

namespace ipl::imp {

IPLDBWrapper::IPLDBWrapper(ipl::PlacerDB* ipl_db) : _iplw_database(new IPLDBWDatabase(ipl_db))
{
  wrapIPLData();
}
IPLDBWrapper::~IPLDBWrapper()
{
  delete _iplw_database;
}

void IPLDBWrapper::wrapIPLData()
{
  const ipl::Layout* ipl_layout = _iplw_database->_ipl_db->get_layout();
  ipl::Design* ipl_design = _iplw_database->_ipl_db->get_design();

  wrapLayout(ipl_layout);
  wrapDesign(ipl_design);
}

void IPLDBWrapper::wrapLayout(const ipl::Layout* ipl_layout)
{
  FPLayout* imp_layout = _iplw_database->_layout;

  // set die shape
  ipl::Rectangle<int32_t> ipl_die = ipl_layout->get_die_shape();
  imp_layout->set_die_x(ipl_die.get_ll_x());
  imp_layout->set_die_y(ipl_die.get_ll_y());
  imp_layout->set_die_width(ipl_die.get_ur_x() - ipl_die.get_ll_x());
  imp_layout->set_die_height(ipl_die.get_ur_y() - ipl_die.get_ll_y());

  // set core shape
  ipl::Rectangle<int32_t> ipl_core = ipl_layout->get_core_shape();
  imp_layout->set_core_x(ipl_core.get_ll_x());
  imp_layout->set_core_y(ipl_core.get_ll_y());
  imp_layout->set_core_width(ipl_core.get_ur_x() - ipl_core.get_ll_x());
  imp_layout->set_core_height(ipl_core.get_ur_y() - ipl_core.get_ll_y());
}

void IPLDBWrapper::wrapDesign(ipl::Design* ipl_design)
{
  FPDesign* imp_design = _iplw_database->_design;

  // set design name.
  const string design_name = ipl_design->get_design_name();
  imp_design->set_design_name(design_name);

  // set instance list
  wrapInstancelist(ipl_design);

  // set net list
  wrapNetlist(ipl_design);
}

void IPLDBWrapper::wrapInstancelist(ipl::Design* ipl_design)
{
  FPDesign* imp_design = _iplw_database->_design;

  for (ipl::Instance* ipl_inst : ipl_design->get_instance_list()) {
    // build instance && set name
    FPInst* inst_ptr = new FPInst();
    inst_ptr->set_name(ipl_inst->get_name());

    // set instance coordinate, width and height.
    ipl::Rectangle<int32_t> ipl_rect = ipl_inst->get_shape();
    inst_ptr->set_x(ipl_rect.get_ll_x());
    inst_ptr->set_y(ipl_rect.get_ll_y());
    inst_ptr->set_width(ipl_rect.get_ur_x() - ipl_rect.get_ll_x());
    inst_ptr->set_height(ipl_rect.get_ur_y() - ipl_rect.get_ll_y());

    // set orient.
    ipl::Orient ipl_orient = ipl_inst->get_orient();
    if (ipl_orient == ipl::Orient::kN_R0) {
      inst_ptr->set_orient(Orient::N);
    } else if (ipl_orient == ipl::Orient::kS_R180) {
      inst_ptr->set_orient(Orient::S);
    } else if (ipl_orient == ipl::Orient::kW_R90) {
      inst_ptr->set_orient(Orient::W);
    } else if (ipl_orient == ipl::Orient::kE_R270) {
      inst_ptr->set_orient(Orient::E);
    } else if (ipl_orient == ipl::Orient::kFN_MY) {
      inst_ptr->set_orient(Orient::FN);
    } else if (ipl_orient == ipl::Orient::kFS_MX) {
      inst_ptr->set_orient(Orient::FS);
    } else if (ipl_orient == ipl::Orient::kFW_MX90) {
      inst_ptr->set_orient(Orient::FW);
    } else if (ipl_orient == ipl::Orient::kFE_MY90) {
      inst_ptr->set_orient(Orient::FE);
    } else {
      inst_ptr->set_orient(Orient::kNone);
    }

    // set inst type.
    if (ipl_inst->get_cell_master() != nullptr) {
      if (ipl_inst->get_cell_master()->isMacro()) {
        inst_ptr->set_type(InstType::MACRO);
        imp_design->add_macro(inst_ptr);
      } else {
        inst_ptr->set_type(InstType::STD);
        imp_design->add_std_cell(inst_ptr);
      }
    }

    // set state
    if (ipl_inst->isFixed()) {
      inst_ptr->set_fixed(true);
    } else {
      inst_ptr->set_fixed(false);
    }

    _iplw_database->_fp_inst_map.emplace(ipl_inst, inst_ptr);
    _iplw_database->_ipl_inst_map.emplace(inst_ptr, ipl_inst);
  }
}

void IPLDBWrapper::wrapNetlist(ipl::Design* ipl_design)
{
  FPDesign* imp_design = _iplw_database->_design;

  for (ipl::Net* ipl_net : ipl_design->get_net_list()) {
    // build net and set name
    string net_name = ipl_net->get_name();
    FPNet* net_ptr = new FPNet();
    net_ptr->set_name(net_name);

    // set pin_list
    ipl::Pin* ipl_driving_pin = ipl_net->get_driver_pin();
    if (ipl_driving_pin) {
      FPPin* pin_ptr = wrapPin(ipl_driving_pin);
      pin_ptr->set_net(net_ptr);
      net_ptr->add_pin(pin_ptr);
    }

    for (ipl::Pin* ipl_load_pin : ipl_net->get_pins()) {
      FPPin* pin_ptr = wrapPin(ipl_load_pin);
      pin_ptr->set_net(net_ptr);
      net_ptr->add_pin(pin_ptr);
    }

    imp_design->add_net(net_ptr);
    _iplw_database->_fp_net_map.emplace(ipl_net, net_ptr);
    _iplw_database->_ipl_net_map.emplace(net_ptr, ipl_net);
  }
}

FPPin* IPLDBWrapper::wrapPin(ipl::Pin* ipl_pin)
{
  FPDesign* imp_design = _iplw_database->_design;
  ipl::Instance* ipl_inst = ipl_pin->get_instance();
  FPPin* pin_ptr = nullptr;

  if (!ipl_inst) {
    pin_ptr = new FPPin();
    pin_ptr->set_name(ipl_pin->get_name());
    pin_ptr->set_io_pin();
    pin_ptr->set_x(ipl_pin->get_center_coordi().get_x());
    pin_ptr->set_y(ipl_pin->get_center_coordi().get_y());
  } else {
    pin_ptr = new FPPin();
    pin_ptr->set_name(ipl_pin->get_name());
    FPInst* imp_inst = _iplw_database->find_imp_inst(ipl_inst);
    pin_ptr->set_instance(imp_inst);
    imp_inst->add_pin(pin_ptr);
    // pin_ptr->set_x(ipl_pin->get_offset_coordi().get_x());
    // pin_ptr->set_y(ipl_pin->get_offset_coordi().get_y());
  }

  imp_design->add_pin(pin_ptr);
  _iplw_database->_fp_pin_map.emplace(ipl_pin, pin_ptr);
  _iplw_database->_ipl_pin_map.emplace(pin_ptr, ipl_pin);

  return pin_ptr;
}

void IPLDBWrapper::writeBackSourceDataBase()
{
  for (FPInst* inst : _iplw_database->_design->get_std_cell_list()) {
    ipl::Instance* ipl_inst = nullptr;
    auto ipl_inst_iter = _iplw_database->_ipl_inst_map.find(inst);
    if (ipl_inst_iter != _iplw_database->_ipl_inst_map.end()) {
      ipl_inst = ipl_inst_iter->second;
    }
    if (ipl_inst) {
      ipl_inst->update_coordi(inst->get_x(), inst->get_y());
    }
  }

  for (FPInst* macro : _iplw_database->_design->get_macro_list()) {
    ipl::Instance* ipl_inst = nullptr;
    auto ipl_inst_iter = _iplw_database->_ipl_inst_map.find(macro);
    if (ipl_inst_iter != _iplw_database->_ipl_inst_map.end()) {
      ipl_inst = ipl_inst_iter->second;
    }
    if (ipl_inst) {
      // set coordinate
      ipl_inst->update_coordi(macro->get_x(), macro->get_y());

      // set orient
      Orient inst_orient = macro->get_orient();
      if (inst_orient == Orient::N) {
        ipl_inst->set_orient(ipl::Orient::kN_R0);
      } else if (inst_orient == Orient::S) {
        ipl_inst->set_orient(ipl::Orient::kS_R180);
      } else if (inst_orient == Orient::W) {
        ipl_inst->set_orient(ipl::Orient::kW_R90);
      } else if (inst_orient == Orient::E) {
        ipl_inst->set_orient(ipl::Orient::kE_R270);
      } else if (inst_orient == Orient::FN) {
        ipl_inst->set_orient(ipl::Orient::kFN_MY);
      } else if (inst_orient == Orient::FS) {
        ipl_inst->set_orient(ipl::Orient::kFS_MX);
      } else if (inst_orient == Orient::FW) {
        ipl_inst->set_orient(ipl::Orient::kFW_MX90);
      } else if (inst_orient == Orient::FE) {
        ipl_inst->set_orient(ipl::Orient::kFE_MY90);
      } else {
        ipl_inst->set_orient(ipl::Orient::kNone);
      }

      // set state
      ipl_inst->set_instance_state(ipl::INSTANCE_STATE::kFixed);
    }
  }
}
}  // namespace ipl::imp