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
#include "MPDB.hh"

namespace ipl::imp {

MPDB::MPDB(ipl::PlacerDB* pldb)
{
  _db_wrapper = new IPLDBWrapper(pldb);
  initMPDB();
}

FPInst* MPDB::findNewMacro(FPInst* inst)
{
  FPInst* new_macro = nullptr;
  auto new_macro_iter = _inst_to_new_macro_map.find(inst);
  if (new_macro_iter != _inst_to_new_macro_map.end()) {
    new_macro = (*new_macro_iter).second;
  }
  return new_macro;
}

void MPDB::showNewNetMessage()
{
  std::vector<int> pin_num(10);
  for (FPNet* net : _new_net_list) {
    switch (net->get_degree()) {
      case 0:
        pin_num[0]++;
        break;
      case 1:
        pin_num[1]++;
        break;
      case 2:
        pin_num[2]++;
        break;
      case 3:
        pin_num[3]++;
        break;
      case 4:
        pin_num[4]++;
        break;
      case 5:
        pin_num[5]++;
        break;
      case 6:
        pin_num[6]++;
        break;
      case 7:
        pin_num[7]++;
        break;
      case 8:
        pin_num[8]++;
        break;
      default:
        pin_num[9]++;
        break;
    }
  }
  float net_num = _new_net_list.size();
  LOG_INFO << "number of pins   number of nets   percentage";
  for (size_t i = 0; i < pin_num.size(); ++i) {
    LOG_INFO << i << " " << pin_num[i] << " " << float(pin_num[i]) / net_num * 100 << "%";
  }
}

void MPDB::initMPDB()
{
  std::vector<FPInst*> macro_list = _db_wrapper->get_design()->get_macro_list();
  _true_index = macro_list.size();
  for (FPInst* macro : macro_list) {
    _total_macro_list.emplace_back(macro);
    _name_to_macro_map.emplace(macro->get_name(), macro);
  }
}

FPInst* MPDB::findMacro(std::string name)
{
  FPInst* macro = nullptr;
  auto macro_iter = _name_to_macro_map.find(name);
  if (macro_iter != _name_to_macro_map.end()) {
    macro = macro_iter->second;
  }
  return macro;
}

void MPDB::setMacroFixed(std::string name, int32_t x, int32_t y)
{
  FPInst* macro = findMacro(name);
  if (nullptr == macro) {
    LOG_INFO << "the fixed macro (" << name << ") is not found!";
    return;
  }
  macro->set_fixed(true);
  if (-1 != x) {
    macro->set_x(x);
  }
  if (-1 != y) {
    macro->set_y(y);
  }

  FPRect* core = _db_wrapper->get_layout()->get_core_shape();
  FPRect* die = _db_wrapper->get_layout()->get_die_shape();
  int32_t llx = macro->get_x();
  int32_t lly = macro->get_y();
  llx -= core->get_x() - die->get_x();
  lly -= core->get_y() - die->get_y();
  macro->set_x(llx);
  macro->set_y(lly);
}

void MPDB::add_guidance_to_macro_name(FPRect* guidance, FPInst* macro)
{
  _guidance_to_macro_map.emplace(guidance, macro);
}

void MPDB::add_guidance_to_macro_name(FPRect* guidance, std::string macro_name)
{
  FPInst* macro = findMacro(macro_name);
  if (nullptr == macro) {
    LOG_INFO << "the fixed macro (" << macro_name << ") is not found!";
    return;
  }
  _guidance_to_macro_map.emplace(guidance, macro);
}

// fixed macro added to _blockage_list, non-fixed macro added to _place_macro_list;
void MPDB::updatePlaceMacroList()
{
  _place_macro_list.clear();
  for (FPInst* macro : _total_macro_list) {
    if (macro->isFixed()) {
      FPRect* rect = new FPRect();
      rect->set_x(macro->get_x());
      rect->set_y(macro->get_x());
      rect->set_width(macro->get_width());
      rect->set_height(macro->get_height());
      _blockage_list.emplace_back(rect);
    } else {
      _place_macro_list.emplace_back(macro);
    }
  }
}

void MPDB::writeDB()
{
  // update location
  uint32_t add_x, add_y;
  FPRect* die = _db_wrapper->get_layout()->get_die_shape();
  FPRect* core = _db_wrapper->get_layout()->get_core_shape();
  add_x = core->get_x() - die->get_x();
  add_y = core->get_y() - die->get_y();
  for (FPInst* macro : _total_macro_list) {
    macro->set_x(macro->get_x() + add_x);
    macro->set_y(macro->get_y() + add_y);
  }

  // write std inst coordinate to iDB
  std::vector<FPInst*> std_cell_list = _db_wrapper->get_design()->get_std_cell_list();
  int32_t x, y;
  FPInst* new_macro;
  for (FPInst* std_cell : std_cell_list) {
    if (!std_cell->isFixed()) {
      new_macro = findNewMacro(std_cell);
      if (nullptr == new_macro) {
        continue;
      }
      x = new_macro->get_center_x();
      y = new_macro->get_center_y();
      std_cell->set_x(x);
      std_cell->set_y(y);
    }
  }
}

void MPDB::writeResult(std::string output_path)
{
  std::ofstream file;
  file.open(output_path + "/ifp_result.csv");
  file << "macro_name,x,y,orient1,orient2" << std::endl;
  LOG_INFO << "macro location: ";
  for (FPInst* macro : _db_wrapper->get_design()->get_macro_list()) {
    file << macro->get_name() << "," << macro->get_x() << "," << macro->get_y() << "," << macro->get_orient_name() << std::endl;
    LOG_INFO << macro->get_name() << ": " << macro->get_x() << " " << macro->get_y() << " " << macro->get_orient_name();
  }
  file.close();
};

void MPDB::clearNewNetList()
{
  for (auto new_net : _new_net_list) {
    if (new_net != nullptr) {
      delete new_net;
      new_net = nullptr;
    }
  }
  _new_net_list.clear();
}

}  // namespace ipl::imp