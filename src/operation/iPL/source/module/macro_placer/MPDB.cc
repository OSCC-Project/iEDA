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

#include <fstream>
#include <set>

using std::set;
using std::endl;

namespace ipl::imp {

FPInst* MPDB::findNewMacro(FPInst* inst)
{
  FPInst* new_macro = nullptr;
  auto new_macro_iter = _inst_to_new_macro_map.find(inst);
  if (new_macro_iter != _inst_to_new_macro_map.end()) {
    new_macro = (*new_macro_iter).second;
  }
  return new_macro;
}

void MPDB::buildNetList()
{
  _new_net_list.clear();
  vector<FPNet*> old_net_list = _db_wrapper->get_design()->get_net_list();
  for (FPNet* old_net : old_net_list) {
    vector<FPPin*> pin_list = old_net->get_pin_list();
    if (pin_list.size() == 0) {
      continue;
    }

    set<FPInst*> net_macro_set;

    // create new net
    FPNet* new_net = new FPNet();
    new_net->set_name(old_net->get_name());
    // read instance pin
    for (FPPin* old_pin : pin_list) {
      if (old_pin->is_io_pin()) {
        new_net->add_pin(old_pin);
        continue;
      }
      FPInst* old_inst = old_pin->get_instance();
      if (nullptr == old_inst) {
        continue;
      }
      if (old_inst->isMacro()) {
        net_macro_set.insert(old_inst);
      } else {
        FPInst* new_macro = findNewMacro(old_inst);
        if (nullptr == new_macro) {
          continue;
        }
        net_macro_set.insert(new_macro);
      }
    }

    // create new pin
    if (net_macro_set.size() < 1 || ((net_macro_set.size() == 1) && (new_net->get_pin_list().size() == 0))) {
      delete new_net;
      continue;
    }
    for (set<FPInst*>::iterator it = net_macro_set.begin(); it != net_macro_set.end(); ++it) {
      FPPin* new_pin = new FPPin();
      new_pin->set_instance(*it);
      (*it)->add_pin(new_pin);
      new_pin->set_x(0);
      new_pin->set_y(0);
      new_pin->set_net(new_net);
      new_net->add_pin(new_pin);
    }
    _new_net_list.emplace_back(new_net);
  }
  showNetMessage();
}

void MPDB::showNetMessage()
{
  vector<int> pin_num(10);
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

void MPDB::writeGDS(string file_name)
{
  std::ofstream gds_file;
  gds_file.open(file_name);
  gds_file << "HEADER 600" << std::endl;
  gds_file << "BGNLIB" << std::endl;
  gds_file << "LIBNAME DensityLib" << std::endl;
  gds_file << "UNITS 0.001 1e-9" << std::endl;
  gds_file << "BGNSTR" << std::endl;
  gds_file << "STRNAME Die" << std::endl;

  // write die
  FPRect* die = _db_wrapper->get_layout()->get_die_shape();
  int dx = die->get_width();
  int dy = die->get_height();
  int x = die->get_x();
  int y = die->get_y();
  gds_file << "BOUNDARY" << std::endl;
  gds_file << "LAYER 0" << std::endl;
  gds_file << "DATATYPE 0" << std::endl;
  gds_file << "XY" << std::endl;
  gds_file << x << " : " << y << std::endl;
  gds_file << x + dx << " : " << y << std::endl;
  gds_file << x + dx << " : " << y + dy << std::endl;
  gds_file << x << " : " << y + dy << std::endl;
  gds_file << x << " : " << y << std::endl;
  gds_file << "ENDEL" << std::endl;

  // write core
  FPRect* core = _db_wrapper->get_layout()->get_core_shape();
  dx = core->get_width();
  dy = core->get_height();
  x = core->get_x();
  y = core->get_y();
  gds_file << "BOUNDARY" << std::endl;
  gds_file << "LAYER 1" << std::endl;
  gds_file << "DATATYPE 0" << std::endl;
  gds_file << "XY" << std::endl;
  gds_file << x << " : " << y << std::endl;
  gds_file << x + dx << " : " << y << std::endl;
  gds_file << x + dx << " : " << y + dy << std::endl;
  gds_file << x << " : " << y + dy << std::endl;
  gds_file << x << " : " << y << std::endl;
  gds_file << "ENDEL" << std::endl;

  // write blockage
  // for (FPRect* blockage : _blockage_list) {
  //   writeBlockage(gds_file, blockage, 2);
  // }

  // write new macro
  int size = _total_macro_list.size();
  for (int i = _true_index; i < size; ++i) {
    writeMacro(gds_file, _total_macro_list[i], 4);
  }

  // // write ture macro
  // for (FPInst* macro : get_design()->get_macro_list()) {
  //   writeMacro(gds_file, macro, 5);
  // }

  for (int i = 0; i < _true_index; ++i) {
    // halo
    writeMacro(gds_file, _total_macro_list[i], 5);
    _total_macro_list[i]->deleteHalo();
    // macro
    writeMacro(gds_file, _total_macro_list[i], 6);
  }

  // write net
  // writeNet(gds_file, _new_net_list);

  gds_file << "ENDSTR" << std::endl;
  gds_file << "ENDLIB" << std::endl;
  gds_file.close();
}

void MPDB::writePartitonGDS(string file_name, map<FPInst*, int> partition_result) {
  std::ofstream gds_file;
  gds_file.open(file_name);
  gds_file << "HEADER 600" << std::endl;
  gds_file << "BGNLIB" << std::endl;
  gds_file << "LIBNAME DensityLib" << std::endl;
  gds_file << "UNITS 0.001 1e-9" << std::endl;
  gds_file << "BGNSTR" << std::endl;
  gds_file << "STRNAME Die" << std::endl;

  // write die
  FPRect* die = _db_wrapper->get_layout()->get_die_shape();
  int dx = die->get_width();
  int dy = die->get_height();
  int x = die->get_x();
  int y = die->get_y();
  gds_file << "BOUNDARY" << std::endl;
  gds_file << "LAYER 0" << std::endl;
  gds_file << "DATATYPE 0" << std::endl;
  gds_file << "XY" << std::endl;
  gds_file << x << " : " << y << std::endl;
  gds_file << x + dx << " : " << y << std::endl;
  gds_file << x + dx << " : " << y + dy << std::endl;
  gds_file << x << " : " << y + dy << std::endl;
  gds_file << x << " : " << y << std::endl;
  gds_file << "ENDEL" << std::endl;

  // write core
  FPRect* core = _db_wrapper->get_layout()->get_core_shape();
  dx = core->get_width();
  dy = core->get_height();
  x = core->get_x();
  y = core->get_y();
  gds_file << "BOUNDARY" << std::endl;
  gds_file << "LAYER 1" << std::endl;
  gds_file << "DATATYPE 0" << std::endl;
  gds_file << "XY" << std::endl;
  gds_file << x << " : " << y << std::endl;
  gds_file << x + dx << " : " << y << std::endl;
  gds_file << x + dx << " : " << y + dy << std::endl;
  gds_file << x << " : " << y + dy << std::endl;
  gds_file << x << " : " << y << std::endl;
  gds_file << "ENDEL" << std::endl;

  for (auto iter = partition_result.begin(); iter != partition_result.end(); ++iter) {
    writeMacro(gds_file, iter->first, iter->second + 5);
  }

  gds_file << "ENDSTR" << std::endl;
  gds_file << "ENDLIB" << std::endl;
  gds_file.close();
}

void MPDB::initMPDB()
{
  vector<FPInst*> macro_list = _db_wrapper->get_design()->get_macro_list();
  _true_index = macro_list.size();
  for (FPInst* macro : macro_list) {
    _total_macro_list.emplace_back(macro);
    _name_to_macro_map.emplace(macro->get_name(), macro);
  }
}

FPInst* MPDB::findMacro(string name)
{
  FPInst* macro = nullptr;
  auto macro_iter = _name_to_macro_map.find(name);
  if (macro_iter != _name_to_macro_map.end()) {
    macro = macro_iter->second;
  }
  return macro;
}

void MPDB::setMacroFixed(string name, int32_t x, int32_t y)
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

void MPDB::add_guidance_to_macro_name(FPRect* guidance, string macro_name)
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

void MPDB::writeMacro(ofstream& gds_file, FPInst* macro, int layer)
{
  int llx = int(macro->get_x());
  int lly = int(macro->get_y());
  int w = int(macro->get_width());
  int h = int(macro->get_height());
  gds_file << "TEXT" << endl;
  gds_file << "LAYER 1000" << endl;
  gds_file << "TEXTTYPE 0" << endl;
  gds_file << "XY" << endl;
  gds_file << macro->get_center_x() << " : " << macro->get_center_y() << endl;
  gds_file << "STRING " << macro->get_name() << endl;
  gds_file << "ENDEL" << endl;
  gds_file << "BOUNDARY" << std::endl;
  gds_file << "LAYER " << layer << std::endl;
  gds_file << "DATATYPE 0" << std::endl;
  gds_file << "XY" << std::endl;
  gds_file << llx << " : " << lly << std::endl;
  gds_file << llx + w << " : " << lly << std::endl;
  gds_file << llx + w << " : " << lly + h << std::endl;
  gds_file << llx << " : " << lly + h << std::endl;
  gds_file << llx << " : " << lly << std::endl;
  gds_file << "ENDEL" << std::endl;
}

void MPDB::writeBlockage(ofstream& gds_file, FPRect* blockage, int layer)
{
  int llx = int(blockage->get_x());
  int lly = int(blockage->get_y());
  int w = int(blockage->get_width());
  int h = int(blockage->get_height());
  gds_file << "BOUNDARY" << std::endl;
  gds_file << "LAYER " << layer << std::endl;
  gds_file << "DATATYPE 0" << std::endl;
  gds_file << "XY" << std::endl;
  gds_file << llx << " : " << lly << std::endl;
  gds_file << llx + w << " : " << lly << std::endl;
  gds_file << llx + w << " : " << lly + h << std::endl;
  gds_file << llx << " : " << lly + h << std::endl;
  gds_file << llx << " : " << lly << std::endl;
  gds_file << "ENDEL" << std::endl;
}

void MPDB::writeNet(ofstream& gds_file, vector<FPNet*> net_list)
{
  int layer = 5;
  for (FPNet* net : net_list) {
    vector<FPPin*> pin_list = net->get_pin_list();
    FPPin* pin0 = pin_list[0];
    for (size_t i = 1; i < pin_list.size(); ++i) {
      writeLine(gds_file, pin0, pin_list[i], layer);
    }
    ++layer;
  }
}

void MPDB::writeLine(ofstream& gds_file, FPPin* start, FPPin* end, int layer)
{
  gds_file << "PATH" << endl;
  gds_file << "LAYER " << layer << endl;
  gds_file << "DATATYPE 0" << endl;
  gds_file << "WIDTH " << 20 << endl;
  gds_file << "XY" << endl;
  gds_file << int(start->get_x()) << ":" << int(start->get_y()) << endl;
  gds_file << int(end->get_x()) << ":" << int(end->get_y()) << endl;
  gds_file << "ENDEL" << endl;
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
  vector<FPInst*> std_cell_list = _db_wrapper->get_design()->get_std_cell_list();
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
}  // namespace ipl::imp