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
 * @project		iDB
 * @file		builder.cpp
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
* @description


        This is common interface to build def and lef database.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "builder.h"

#include "log/Log.hh"

using std::cout;
using std::endl;

namespace idb {

IdbBuilder::IdbBuilder()
{
}

IdbBuilder::~IdbBuilder()
{
}

void IdbBuilder::log()
{
  /// lef
  IdbLayout* layout = _lef_service->get_layout();
  logModule("*************************Layout*******************************");
  logModule("Die : ");
  logModule("Core : ");
  logModule("Layers : ", layout->get_layers()->get_layers_num());
  logModule("Sites : ", layout->get_sites()->get_sites_num());
  logModule("Rows : ", layout->get_rows()->get_row_num());
  logModule("GCells : ", layout->get_gcell_grid_list()->get_gcell_grid_num());
  logModule("Track Grids : ", layout->get_track_grid_list()->get_track_grid_num());
  logModule("Cell Masters : ", layout->get_cell_master_list()->get_cell_master_num());
  logModule("Vias : ", layout->get_via_list()->get_num_via());
  logModule("Via Rules : ", layout->get_via_rule_list()->get_num_via_rule_generate());

  ////def
  logModule("*************************Design*******************************");
  IdbDesign* design = _def_service->get_design();
  logModule("Design : " + design->get_design_name());
  logModule("Version : " + design->get_version());
  logModule("Instance : ", design->get_instance_list()->get_num());
  logModule("IO Pins : ", design->get_io_pin_list()->get_pin_num());
  logModule("Vias : ", design->get_via_list()->get_num_via());
  logModule("Nets : ", design->get_net_list()->get_num());
  int32_t segment_number = 0;
  for (IdbNet* net : design->get_net_list()->get_net_list()) {
    for (IdbRegularWire* wire : net->get_wire_list()->get_wire_list()) {
      segment_number += wire->get_num();
    }
  }
  logNumber("Nets segment : ", segment_number);
  logModule("Special Nets : ", design->get_special_net_list()->get_num());
  segment_number = 0;
  for (IdbSpecialNet* net : design->get_special_net_list()->get_net_list()) {
    for (IdbSpecialWire* wire : net->get_wire_list()->get_wire_list()) {
      segment_number += wire->get_num();
    }
  }
  logNumber("Special Nets segment : ", segment_number);
  logModule("Blockages : ", design->get_blockage_list()->get_num());
  logModule("Regions : ", design->get_region_list()->get_num());
  logModule("Slots : ", design->get_slot_list()->get_num());
  logModule("Groups : ", design->get_group_list()->get_num());
  logModule("Fillers : ", design->get_fill_list()->get_num_fill());

  logSeperate();
}

IdbDefService* IdbBuilder::buildDef(string file)
{
  if (_def_service != nullptr) {
    delete _def_service;
    _def_service = nullptr;
  }

  IdbLayout* layout = _lef_service->get_layout();
  _def_service = new IdbDefService(layout);

  if (IdbDefServiceResult::kServiceFailed == _def_service->DefFileInit(file.c_str())) {
    std::cout << "Read DEF file failed..." << endl;
    return nullptr;
  }

  std::cout << "Read DEF file : " << file << endl;

  std::shared_ptr<DefRead> def_read = std::make_shared<DefRead>(_def_service);
  if (const auto ret = def_read->createDb(file.c_str()); !ret) {
    LOG_FATAL << "Def file read failed..." << endl;
  }

  buildNet();
  buildBus();
  log();

  return _def_service;
}

IdbDefService* IdbBuilder::buildDefGzip(string gzip_file)
{
  if (_def_service != nullptr) {
    delete _def_service;
    _def_service = nullptr;
  }

  IdbLayout* layout = _lef_service->get_layout();
  _def_service = new IdbDefService(layout);

  if (IdbDefServiceResult::kServiceFailed == _def_service->DefFileInit(gzip_file.c_str())) {
    std::cout << "Read DEF ZIP file failed..." << endl;
    return nullptr;
  }

  std::cout << "Read DEF ZIP file : " << gzip_file << endl;

  std::shared_ptr<DefRead> def_read = std::make_shared<DefRead>(_def_service);
  if (const auto ret = def_read->createDbGzip(gzip_file.c_str()); !ret) {
    LOG_FATAL << "Def file read failed..." << endl;
  }

  buildNet();
  buildBus();
  log();

  return _def_service;
}

IdbLefService* IdbBuilder::buildLef(vector<string>& files, bool b_techfile)
{
  if (_lef_service == nullptr) {
    _lef_service = new IdbLefService();
  } else {
    // delete service before read tech file
    if (b_techfile) {
      delete _lef_service;
      _lef_service = nullptr;
      _lef_service = new IdbLefService();
    }
  }

  _lef_service->LefFileInit(files);

  vector<string>::iterator it = files.begin();
  for (; it != files.end(); ++it) {
    string file = *it;
    std::cout << "Read LEF file : " << file << endl;
    std::shared_ptr<LefRead> lef_read = std::make_shared<LefRead>(_lef_service);
    lef_read->createDb(file.c_str());
  }
  if (!b_techfile) {
    updateLefData();
  }
  // tech add
  //   std::unique_ptr<LefRead> lef_read =
  //   std::make_unique<LefRead>(_lef_service); lef_read->createTechDb();

  return _lef_service;
}

IdbDefService* IdbBuilder::rustBuildVerilog(string file, std::string top_module_name)
{
  if (_def_service != nullptr) {
    delete _def_service;
    _def_service = nullptr;
  }

  IdbLayout* layout = _lef_service->get_layout();
  _def_service = new IdbDefService(layout);

  if (IdbDefServiceResult::kServiceFailed == _def_service->VerilogFileInit(file.c_str())) {
    std::cout << "Read Verilog file failed..." << endl;
    return nullptr;
  } else {
    std::cout << "Read Verilog file success : " << file << endl;
  }

  std::shared_ptr<RustVerilogRead> rust_verilog_read = std::make_shared<RustVerilogRead>(_def_service);
  if (top_module_name.empty())
    rust_verilog_read->createDbAutoTop(file);
  else
    rust_verilog_read->createDb(file.c_str(), top_module_name);

  checkNetPins();

  return _def_service;
}

IdbDefService* IdbBuilder::buildDefFloorplan(string file)
{
  if (_def_service != nullptr) {
    delete _def_service;
    _def_service = nullptr;
  }

  IdbLayout* layout = _lef_service->get_layout();
  _def_service = new IdbDefService(layout);

  if (IdbDefServiceResult::kServiceFailed == _def_service->DefFileInit(file.c_str())) {
    std::cout << "Read DEF file failed..." << endl;
    return nullptr;
  } else {
    std::cout << "Read DEF file : " << file << endl;
  }

  std::shared_ptr<DefRead> def_read = std::make_shared<DefRead>(_def_service);
  def_read->createFloorplanDb(file.c_str());
  //   def_read->createDb(file.c_str());

  return _def_service;
}

//   IdbDataService* IdbBuilder::buildData() {
//     if (_data_service == nullptr) {
//       _data_service = new IdbDataService();
//     }

//     return _data_service;
//   }

//   IdbDataService* IdbBuilder::buildData(IdbDefService* def_service) {
//     if (_data_service == nullptr) {
//       _data_service = std::make_shared<IdbDataService>(def_service);
//     }

//     if (IdbDataServiceResult::kServiceFailed == _data_service->DefServiceInit(def_service)) {
//       std::cout << "Get def_service failed..." << endl;
//       return nullptr;
//     }

//     return _data_service;
//   }

bool IdbBuilder::saveDef(string file, DefWriteType type)
{
  if (IdbDefServiceResult::kServiceFailed == _def_service->DefFileWriteInit(file.c_str())) {
    std::cout << "Create DEF file failed..." << endl;
    return false;
  }

  std::shared_ptr<DefWrite> def_write = std::make_shared<DefWrite>(_def_service, type);
  return def_write->writeDb(file.c_str());
}

bool IdbBuilder::saveLef(string file)
{
  if (IdbDefServiceResult::kServiceFailed == _def_service->DefFileWriteInit(file.c_str())) {
    std::cout << "Create LEF file failed..." << endl;
    return false;
  }

  std::shared_ptr<DefWrite> def_writer = std::make_shared<DefWrite>(_def_service, DefWriteType::kLef);
  return def_writer->writeDb(file.c_str());
}

void IdbBuilder::saveVerilog(std::string verilog_file_name, std::set<std::string>& exclude_cell_names, bool is_add_space_for_escape_name)
{
  IdbDesign* idb_design = _def_service->get_design();
  VerilogWriter writer(verilog_file_name.c_str(), exclude_cell_names, *idb_design, is_add_space_for_escape_name);
  writer.writeModule();
}

bool IdbBuilder::saveGDSII(string file)
{
  if (IdbDefServiceResult::kServiceFailed == _def_service->DefFileWriteInit(file.c_str())) {
    std::cout << "Create GDSII file failed..." << endl;
    return false;
  }

  std::shared_ptr<Def2GdsWrite> gds_write = std::make_shared<Def2GdsWrite>(_def_service);
  return gds_write->writeDb(file.c_str());
}

bool IdbBuilder::saveJSON(string file, string options)
{
  if (IdbDefServiceResult::kServiceFailed == _def_service->DefFileWriteInit(file.c_str())) {
    std::cout << "Create JSON file failed..." << endl;
    return false;
  }
  // std::cout << options << endl;
  std::shared_ptr<Gds2JsonWrite> json_write = std::make_shared<Gds2JsonWrite>(_def_service);
  return json_write->writeDb(file.c_str(), options);
}

// void IdbBuilder::saveLayout(string folder)
// {
//   if (IdbDataServiceResult::kServiceFailed == _data_service->LayoutFileWriteInit(folder.c_str())) {
//     std::cout << "Write layout failed..." << endl;
//   }

//   std::shared_ptr<LayoutWrite> layout_write = std::make_shared<LayoutWrite>(_data_service->get_def_service()->get_layout());

//   layout_write->writeLayout(folder.c_str());
// }

// void IdbBuilder::loadLayout(string folder)
// {
//   if (IdbDataServiceResult::kServiceFailed == _data_service->LayoutFileReadInit(folder.c_str())) {
//     std::cout << "Read layout failed..." << endl;
//   }

//   LayoutRead* layout_read = new LayoutRead();

//   IdbLayout* layout = layout_read->readLayout(folder.c_str());

//   IdbDefService* def_service = new IdbDefService(layout);
//   _data_service->set_def_service(def_service);
// }

}  // namespace idb
