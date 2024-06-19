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
#include <sstream>
#include "IdbDesign.h"
#include "IdbEnum.h"
#include "ReportTable.hh"
#include "idm.h"
#include "report_db.h"

namespace iplf {


static std::string getInstOrient(idb::IdbOrient orient)
{
  switch (orient) {
    case idb::IdbOrient::kNone:
      return "None";
    case idb::IdbOrient::kN_R0:
      return "N_RO";
    case idb::IdbOrient::kS_R180:
      return "S_R180";
    case idb::IdbOrient::kW_R90:
      return "W_R90";
    case idb::IdbOrient::kE_R270:
      return "E_R270";
    case idb::IdbOrient::kFN_MY:
      return "FN_MY";
    case idb::IdbOrient::kFS_MX:
      return "FS_MX";
    case idb::IdbOrient::kFW_MX90:
      return "FW_MX90";
    case idb::IdbOrient::kFE_MY90:
      return "FE_MY90";
    default:
      return "Unknown";
  }
}

std::string ReportDesign::title()
{
  return "";
}

std::shared_ptr<ieda::ReportTable> ReportDesign::createInstanceTable(const std::string& inst_name)
{
  auto* inst_ptr = dmInst->get_idb_design()->get_instance_list()->find_instance(inst_name);
  if (inst_ptr == nullptr) {
    return {};
  }
  
  return createInstanceTable(inst_ptr);
}

std::shared_ptr<ieda::ReportTable> ReportDesign::createInstanceTable(idb::IdbInstance* inst_ptr)

{
  assert(inst_ptr);
  auto& inst = *inst_ptr;
  std::vector<std::string> header = {"Property", "Value"};
  auto tbl = std::make_shared<ieda::ReportTable>("IdbInstance", header, static_cast<int>(ReportDBType::kInstance));
  auto& instProperty =* IdbEnum::GetInstance()->get_instance_property();

  *tbl << "Name" << inst.get_name() << TABLE_ENDLINE;
  *tbl << "CellMaster" << inst.get_cell_master()->get_name() << TABLE_ENDLINE;
  *tbl << "Type" << instProperty.get_type_str(inst.get_type()) << TABLE_ENDLINE;
  *tbl << "Status" << instProperty.get_status_str(inst.get_status()) << TABLE_ENDLINE;
  if (inst.is_placed() || inst.is_fixed()) {
    *tbl << "Position" << ieda::Str::printf("(%d,%d)", inst.get_coordinate()->get_x(), inst.get_coordinate()->get_y()) << TABLE_ENDLINE;
  }
  *tbl << "Orient" << getInstOrient(inst.get_orient()) << TABLE_ENDLINE;

  auto* box = inst.get_bounding_box();
  *tbl << "Box" << ieda::Str::printf("(%d, %d, %d, %d)", box->get_low_x(), box->get_low_y(), box->get_high_x(), box->get_high_y())
       << TABLE_ENDLINE;
  *tbl << "Box Size" << ieda::Str::printf("%d, %d", box->get_width(), box->get_height()) << TABLE_ENDLINE;
  *tbl << "Pins Nums" << inst.get_pin_list()->get_pin_num() << TABLE_ENDLINE;
  
  return tbl;
}

std::shared_ptr<ieda::ReportTable> ReportDesign::createInstancePinTable(idb::IdbInstance* inst_ptr)
{
  assert(inst_ptr);
  std::vector<std::string> header = {"Pin Name", "Coordinate", "Type", "Net Name"};
  auto tbl = std::make_shared<ieda::ReportTable>("Instance pin list", header, static_cast<int>(ReportDBType::kInstancePinList));
  for (auto* pin : inst_ptr->get_pin_list()->get_pin_list()) {
    *tbl << pin->get_pin_name() << ieda::Str::printf("(%d, %d)", pin->get_average_coordinate()->get_x(), pin->get_average_coordinate()->get_y());
    if (pin->is_net_pin()) {
      *tbl << "net pin" << pin->get_net()->get_net_name();
    } else if (pin->is_special_net_pin()) {
      *tbl << " special net pin" << pin->get_special_net()->get_net_name();
    } else if (pin->is_io_pin()) {
      *tbl << "IO pin" << TABLE_SKIP;
    }
    *tbl << TABLE_ENDLINE;
  }
  return tbl;
}

std::shared_ptr<ieda::ReportTable> ReportDesign::createNetTable(idb::IdbNet* net_ptr)
{
  assert(net_ptr);
  auto& net = *net_ptr;

  std::vector<std::string> header = {"Property", "Value"};
  auto tbl = std::make_shared<ieda::ReportTable>("IdbNet", header, static_cast<int>(ReportDBType::kNet));

  auto& netEnum = *IdbEnum::GetInstance()->get_connect_property();
  auto fullName = [](IdbPin* pin) -> std::string { return pin ? pin->get_instance()->get_name() + "/" + pin->get_pin_name() : ""; };
  *tbl << "Name" << net.get_net_name() << TABLE_ENDLINE;
  *tbl << "Original Name" << net.get_original_net_name() << TABLE_ENDLINE;
  *tbl << "Type" << netEnum.get_type_name(net.get_connect_type()) << TABLE_ENDLINE;
  *tbl << "NetLength" << net.wireLength() << TABLE_ENDLINE;
  *tbl << "DrivingPin" << fullName(net.get_driving_pin()) << TABLE_ENDLINE;
  *tbl << "LoadPin" << [&net, &fullName]() -> std::string {
    std::stringstream ss;
    for (auto* pin : net.get_load_pins()) {
      ss << fullName(pin) << std::endl;
    }
    return ss.str();
  }() << TABLE_ENDLINE;
  return tbl;
}
}  // namespace iplf