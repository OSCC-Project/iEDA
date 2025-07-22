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
 * @file		def_write.cpp
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
* @description


        There is a def builder to write def file from data structure.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "def_write.h"

#include <algorithm>
#include <cstdint>

#include "../../../data/design/IdbDesign.h"
#include "Str.hh"
#include "boost_definition.h"

using std::cout;
using std::endl;

namespace idb {

/**
 * @brief Constructor for DefWrite class.
 *
 * @param def_service Pointer to IdbDefService object.
 * @param type Type of DefWrite.
 */
DefWrite::DefWrite(IdbDefService* def_service, DefWriteType type)
{
  _def_service = def_service;
  _file_write = nullptr;
  _type = type;
}

DefWrite::~DefWrite()
{
}

/**
 * @brief Initialize the output file.
 *
 * @param file Path to the file.
 * @return true if initialization succeeds, false otherwise.
 */
bool DefWrite::initFile(const char* file)
{
  // Creating a compressed format handle
  if (ieda::Str::contain(file, ".gz")) {
    _font = SaveFormat::kGzip;
    _file_write_gz = gzopen(file, "w");

    if (_file_write_gz == nullptr) {
      std::cout << "Open gz file failed..." << std::endl;
      return false;
    }
  } else {
    _font = SaveFormat::kUnzip;
    _file_write = fopen(file, "w+");
    if (_file_write == nullptr) {
      std::cerr << "Open def file failed..." << std::endl;
      return false;
    }
  }

  return true;
}

/**
 * @brief Close the output file.
 *
 * @return true if file closure succeeds, false otherwise.
 */
bool DefWrite::closeFile()
{
  bool result;
  switch (_font) {
    case SaveFormat::kGzip:
      result = gzclose(_file_write_gz);
      _file_write_gz = nullptr;
      break;

    case SaveFormat::kUnzip:
    default:
      result = fclose(_file_write);
      _file_write = nullptr;
      break;
  }
  return result;
}

/**
 * @brief Write formatted string data to the output file.
 *
 * @param strdata Formatted string data.
 */
void DefWrite::writestr(const char* strdata, ...)
{
  va_list args;
  va_start(args, strdata);
  switch (_font) {
    case SaveFormat::kUnzip:
      vfprintf(_file_write, strdata, args);
      break;
    case SaveFormat::kGzip:
      gzvprintf(_file_write_gz, strdata, args);
      break;
  }
  va_end(args);
}

/**
 * @brief Write the design to the output file.
 *
 * @param file Path to the file.
 * @return true if writing succeeds, false otherwise.
 */
bool DefWrite::writeDb(const char* file)
{
  if (!initFile(file)) {
    return false;
  }

  switch (_type) {
    case DefWriteType::kChip: {
      writeChip();
      break;
    }
    case DefWriteType::kSynthesis: {
      writeDbSynthesis();
      break;
    }
    case DefWriteType::kFloorplan:
    case DefWriteType::kGlobalPlace:
    case DefWriteType::kDetailPlace:
    case DefWriteType::kGlobalRouting:
    case DefWriteType::kDetailRouting: {
      writeChip();
      break;
    }
    case DefWriteType::kLef: {
      writeLef();
      break;
    }

    default: {
      writeChip();
      break;
    }
  }

  return closeFile();
}

/**
 * @brief Write the design to the lef fine.
 *
 * @param file Path to the file.
 * @return true if writing succeeds, false otherwise.
 */
bool DefWrite::writeLef()
{
  write_version();
  write_divider_char();
  write_busbit_char();
  write_lef_macro();

  writestr("\n");
  write_libreary_end();

  return true;
}

/**
 * @brief Write the design to the output file.
 *
 * @param file Path to the file.
 * @return true if writing succeeds, false otherwise.
 */
bool DefWrite::writeChip()
{
  write_version();
  write_divider_char();
  write_busbit_char();
  write_design();
  write_units();
  write_die();
  write_row();
  write_track_grid();
  write_gcell_grid();
  write_via();
  write_component();
  write_pin();
  write_blockage();
  write_region();
  write_slot();
  write_group();
  write_fill();
  write_special_net();
  write_net();

  write_end();

  return true;
}

/**
 * @brief Writes Synthesis related information to the DEF file.
 * @return True if writing is successful, false otherwise.
 */
bool DefWrite::writeDbSynthesis()
{
  write_version();
  write_design();
  write_units();
  write_component();
  write_pin();
  write_net();

  write_end();

  return true;
}

/**
 * @brief Writes END DESIGN marker to the DEF file.
 * @return Status code indicating success or failure.
 */
int32_t DefWrite::write_end()
{
  writestr("END DESIGN\n");
  return kDbSuccess;
}

int32_t DefWrite::write_libreary_end()
{
  writestr("END LIBRARY\n");
  return kDbSuccess;
}

/**
 * @brief Writes the VERSION information to the DEF file.
 * @return Status code indicating success or failure.
 */
int32_t DefWrite::write_version()
{
  IdbDesign* design = _def_service->get_design();
  /// support version 5.8
  string version = design->get_version().empty() ? "5.8" : design->get_version();
  writestr("VERSION %s ;\n", version.c_str());

  std::cout << "Write VERSION success..." << std::endl;
  return kDbSuccess;
}

int32_t DefWrite::write_divider_char()
{
  writestr("DIVIDERCHAR \"/\" ;\n");

  return kDbSuccess;
}

int32_t DefWrite::write_busbit_char()
{
  IdbDesign* design = this->get_service()->get_design();
  IdbBusBitChars* bus_bit_chars = design->get_bus_bit_chars();

  if (bus_bit_chars == nullptr) {
    return kDbFail;
  }

  writestr("BUSBITCHARS \"%c%c\" ;\n", bus_bit_chars->getLeftDelimiter(), bus_bit_chars->getRightDelimiter());

  std::cout << "Write BUSBITCHARS success..." << std::endl;

  return kDbSuccess;
}

/**
 * @brief Writes the DESIGN name to the DEF file.
 * @return Status code indicating success or failure.
 */
int32_t DefWrite::write_design()
{
  IdbDesign* design = _def_service->get_design();
  string design_name = design->get_design_name();
  writestr("DESIGN %s ;\n", design_name.c_str());

  std::cout << "Write DESIGN name success..." << std::endl;
  return kDbSuccess;
}

int32_t DefWrite::write_units()
{
  IdbDesign* design = _def_service->get_design();
  IdbUnits* def_units = design->get_units();
  IdbUnits* lef_units = design->get_layout()->get_units();
  if (def_units == nullptr && lef_units == nullptr) {
    std::cout << "Write UNITS error..." << std::endl;

    return kDbFail;
  }

  uint32_t def_microns = def_units->get_micron_dbu() > 0 ? def_units->get_micron_dbu() : lef_units->get_micron_dbu();
  if (def_microns <= 0) {
    std::cout << "Write UNITS error..." << std::endl;

    return kDbFail;
  }
  writestr("UNITS DISTANCE MICRONS %u ;\n", def_microns);
  std::cout << "Write UNITS success..." << std::endl;
  return kDbSuccess;
}

int32_t DefWrite::write_die()
{
  IdbLayout* layout = _def_service->get_layout();
  IdbDie* die = layout->get_die();
  if (die == nullptr) {
    std::cout << "Write DIE error..." << std::endl;

    return kDbFail;
  }

  writestr("DIEAREA ");

  for (IdbCoordinate<int32_t>* point : die->get_points()) {
    writestr("( %d %d ) ", point->get_x(), point->get_y());
  }

  writestr(";\n");

  std::cout << "Write DIE success..." << std::endl;
  return kDbSuccess;
}

int32_t DefWrite::write_track_grid()
{
  IdbLayout* layout = _def_service->get_layout();
  IdbTrackGridList* track_grid_list = layout->get_track_grid_list();
  if (track_grid_list == nullptr) {
    std::cout << "Write Track Grid error..." << std::endl;
    return kDbFail;
  }

  for (IdbTrackGrid* track : track_grid_list->get_track_grid_list()) {
    string direction = IdbEnum::GetInstance()->get_layer_property()->get_track_direction_name(track->get_track()->get_direction());

    writestr("TRACKS %s %d DO %d STEP %d ", direction.c_str(), track->get_track()->get_start(), track->get_track_num(),
             track->get_track()->get_pitch());

    writestr("LAYER ");

    for (IdbLayer* layer : track->get_layer_list()) {
      writestr("%s ", layer->get_name().c_str());
    }

    writestr(";\n \n");
  }

  std::cout << "Write Track Grid success..." << std::endl;
  return kDbSuccess;
}

int32_t DefWrite::write_via()
{
  IdbDesign* design = _def_service->get_design();  // Def
  IdbVias* via_list = design->get_via_list();
  if (via_list == nullptr) {
    std::cout << "Write VIAS error" << std::endl;
    return kDbFail;
  }

  if (via_list->get_num_via() == 0) {
    std::cout << "No VIAS To Write..." << std::endl;
    return kDbFail;
  }

  writestr("VIAS %ld ;\n", via_list->get_num_via());

  for (IdbVia* via : via_list->get_via_list()) {
    IdbViaMaster* via_master = via->get_instance();

    if (via_master->is_generate()) {
      IdbViaMasterGenerate* master_generate = via_master->get_master_generate();

      writestr(
          "- %s + VIARULE %s + CUTSIZE %d %d + LAYERS %s %s %s + CUTSPACING %d %d + ENCLOSURE %d %d %d %d "
          " + ROWCOL %d %d \n",
          via->get_name().c_str(), master_generate->get_rule_name().c_str(), master_generate->get_cut_size_x(),
          master_generate->get_cut_size_y(), master_generate->get_layer_bottom()->get_name().c_str(),
          master_generate->get_layer_cut()->get_name().c_str(), master_generate->get_layer_top()->get_name().c_str(),
          master_generate->get_cut_spcing_x(), master_generate->get_cut_spcing_y(), master_generate->get_enclosure_bottom_x(),
          master_generate->get_enclosure_bottom_y(), master_generate->get_enclosure_top_x(), master_generate->get_enclosure_top_y(),
          master_generate->get_cut_rows(), master_generate->get_cut_cols());

      if (nullptr != master_generate->get_patttern()) {
        writestr(" + PATTERN %s \n", master_generate->get_patttern()->get_pattern_string().c_str());
      }

      writestr(" ;\n");
    }
  }

  writestr("END VIAS\n \n");

  std::cout << "Write VIAS success..." << std::endl;

  return kDbSuccess;
}

int32_t DefWrite::write_row()
{
  IdbLayout* layout = _def_service->get_layout();
  IdbRows* rows = layout->get_rows();
  if (rows == nullptr) {
    std::cout << "Write ROWS error..." << std::endl;
    return kDbFail;
  }

  for (IdbRow* row : rows->get_row_list()) {
    IdbSite* site = row->get_site();
    string site_orient = IdbEnum::GetInstance()->get_site_property()->get_orient_name(site->get_orient());
    writestr("ROW %s %s %d %d %s DO %d BY %d STEP %d %d ;\n", row->get_name().c_str(), row->get_site()->get_name().c_str(),
             row->get_original_coordinate()->get_x(), row->get_original_coordinate()->get_y(), site_orient.c_str(), row->get_row_num_x(),
             row->get_row_num_y(), row->get_step_x(), row->get_step_y());
  }

  writestr(" \n");

  std::cout << "Write ROWS success..." << std::endl;
  return kDbSuccess;
}

int32_t DefWrite::write_component()
{
  IdbDesign* design = _def_service->get_design();  // Def
  IdbInstanceList* instance_list = design->get_instance_list();
  if (instance_list == nullptr) {
    std::cout << "Write COMPONENTS error..." << std::endl;
    return kDbFail;
  }

  if (instance_list->get_num() == 0) {
    std::cout << "No COMPONENT To Write..." << std::endl;
    return kDbFail;
  }

  writestr("COMPONENTS %d ;\n", instance_list->get_num());

  for (IdbInstance* instance : instance_list->get_instance_list()) {
    std::string inst_name = instance->get_name();
    std::string new_inst_name = ieda::Str::addBackslash(inst_name);

    string type = instance->get_type() != IdbInstanceType::kNone
                      ? "+ SOURCE " + IdbEnum::GetInstance()->get_instance_property()->get_type_str(instance->get_type())
                      : "";
    string status = IdbEnum::GetInstance()->get_instance_property()->get_status_str(instance->get_status());
    string orient = IdbEnum::GetInstance()->get_site_property()->get_orient_name(instance->get_orient());

    if (instance->has_placed()) {
      writestr("    - %s %s %s + %s ( %d %d ) %s \n", new_inst_name.c_str(), instance->get_cell_master()->get_name().c_str(), type.c_str(),
               status.c_str(), instance->get_coordinate()->get_x(), instance->get_coordinate()->get_y(), orient.c_str());
    } else {
      writestr("    - %s %s %s \n", new_inst_name.c_str(), instance->get_cell_master()->get_name().c_str(), type.c_str());
    }

    /// halo
    auto halo = instance->get_halo();
    if (halo != nullptr) {
      std::string str_soft = halo->is_soft() ? " [SOFT] " : " ";
      writestr("      + HALO%s%d %d %d %d\n", str_soft.c_str(), halo->get_extend_lef(), halo->get_extend_bottom(), halo->get_extend_right(),
               halo->get_extend_top());
    }

    /// routed halo
    auto route_halo = instance->get_route_halo();
    if (route_halo != nullptr) {
      writestr("      + ROUTEHALO %d %s %s\n", route_halo->get_route_distance(), route_halo->get_layer_bottom()->get_name().c_str(),
               route_halo->get_layer_top()->get_name().c_str());
    }

    writestr("      ;\n");
  }

  writestr("END COMPONENTS\n \n");

  std::cout << "Write COMPONENTS success..." << std::endl;
  return kDbSuccess;
}

int32_t DefWrite::write_pin()
{
  IdbDesign* design = _def_service->get_design();
  IdbPins* pin_list = design->get_io_pin_list();
  if (pin_list == nullptr) {
    std::cout << "Write PINS error..." << std::endl;
    return kDbFail;
  }

  writestr("PINS %d ;\n", pin_list->get_pin_num());

  for (IdbPin* pin : pin_list->get_pin_list()) {
    string direction = IdbEnum::GetInstance()->get_connect_property()->get_direction_name(pin->get_term()->get_direction());
    string use = IdbEnum::GetInstance()->get_connect_property()->get_type_name(pin->get_term()->get_type());
    string is_special = pin->is_special_net_pin() || pin->get_term()->is_special_net() ? "+ SPECIAL " : "";

    writestr(" - %s + NET %s %s+ DIRECTION %s", pin->get_pin_name().c_str(), pin->get_net_name().c_str(), is_special.c_str(),
             direction.c_str());

    if (use.empty()) {
      writestr("  \n");
    } else {
      writestr("  + USE %s\n", use.c_str());
    }

    if (pin->get_term()->is_port_exist() || pin->is_special_net_pin()) {
      for (IdbPort* port : pin->get_term()->get_port_list()) {
        writestr("  + PORT\n");

        string status = IdbEnum::GetInstance()->get_instance_property()->get_status_str(port->get_placement_status());
        string orient = IdbEnum::GetInstance()->get_site_property()->get_orient_name(port->get_orient());
        for (IdbLayerShape* layer_shape : port->get_layer_shape()) {
          writestr("   + LAYER %s ", layer_shape->get_layer()->get_name().c_str());
          for (IdbRect* rect : layer_shape->get_rect_list()) {
            writestr("( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
          }

          if (port->is_placed()) {
            writestr("+ %s ( %d %d ) %s", status.c_str(), port->get_coordinate()->get_x(), port->get_coordinate()->get_y(), orient.c_str());
          }
          writestr("\n");
        }
      }
    } else {
      string status = IdbEnum::GetInstance()->get_instance_property()->get_status_str(pin->get_term()->get_placement_status());
      string orient = IdbEnum::GetInstance()->get_site_property()->get_orient_name(pin->get_orient());
      for (IdbPort* port : pin->get_term()->get_port_list()) {
        for (IdbLayerShape* layer_shape : port->get_layer_shape()) {
          writestr(" + LAYER %s ", layer_shape->get_layer()->get_name().c_str());
          for (IdbRect* rect : layer_shape->get_rect_list()) {
            writestr("( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
          }

          if (pin->get_term()->is_placed()) {
            writestr("+ %s ( %d %d ) %s", status.c_str(), pin->get_location()->get_x(), pin->get_location()->get_y(), orient.c_str());
          }
        }
      }
      writestr("\n");
    }

    writestr(";\n");
  }

  writestr("END PINS\n \n");

  cout << "Write PINS success..." << endl;

  return kDbSuccess;
}

int32_t DefWrite::write_blockage()
{
  IdbDesign* design = _def_service->get_design();
  IdbBlockageList* blockage_list = design->get_blockage_list();
  if (blockage_list == nullptr) {
    std::cout << "Write VIAS error..." << std::endl;
    return kDbFail;
  }

  if (blockage_list->get_num() == 0) {
    std::cout << "No VIA To Write..." << std::endl;
    return kDbFail;
  }

  writestr("BLOCKAGES %d ;\n", blockage_list->get_num());

  for (IdbBlockage* blockage : blockage_list->get_blockage_list()) {
    if (blockage->get_type() == IdbBlockage::IdbBlockageType::kRoutingBlockage) {
      IdbRoutingBlockage* routing_blockage = dynamic_cast<IdbRoutingBlockage*>(blockage);

      writestr("    - LAYER %s ", routing_blockage->get_layer_name().c_str());

      if (routing_blockage->is_pushdown() == true) {
        writestr("+ PUSHDOWN ");
      }

      if (routing_blockage->is_except_pgnet() == true) {
        writestr("+ EXCEPTPGNET ");
      }

      if (routing_blockage->get_instance() != nullptr) {
        writestr("+ COMPONENT %s ", routing_blockage->get_instance_name().c_str());
      }

      for (IdbRect* rect : routing_blockage->get_rect_list()) {
        writestr("RECT ( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
      }
    } else if (blockage->get_type() == IdbBlockage::IdbBlockageType::kPlacementBlockage) {
      IdbPlacementBlockage* placement_blockage = dynamic_cast<IdbPlacementBlockage*>(blockage);

      writestr("    - PLACEMENT ");

      if (placement_blockage->is_pushdown() == true) {
        writestr("+ PUSHDOWN ");
      }

      if (placement_blockage->get_instance() != nullptr) {
        writestr("+ COMPONENT %s ", placement_blockage->get_instance_name().c_str());
      }

      for (IdbRect* rect : placement_blockage->get_rect_list()) {
        writestr("RECT ( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
      }
    }

    writestr(";\n");
  }

  writestr("END BLOCKAGES\n \n");

  std::cout << "Write BLOCKAGE success..." << std::endl;
  return kDbSuccess;
}

int32_t DefWrite::write_specialnet_wire_segment_points(IdbSpecialWireSegment* segment, string& wire_new_str)
{
  if (segment->get_point_list().size() < _POINT_MAX_) {
    std::cout << "Error special net wire point..." << std::endl;
    return kDbFail;
  }

  string shape = "";
  if (segment->get_shape_type() > IdbWireShapeType::kNone && segment->get_shape_type() < IdbWireShapeType::kMax) {
    shape = "+ SHAPE " + IdbEnum::GetInstance()->get_connect_property()->get_wire_shape_name(segment->get_shape_type());
  }

  if (segment->get_point_start()->get_x() == segment->get_point_second()->get_x()) {
    writestr(" %s%s %d %s ( %d %d ) ( * %d )\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(), segment->get_route_width(),
             shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), segment->get_point_second()->get_y());
  } else if (segment->get_point_start()->get_y() == segment->get_point_second()->get_y()) {
    writestr(" %s%s %d %s ( %d %d ) ( %d * )\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(), segment->get_route_width(),
             shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), segment->get_point_second()->get_x());
  } else {
    writestr(" %s%s %d %s ( %d %d ) ( %d %d )\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
             segment->get_route_width(), shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(),
             segment->get_point_second()->get_x(), segment->get_point_second()->get_y());
  }
  return kDbSuccess;
}

int32_t DefWrite::write_specialnet_wire_segment_via(IdbSpecialWireSegment* segment, string& wire_new_str)
{
  if (segment->get_point_list().size() <= 0 || segment->get_via() == nullptr) {
    std::cout << "Error special wire segment via..." << std::endl;
    return kDbFail;
  }

  string shape = "";
  if (segment->get_shape_type() > IdbWireShapeType::kNone && segment->get_shape_type() < IdbWireShapeType::kMax) {
    shape = "+ SHAPE " + IdbEnum::GetInstance()->get_connect_property()->get_wire_shape_name(segment->get_shape_type());
  }

  if (segment->get_point_list().size() == _POINT_MAX_) {
    if (segment->get_point_start()->get_x() == segment->get_point_second()->get_x()) {
      writestr(" %s%s %d %s ( %d %d ) ( * %d ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
               segment->get_route_width(), shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(),
               segment->get_point_second()->get_y(), segment->get_via()->get_name().c_str());
    } else if (segment->get_point_start()->get_y() == segment->get_point_second()->get_y()) {
      writestr(" %s%s %d %s ( %d %d ) ( %d * ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
               segment->get_route_width(), shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(),
               segment->get_point_second()->get_x(), segment->get_via()->get_name().c_str());
    } else {
      writestr(" %s%s %d %s ( %d %d ) ( %d %d ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
               segment->get_route_width(), shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(),
               segment->get_point_second()->get_x(), segment->get_point_second()->get_y(), segment->get_via()->get_name().c_str());
    }
  } else {
    writestr(" %s%s %d %s ( %d %d ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(), segment->get_route_width(),
             shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(),
             segment->get_via()->get_name().c_str());
  }

  return kDbSuccess;
}

int32_t DefWrite::write_specialnet_wire_segment_rect(IdbSpecialWireSegment* segment, string& wire_new_str)
{
  if (segment->get_layer() == nullptr || segment->get_delta_rect() == nullptr) {
    std::cout << "Error net wire segment rect..." << std::endl;
    return kDbFail;
  }

  string shape = "";
  if (segment->get_shape_type() > IdbWireShapeType::kNone && segment->get_shape_type() < IdbWireShapeType::kMax) {
    shape = "+ SHAPE " + IdbEnum::GetInstance()->get_connect_property()->get_wire_shape_name(segment->get_shape_type());
  }

  writestr(" %s%s + RECT %s ( %d %d ) ( %d %d ) \n", wire_new_str.c_str(), shape.c_str(), segment->get_layer()->get_name().c_str(),
           segment->get_delta_rect()->get_low_x(), segment->get_delta_rect()->get_low_y(), segment->get_delta_rect()->get_high_x(),
           segment->get_delta_rect()->get_high_x());

  return kDbSuccess;
}

int32_t DefWrite::write_specialnet_wire_segment(IdbSpecialWireSegment* segment, string& wire_new_str)
{
  if (segment->is_via()) {
    return write_specialnet_wire_segment_via(segment, wire_new_str);
  } else if (segment->is_rect()) {
    return write_specialnet_wire_segment_rect(segment, wire_new_str);
  } else {
    return write_specialnet_wire_segment_points(segment, wire_new_str);
  }

  return kDbSuccess;
}

int32_t DefWrite::write_specialnet_wire(IdbSpecialWire* wire)
{
  string wire_state = IdbEnum::GetInstance()->get_connect_property()->get_wiring_state_name(wire->get_wire_state());

  if (wire->get_wire_state() == IdbWiringStatement::kShield) {
    /// tbd do not support shield
    return kDbFail;
    wire_state = "  + " + wire_state + " " + wire->get_shiled_name() + " ";

  } else {
    wire_state = "  + " + wire_state + " ";
  }

  int32_t index = 0;
  for (IdbSpecialWireSegment* segment : wire->get_segment_list()) {
    string str_head = index == 0 ? wire_state : "    NEW ";
    write_specialnet_wire_segment(segment, str_head);
    index++;
  }

  return kDbSuccess;
}

int32_t DefWrite::write_special_net()
{
  IdbSpecialNetList* special_net_list = _def_service->get_design()->get_special_net_list();
  if (special_net_list == nullptr || special_net_list->get_num() == 0) {
    std::cout << "No SPECIALNETS..." << std::endl;
    return kDbFail;
  }

  writestr("SPECIALNETS %ld ;\n", special_net_list->get_num());

  for (IdbSpecialNet* special_net : special_net_list->get_net_list()) {
    writestr("- %s ", special_net->get_net_name().c_str());

    if (special_net->get_pin_string_list().size() > 0) {
      for (string pin_string : special_net->get_pin_string_list()) {
        writestr("( * %s ) ", pin_string.c_str());
      }
    } else {
      for (IdbPin* pin_io : special_net->get_io_pin_list()->get_pin_list()) {
        writestr("( PIN %s ) ", pin_io->get_pin_name().c_str());
      }

      for (IdbPin* pin_instance : special_net->get_instance_pin_list()->get_pin_list()) {
        writestr("( %s %s ) ", pin_instance->get_instance()->get_name().c_str(), pin_instance->get_pin_name().c_str());
      }
    }

    writestr("\n");

    string connect_str = IdbEnum::GetInstance()->get_connect_property()->get_type_name(special_net->get_connect_type());
    writestr("  + USE %s \n", connect_str.c_str());

    for (IdbSpecialWire* wire : special_net->get_wire_list()->get_wire_list()) {
      write_specialnet_wire(wire);
    }

    writestr(" ;\n");
  }

  writestr("END SPECIALNETS\n \n");

  std::cout << "Write SPECIALNETS success..." << std::endl;

  return kDbSuccess;
}

int32_t DefWrite::write_net()
{
  IdbDesign* design = _def_service->get_design();  // Def
  IdbNetList* net_list = design->get_net_list();
  if (net_list == nullptr) {
    std::cout << "No NET To Write..." << std::endl;
    return kDbFail;
  }

  if (net_list->get_num() == 0) {
    std::cout << "NO NET ..." << std::endl;
    return kDbFail;
  }

  writestr("NETS %ld ;\n", net_list->get_num());

  for (IdbNet* net : net_list->get_net_list()) {
    std::string net_name = net->get_net_name();
    std::string net_name_new = ieda::Str::addBackslash(net_name);
    writestr("- %s", net_name_new.c_str());

    auto* io_pins = net->get_io_pins();
    for (auto* io_pin : io_pins->get_pin_list()) {
      writestr(" ( PIN %s )", io_pin->get_pin_name().c_str());
    }

    for (IdbPin* instance : net->get_instance_pin_list()->get_pin_list()) {
      writestr(" ( %s %s )", instance->get_instance()->get_name().c_str(), instance->get_pin_name().c_str());
    }

    writestr("\n");

    if (IdbConnectType::kNone < net->get_connect_type() && IdbConnectType::kMax > net->get_connect_type()) {
      string use = IdbEnum::GetInstance()->get_connect_property()->get_type_name(net->get_connect_type());
      writestr("  + USE %s \n", use.c_str());
    } else {
    }

    if (net->get_wire_list()->get_num() > 0) {
      for (IdbRegularWire* wire : net->get_wire_list()->get_wire_list()) {
        write_net_wire(wire);
      }
    }

    writestr(" ;\n");
  }

  writestr("END NETS\n \n");

  std::cout << "Write NETS success..." << std::endl;
  return kDbSuccess;
}

int32_t DefWrite::write_net_wire(IdbRegularWire* wire)
{
  string wire_state = IdbEnum::GetInstance()->get_connect_property()->get_wiring_state_name(wire->get_wire_statement());

  if (wire->get_wire_statement() == IdbWiringStatement::kShield) {
    wire_state = "  + " + wire_state + " " + wire->get_shiled_name() + " ";

  } else {
    wire_state = "  + " + wire_state + " ";
  }

  int index = 0;
  for (IdbRegularWireSegment* segment : wire->get_segment_list()) {
    string str_head = index == 0 ? wire_state : "    NEW ";
    write_net_wire_segment(segment, str_head);
    index++;
  }

  return kDbSuccess;
}

int32_t DefWrite::write_net_wire_segment(IdbRegularWireSegment* segment, string& wire_new_str)
{
  if (segment->is_rect()) {
    return write_net_wire_segment_rect(segment, wire_new_str);

  } else if (segment->is_via()) {
    return write_net_wire_segment_via(segment, wire_new_str);

  } else {
    // two points
    return write_net_wire_segment_points(segment, wire_new_str);
  }

  return kDbFail;
}

int32_t DefWrite::write_net_wire_segment_points(IdbRegularWireSegment* segment, string& wire_new_str)
{
  if (segment->get_point_list().size() < _POINT_MAX_ || segment->get_layer() == nullptr) {
    // std::cout << "Error net wire point..." << std::endl;
    return kDbFail;
  }
  bool is_virtual = segment->is_virtual(segment->get_point_second());
  const char* virtual_str = is_virtual ? "VIRTUAL " : "";

  if (segment->get_point_start()->get_x() == segment->get_point_second()->get_x()) {
    writestr("%s %s ( %d %d ) %s( * %d )\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
             segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), virtual_str, segment->get_point_second()->get_y());
  } else if (segment->get_point_start()->get_y() == segment->get_point_second()->get_y()) {
    writestr("%s %s ( %d %d ) %s( %d * )\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
             segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), virtual_str, segment->get_point_second()->get_x());
  } else {
    writestr("%s %s ( %d %d ) %s( %d %d )\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
             segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), virtual_str, segment->get_point_second()->get_x(),
             segment->get_point_second()->get_y());
  }
  return kDbSuccess;
}

int32_t DefWrite::write_net_wire_segment_via(IdbRegularWireSegment* segment, string& wire_new_str)
{
  if (segment->get_point_list().size() <= 0 || segment->get_layer() == nullptr || segment->get_via_list().size() <= 0) {
    std::cout << "Error net wire segment via..." << std::endl;
    return kDbFail;
  }

  if (segment->get_point_list().size() == _POINT_MAX_) {
    if (segment->get_point_start()->get_x() == segment->get_point_second()->get_x()) {
      writestr("%s %s ( %d %d ) ( * %d ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
               segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), segment->get_point_second()->get_y(),
               segment->get_via_list().at(_POINT_START_)->get_name().c_str());
    } else if (segment->get_point_start()->get_y() == segment->get_point_second()->get_y()) {
      writestr("%s %s ( %d %d ) ( %d * ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
               segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), segment->get_point_second()->get_x(),
               segment->get_via_list().at(_POINT_START_)->get_name().c_str());
    } else {
      writestr("%s %s ( %d %d ) ( %d %d ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
               segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), segment->get_point_second()->get_x(),
               segment->get_point_second()->get_y(), segment->get_via_list().at(0)->get_name().c_str());
    }
  } else {
    writestr("%s %s ( %d %d ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(), segment->get_point_start()->get_x(),
             segment->get_point_start()->get_y(), segment->get_via_list().at(_POINT_START_)->get_name().c_str());
  }

  return kDbSuccess;
}

int32_t DefWrite::write_net_wire_segment_rect(IdbRegularWireSegment* segment, string& wire_new_str)
{
  if (segment->get_point_list().size() <= 0 || segment->get_layer() == nullptr || segment->get_delta_rect() == nullptr) {
    std::cout << "Error net wire segment rect..." << std::endl;
    return kDbFail;
  }

  writestr("%s %s ( %d %d ) RECT ( %d %d %d %d ) \n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
           segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), segment->get_delta_rect()->get_low_x(),
           segment->get_delta_rect()->get_low_y(), segment->get_delta_rect()->get_high_x(), segment->get_delta_rect()->get_high_y());

  return kDbSuccess;
}

/**
 * @brief Write IO pins, create each IO Term in IdbPin
 *
 */

int32_t DefWrite::write_gcell_grid()
{
  IdbLayout* layout = _def_service->get_layout();  // Lef
  IdbGCellGridList* gcell_grid_list = layout->get_gcell_grid_list();
  if (gcell_grid_list == nullptr) {
    std::cout << "Write GCELLGRID error..." << std::endl;
    return kDbFail;
  }

  if (gcell_grid_list->get_gcell_grid_num() <= 0) {
    std::cout << "No GCELLGRID..." << std::endl;
    return kDbFail;
  }

  //   writestr( "GCELLGRID\n");

  for (IdbGCellGrid* gcell_grid : gcell_grid_list->get_gcell_grid_list()) {
    string direction_str = gcell_grid->get_direction() == IdbTrackDirection::kDirectionX ? "X" : "Y";

    writestr("GCELLGRID %s %d DO %d STEP %d ;\n", direction_str.c_str(), gcell_grid->get_start(), gcell_grid->get_num(),
             gcell_grid->get_space());
  }

  cout << "Write GCELLGRID success..." << endl;
  return kDbSuccess;
}

int32_t DefWrite::write_region()
{
  IdbDesign* design = _def_service->get_design();  // def
  IdbRegionList* region_list = design->get_region_list();
  if (region_list == nullptr) {
    std::cout << "Write REGIONS error..." << std::endl;
    return kDbFail;
  }
  if (region_list->get_num() == 0) {
    std::cout << "No REGION To Write..." << std::endl;
    return kDbFail;
  }

  writestr("REGIONS %d ;\n", region_list->get_num());

  for (IdbRegion* region : region_list->get_region_list()) {
    writestr("    - %s ", region->get_name().c_str());

    for (IdbRect* rect : region->get_boundary()) {
      writestr("( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
    }

    string type = IdbEnum::GetInstance()->get_region_property()->get_name(region->get_type());
    writestr("+ TYPE %s ", type.c_str());

    writestr(";\n");
  }

  cout << "Write REGIONS success..." << endl;
  return kDbSuccess;
}

int32_t DefWrite::write_slot()
{
  IdbDesign* design = _def_service->get_design();  // def
  IdbSlotList* slot_list = design->get_slot_list();
  if (slot_list == nullptr) {
    std::cout << "Write SLOTS error..." << std::endl;
    return kDbFail;
  }

  if (slot_list->get_num() == 0) {
    std::cout << "No SLOT To Write..." << std::endl;
    return kDbFail;
  }

  writestr("SLOTS %d ;\n", slot_list->get_num());

  for (IdbSlot* slot : slot_list->get_slot_list()) {
    writestr("    - LAYER %s ", slot->get_layer_name().c_str());

    for (IdbRect* rect : slot->get_rect_list()) {
      writestr("RECT ( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
    }

    writestr(";\n");
  }

  writestr("END SLOTS\n");

  cout << "Write SLOTS success..." << endl;
  return kDbSuccess;
}

int32_t DefWrite::write_group()
{
  IdbDesign* design = _def_service->get_design();  // def
  IdbGroupList* group_list = design->get_group_list();
  if (group_list == nullptr) {
    std::cout << "Write GROUPS error..." << std::endl;
    return kDbFail;
  }

  if (group_list->get_num() == 0) {
    std::cout << "No GROUP To Write..." << std::endl;
    return kDbFail;
  }

  writestr("GROUPS %d ;\n", group_list->get_num());

  for (IdbGroup* group : group_list->get_group_list()) {
    writestr("    - %s ", group->get_group_name().c_str());

    for (IdbInstance* instance : group->get_instance_list()->get_instance_list()) {
      writestr("%s ", instance->get_name().c_str());
    }

    writestr("+ REGION %s ", group->get_region()->get_name().c_str());

    writestr(";\n");
  }

  writestr("END GROUPS\n");

  cout << "Write GROUPS success..." << endl;
  return kDbSuccess;
}

int32_t DefWrite::write_fill()
{
  IdbDesign* design = _def_service->get_design();  // def
  IdbFillList* fill_list = design->get_fill_list();
  if (fill_list == nullptr) {
    std::cout << "Write FILLS error..." << std::endl;
    return kDbFail;
  }

  if (fill_list->get_num_fill() == 0) {
    std::cout << "No FILL To Write..." << std::endl;
    return kDbFail;
  }

  writestr("FILLS %d ;\n", fill_list->get_num_fill());

  for (IdbFill* fill : fill_list->get_fill_list()) {
    writestr("    - LAYER %s ", fill->get_layer()->get_layer()->get_name().c_str());

    for (IdbRect* rect : fill->get_layer()->get_rect_list()) {
      writestr("RECT ( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
    }

    writestr(";\n");

    writestr("    - VIA %s ", fill->get_via()->get_via()->get_name().c_str());

    for (IdbCoordinate<int32_t>* point : fill->get_via()->get_coordinate_list()) {
      writestr("( %d %d ) ", point->get_x(), point->get_y());
    }

    writestr(";\n");
  }

  cout << "Write FILLS success..." << endl;
  return kDbSuccess;
}

int32_t DefWrite::write_lef_macro()
{
  auto* design = _def_service->get_design();
  auto* layout = _def_service->get_layout();

  writestr("\n");
  writestr("MACRO %s\n", design->get_design_name().c_str());
  writestr("%sCLASS BLOCK ;\n", _spacer);
  writestr("%sFOREIGN %s ;\n", _spacer, design->get_design_name().c_str());
  writestr("%sORIGIN 0.0 0.0 ;\n", _spacer);
  writestr("%sSIZE %.3f BY %.3f ;\n", _spacer, design->transToUDB(layout->get_die()->get_width()),
           design->transToUDB(layout->get_die()->get_height()));
  writestr("%sSYMMETRY X Y ;\n", _spacer);

  write_lef_macro_pins();
  write_lef_macro_obs();

  writestr("\n");
  writestr("END %s\n", design->get_design_name().c_str());

  cout << "Write Macro success..." << endl;
  return kDbSuccess;
}

int32_t DefWrite::write_lef_macro_pins()
{
  auto* design = _def_service->get_design();
  auto* layout = _def_service->get_layout();
  auto* layers = layout->get_layers();
  auto* pdn_list = design->get_special_net_list();
  auto* die = layout->get_die();

  /// write io ports
  auto* io_pins = design->get_io_pin_list();
  for (auto idb_pin : io_pins->get_pin_list()) {
    writestr("%sPIN %s\n", _spacer, idb_pin->get_pin_name().c_str());
    /// definition
    {
      auto* io_term = idb_pin->get_term();

      auto* enum_property = IdbEnum::GetInstance()->get_connect_property();
      /// DIRECTION
      string direction = enum_property->get_direction_name(io_term->get_direction());
      writestr("%s%sDIRECTION %s ;\n", _spacer, _spacer, direction.c_str());
      /// USE
      string use = enum_property->get_type_name(io_term->get_type());
      writestr("%s%sUSE %s ;\n", _spacer, _spacer, use.c_str());
    }

    /// PORT
    {
      writestr("%s%sPORT\n", _spacer, _spacer);

      for (auto* layer_shape : idb_pin->get_port_box_list()) {
        writestr("%s%s%sLAYER %s ;\n", _spacer, _spacer, _spacer, layer_shape->get_layer()->get_name().c_str());
        for (auto* rect : layer_shape->get_rect_list()) {
          if (rect->get_low_x() < die->get_bounding_box()->get_low_x() || rect->get_low_y() < die->get_bounding_box()->get_low_y()
              || rect->get_high_x() > die->get_bounding_box()->get_high_x() || rect->get_high_y() > die->get_bounding_box()->get_high_y()) {
            std::cout << "error boundry" << std::endl;
          }

          writestr("%s%s%s%sRECT %.3f %.3f %.3f %.3f ;\n", _spacer, _spacer, _spacer, _spacer, design->transToUDB(rect->get_low_x()),
                   design->transToUDB(rect->get_low_y()), design->transToUDB(rect->get_high_x()), design->transToUDB(rect->get_high_y()));
        }
      }

      writestr("%s%sEND\n", _spacer, _spacer);
    }

    writestr("%sEND %s\n", _spacer, idb_pin->get_pin_name().c_str());
  }

  /// write top pdn
  auto get_top_pdn_rect = [&](IdbLayer* layer, bool is_power) -> std::vector<IdbRect> {
    std::vector<IdbRect> pdn_rects;

    for (auto* net : pdn_list->get_net_list()) {
      if ((net->is_vdd() && !is_power) || (net->is_vss() && is_power)) {
        continue;
      }

      for (auto* wire : net->get_wire_list()->get_wire_list()) {
        for (auto* segment : wire->get_segment_list()) {
          if ((segment->is_via() && segment->get_point_num() < 2) || segment->get_layer() != layer) {
            continue;
          }

          /// build segment rect
          int32_t routing_width = segment->get_route_width() == 0 ? ((IdbLayerRouting*) layer)->get_width() : segment->get_route_width();

          auto* point_1 = segment->get_point_start();
          auto* point_2 = segment->get_point_second();

          int32_t ll_x = 0;
          int32_t ll_y = 0;
          int32_t ur_x = 0;
          int32_t ur_y = 0;

          if (point_1->get_y() == point_2->get_y()) {
            // horizontal
            ll_x = std::min(point_1->get_x(), point_2->get_x());
            ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width / 2;
            ur_x = std::max(point_1->get_x(), point_2->get_x());
            ur_y = ll_y + routing_width;
          } else if (point_1->get_x() == point_2->get_x()) {
            // vertical
            ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width / 2;
            ll_y = std::min(point_1->get_y(), point_2->get_y());
            ur_x = ll_x + routing_width;
            ur_y = std::max(point_1->get_y(), point_2->get_y());
          }

          pdn_rects.emplace_back(IdbRect(ll_x, ll_y, ur_x, ur_y));
        }
      }
    }

    return pdn_rects;
  };

  auto layer_pair = get_pdn_layer_order_range();
  IdbLayer* top_layer = layers->find_layer_by_order(layer_pair.second);
  auto top_vdd = get_top_pdn_rect(top_layer, true);
  auto top_vss = get_top_pdn_rect(top_layer, false);

  auto write_pdn_as_port = [&](bool is_power) {
    auto& top_rects = is_power ? top_vdd : top_vss;
    if (top_rects.size() <= 0) {
      return;
    }

    std::string pin_name = is_power ? "vdd_stripe_generate" : "vss_stripe_generate";
    writestr("%sPIN %s\n", _spacer, pin_name.c_str());
    /// definition
    {
      /// DIRECTION
      string direction = "INOUT";
      writestr("%s%sDIRECTION %s ;\n", _spacer, _spacer, direction.c_str());
      /// USE
      string use = is_power ? "POWER" : "GROUND";
      writestr("%s%sUSE %s ;\n", _spacer, _spacer, use.c_str());
    }

    /// PORT
    {
      writestr("%s%sPORT\n", _spacer, _spacer);

      for (auto& rect : top_rects) {
        writestr("%s%s%sLAYER %s ;\n", _spacer, _spacer, _spacer, top_layer->get_name().c_str());
        writestr("%s%s%s%sRECT %.3f %.3f %.3f %.3f ;\n", _spacer, _spacer, _spacer, _spacer, design->transToUDB(rect.get_low_x()),
                 design->transToUDB(rect.get_low_y()), design->transToUDB(rect.get_high_x()), design->transToUDB(rect.get_high_y()));
      }

      writestr("%s%sEND\n", _spacer, _spacer);
    }

    writestr("%sEND %s\n", _spacer, pin_name.c_str());
  };

  write_pdn_as_port(true);
  write_pdn_as_port(false);

  return kDbSuccess;
}
std::pair<int, int> DefWrite::get_pdn_layer_order_range()
{
  auto* design = _def_service->get_design();
  auto* layout = _def_service->get_layout();
  auto* pdn_list = design->get_special_net_list();

  int min_layer = layout->get_layers()->get_layers_num() - 1;
  int max_layer = 0;

  for (auto* net : pdn_list->get_net_list()) {
    for (auto* wire : net->get_wire_list()->get_wire_list()) {
      for (auto* segment : wire->get_segment_list()) {
        if (segment->is_via() && segment->get_point_num() < 2) {
          continue;
        }
        auto order = segment->get_layer()->get_order();
        min_layer = std::min(min_layer, (int) order);
        max_layer = std::max(max_layer, (int) order);
      }
    }
  }
  return std::make_pair(min_layer, max_layer);
}

int32_t DefWrite::write_lef_macro_obs()
{
  auto* design = _def_service->get_design();
  auto* layout = _def_service->get_layout();
  auto* die = layout->get_die();
  auto* layers = layout->get_layers();
  auto* io_pins = design->get_io_pin_list();
  auto* pdn_list = design->get_special_net_list();

  auto get_obs_rect = [&](IdbLayer* layer, bool is_top) -> std::vector<IdbRect> {
    std::vector<IdbRect> obs_list;

    ieda_solver::GtlPolygon90Set polyset_die;

    auto* die_bbox = die->get_bounding_box();
    ieda_solver::GtlRect die_rect(die_bbox->get_low_x(), die_bbox->get_low_y(), die_bbox->get_high_x(), die_bbox->get_high_y());
    polyset_die += die_rect;

    ieda_solver::GtlPolygon90Set polyset_data;
    if (is_top) {
      /// exclude pdn data for top layer of pdn
      for (auto* net : pdn_list->get_net_list()) {
        for (auto* wire : net->get_wire_list()->get_wire_list()) {
          for (auto* segment : wire->get_segment_list()) {
            if ((segment->is_via() && segment->get_point_num() < 2) || segment->get_layer() != layer) {
              continue;
            }

            /// build segment rect
            int32_t routing_width = segment->get_route_width() == 0 ? ((IdbLayerRouting*) layer)->get_width() : segment->get_route_width();

            auto* point_1 = segment->get_point_start();
            auto* point_2 = segment->get_point_second();

            int32_t ll_x = 0;
            int32_t ll_y = 0;
            int32_t ur_x = 0;
            int32_t ur_y = 0;

            if (point_1->get_y() == point_2->get_y()) {
              // horizontal
              ll_x = std::min(point_1->get_x(), point_2->get_x());
              ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width / 2;
              ur_x = std::max(point_1->get_x(), point_2->get_x());
              ur_y = ll_y + routing_width;
            } else if (point_1->get_x() == point_2->get_x()) {
              // vertical
              ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width / 2;
              ll_y = std::min(point_1->get_y(), point_2->get_y());
              ur_x = ll_x + routing_width;
              ur_y = std::max(point_1->get_y(), point_2->get_y());
            }

            /// enlarge spacing required for drc
            int32_t required_size_h = ((IdbLayerRouting*) layer)->get_spacing(ur_x - ll_x, ur_y - ll_y);
            int32_t required_size_v = ((IdbLayerRouting*) layer)->get_spacing(ur_y - ll_y, ur_x - ll_x);

            ieda_solver::GtlRect bloat_rect(ll_x, ll_y, ur_x, ur_y);

            gtl::bloat(bloat_rect, gtl::HORIZONTAL, required_size_h);
            gtl::bloat(bloat_rect, gtl::VERTICAL, required_size_v);

            polyset_data += bloat_rect;
          }
        }
      }
    }

    /// exclude io port data
    for (auto* pin : io_pins->get_pin_list()) {
      for (auto* layer_shape : pin->get_port_box_list()) {
        if (layer_shape->get_layer() != layer) {
          continue;
        }

        for (auto* port_rect : layer_shape->get_rect_list()) {
          // get require spacing
          int32_t required_size_h = ((IdbLayerRouting*) layer)->get_spacing(port_rect->get_width(), port_rect->get_height());
          int32_t required_size_v = ((IdbLayerRouting*) layer)->get_spacing(port_rect->get_height(), port_rect->get_width());

          ieda_solver::GtlRect bloat_rect(port_rect->get_low_x(), port_rect->get_low_y(), port_rect->get_high_x(), port_rect->get_high_y());

          gtl::bloat(bloat_rect, gtl::HORIZONTAL, required_size_h);
          gtl::bloat(bloat_rect, gtl::VERTICAL, required_size_v);

          polyset_data += bloat_rect;
        }
      }
    }

    polyset_die.clean();
    polyset_data.clean();

    auto polyset_obs = polyset_die - polyset_data;

    auto direction = ((IdbLayerRouting*) layer)->is_horizontal() ? gtl::HORIZONTAL : gtl::VERTICAL;
    std::vector<ieda_solver::GtlRect> obs_rects;
    gtl::get_rectangles(obs_rects, polyset_obs, direction);
    obs_list.reserve(obs_rects.size());
    for (auto obs_rect : obs_rects) {
      obs_list.emplace_back(IdbRect(gtl::xl(obs_rect), gtl::yl(obs_rect), gtl::xh(obs_rect), gtl::yh(obs_rect)));
    }

    return obs_list;
  };

  /// write obs
  writestr("%sOBS\n", _spacer);

  auto layer_pair = get_pdn_layer_order_range();
  for (auto layer_order = layer_pair.first; layer_order <= layer_pair.second; layer_order += 2) {
    IdbLayer* layer = layers->find_layer_by_order(layer_order);
    auto is_top = layer_order == layer_pair.second ? true : false;
    auto obs_rects = get_obs_rect(layer, is_top);

    for (auto& obs_rect : obs_rects) {
      writestr("%s%sLAYER %s ;\n", _spacer, _spacer, layer->get_name().c_str());
      writestr("%s%sRECT %.3f %.3f %.3f %.3f ;\n", _spacer, _spacer, design->transToUDB(obs_rect.get_low_x()),
               design->transToUDB(obs_rect.get_low_y()), design->transToUDB(obs_rect.get_high_x()),
               design->transToUDB(obs_rect.get_high_y()));
    }
  }

  writestr("%sEND\n", _spacer);

  return kDbSuccess;
}

}  // namespace idb
