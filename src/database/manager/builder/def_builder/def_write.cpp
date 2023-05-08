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

#include "../../../data/design/IdbDesign.h"
#include "Str.hh"

using std::cout;
using std::endl;

namespace idb {

DefWrite::DefWrite(IdbDefService* def_service, DefWriteType type)
{
  _def_service = def_service;
  file_write = nullptr;
  _type = type;
}

DefWrite::~DefWrite()
{
}

bool DefWrite::initFile(const char* file)
{
  file_write = fopen(file, "w+");
  if (file_write == nullptr) {
    std::cout << "Open def file failed..." << std::endl;
    return false;
  }
  return true;
}

bool DefWrite::closeFile()
{
  bool result = fclose(file_write);
  file_write = nullptr;
  return result;
}

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

    default: {
      writeChip();
      break;
    }
  }

  return closeFile();
}

bool DefWrite::writeChip()
{
  write_version();
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

int32_t DefWrite::write_end()
{
  fprintf(file_write, "END DESIGN\n");
  return kDbSuccess;
}

int32_t DefWrite::write_version()
{
  IdbDesign* design = _def_service->get_design();
  /// support version 5.8
  string version = design->get_version().empty() ? "5.8" : design->get_version();
  fprintf(file_write, "VERSION %s ;\n", version.c_str());

  std::cout << "Write VERSION success..." << std::endl;
  return kDbSuccess;
}

int32_t DefWrite::write_design()
{
  IdbDesign* design = _def_service->get_design();
  string design_name = design->get_design_name();
  fprintf(file_write, "DESIGN %s ;\n", design_name.c_str());

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
  fprintf(file_write, "UNITS DISTANCE MICRONS %u ;\n", def_microns);
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

  fprintf(file_write, "DIEAREA ");

  for (IdbCoordinate<int32_t>* point : die->get_points()) {
    fprintf(file_write, "( %d %d ) ", point->get_x(), point->get_y());
  }

  fprintf(file_write, ";\n");

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

    fprintf(file_write, "TRACKS %s %d DO %d STEP %d ", direction.c_str(), track->get_track()->get_start(), track->get_track_num(),
            track->get_track()->get_pitch());

    fprintf(file_write, "LAYER ");

    for (IdbLayer* layer : track->get_layer_list()) {
      fprintf(file_write, "%s ", layer->get_name().c_str());
    }

    fprintf(file_write, ";\n \n");
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

  fprintf(file_write, "VIAS %ld ;\n", via_list->get_num_via());

  for (IdbVia* via : via_list->get_via_list()) {
    IdbViaMaster* via_master = via->get_instance();

    if (via_master->is_generate()) {
      IdbViaMasterGenerate* master_generate = via_master->get_master_generate();

      fprintf(file_write,
              "- %s + VIARULE %s + CUTSIZE %d %d + LAYERS %s %s %s + CUTSPACING %d %d + ENCLOSURE %d %d %d %d "
              " + ROWCOL %d %d \n",
              via->get_name().c_str(), master_generate->get_rule_name().c_str(), master_generate->get_cut_size_x(),
              master_generate->get_cut_size_y(), master_generate->get_layer_bottom()->get_name().c_str(),
              master_generate->get_layer_cut()->get_name().c_str(), master_generate->get_layer_top()->get_name().c_str(),
              master_generate->get_cut_spcing_x(), master_generate->get_cut_spcing_y(), master_generate->get_enclosure_bottom_x(),
              master_generate->get_enclosure_bottom_y(), master_generate->get_enclosure_top_x(), master_generate->get_enclosure_top_y(),
              master_generate->get_cut_rows(), master_generate->get_cut_cols());

      if (nullptr != master_generate->get_patttern()) {
        fprintf(file_write, " + PATTERN %s \n", master_generate->get_patttern()->get_pattern_string().c_str());
      }

      fprintf(file_write, " ;\n");
    }
  }

  fprintf(file_write, "END VIAS\n \n");

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
    fprintf(file_write, "ROW %s %s %d %d %s DO %d BY %d STEP %d %d ;\n", row->get_name().c_str(), row->get_site()->get_name().c_str(),
            row->get_original_coordinate()->get_x(), row->get_original_coordinate()->get_y(), site_orient.c_str(), row->get_row_num_x(),
            row->get_row_num_y(), row->get_step_x(), row->get_step_y());
  }

  fprintf(file_write, " \n");

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

  fprintf(file_write, "COMPONENTS %d ;\n", instance_list->get_num());

  for (IdbInstance* instance : instance_list->get_instance_list()) {
    string type = instance->get_type() != IdbInstanceType::kNone
                      ? "+ SOURCE " + IdbEnum::GetInstance()->get_instance_property()->get_type_str(instance->get_type())
                      : "";
    string status = IdbEnum::GetInstance()->get_instance_property()->get_status_str(instance->get_status());
    string orient = IdbEnum::GetInstance()->get_site_property()->get_orient_name(instance->get_orient());

    if (instance->has_placed()) {
      fprintf(file_write, "    - %s %s %s + %s ( %d %d ) %s \n", instance->get_name().c_str(),
              instance->get_cell_master()->get_name().c_str(), type.c_str(), status.c_str(), instance->get_coordinate()->get_x(),
              instance->get_coordinate()->get_y(), orient.c_str());
    } else {
      fprintf(file_write, "    - %s %s %s \n", instance->get_name().c_str(), instance->get_cell_master()->get_name().c_str(), type.c_str());
    }

    /// halo
    auto halo = instance->get_halo();
    if (halo != nullptr) {
      std::string str_soft = halo->is_soft() ? " [SOFT] " : " ";
      fprintf(file_write, "      + HALO%s%d %d %d %d\n", str_soft.c_str(), halo->get_extend_lef(), halo->get_extend_bottom(),
              halo->get_extend_right(), halo->get_extend_top());
    }

    /// routed halo
    auto route_halo = instance->get_route_halo();
    if (route_halo != nullptr) {
      fprintf(file_write, "      + ROUTEHALO %d %s %s\n", route_halo->get_route_distance(),
              route_halo->get_layer_bottom()->get_name().c_str(), route_halo->get_layer_top()->get_name().c_str());
    }

    fprintf(file_write, "      ;\n");
  }

  fprintf(file_write, "END COMPONENTS\n \n");

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

  fprintf(file_write, "PINS %d ;\n", pin_list->get_pin_num());

  for (IdbPin* pin : pin_list->get_pin_list()) {
    string direction = IdbEnum::GetInstance()->get_connect_property()->get_direction_name(pin->get_term()->get_direction());
    string use = IdbEnum::GetInstance()->get_connect_property()->get_type_name(pin->get_term()->get_type());
    string is_special = pin->is_special_net_pin() || pin->get_term()->is_special_net() ? "+ SPECIAL " : "";

    fprintf(file_write, " - %s + NET %s %s+ DIRECTION %s", pin->get_pin_name().c_str(), pin->get_net_name().c_str(), is_special.c_str(),
            direction.c_str());

    if (use.empty()) {
      fprintf(file_write, "  \n");
    } else {
      fprintf(file_write, "  + USE %s\n", use.c_str());
    }

    if (pin->get_term()->is_port_exist() || pin->is_special_net_pin()) {
      for (IdbPort* port : pin->get_term()->get_port_list()) {
        fprintf(file_write, "  + PORT\n");

        string status = IdbEnum::GetInstance()->get_instance_property()->get_status_str(port->get_placement_status());
        string orient = IdbEnum::GetInstance()->get_site_property()->get_orient_name(port->get_orient());
        for (IdbLayerShape* layer_shape : port->get_layer_shape()) {
          fprintf(file_write, "   + LAYER %s ", layer_shape->get_layer()->get_name().c_str());
          for (IdbRect* rect : layer_shape->get_rect_list()) {
            fprintf(file_write, "( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
          }

          if (port->is_placed()) {
            fprintf(file_write, "+ %s ( %d %d ) %s", status.c_str(), port->get_coordinate()->get_x(), port->get_coordinate()->get_y(),
                    orient.c_str());
          }
          fprintf(file_write, "\n");
        }
      }
    } else {
      string status = IdbEnum::GetInstance()->get_instance_property()->get_status_str(pin->get_term()->get_placement_status());
      string orient = IdbEnum::GetInstance()->get_site_property()->get_orient_name(pin->get_orient());
      for (IdbPort* port : pin->get_term()->get_port_list()) {
        for (IdbLayerShape* layer_shape : port->get_layer_shape()) {
          fprintf(file_write, " + LAYER %s ", layer_shape->get_layer()->get_name().c_str());
          for (IdbRect* rect : layer_shape->get_rect_list()) {
            fprintf(file_write, "( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
          }

          if (pin->get_term()->is_placed()) {
            fprintf(file_write, "+ %s ( %d %d ) %s", status.c_str(), pin->get_location()->get_x(), pin->get_location()->get_y(),
                    orient.c_str());
          }
        }
      }
      fprintf(file_write, "\n");
    }

    fprintf(file_write, ";\n");
  }

  fprintf(file_write, "END PINS\n \n");

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

  fprintf(file_write, "BLOCKAGES %d ;\n", blockage_list->get_num());

  for (IdbBlockage* blockage : blockage_list->get_blockage_list()) {
    if (blockage->get_type() == IdbBlockage::IdbBlockageType::kRoutingBlockage) {
      IdbRoutingBlockage* routing_blockage = dynamic_cast<IdbRoutingBlockage*>(blockage);

      fprintf(file_write, "    - LAYER %s ", routing_blockage->get_layer_name().c_str());

      if (routing_blockage->is_pushdown() == true) {
        fprintf(file_write, "+ PUSHDOWN ");
      }

      if (routing_blockage->is_except_pgnet() == true) {
        fprintf(file_write, "+ EXCEPTPGNET ");
      }

      if (routing_blockage->get_instance() != nullptr) {
        fprintf(file_write, "+ COMPONENT %s ", routing_blockage->get_instance_name().c_str());
      }

      for (IdbRect* rect : routing_blockage->get_rect_list()) {
        fprintf(file_write, "RECT ( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
      }
    } else if (blockage->get_type() == IdbBlockage::IdbBlockageType::kPlacementBlockage) {
      IdbPlacementBlockage* placement_blockage = dynamic_cast<IdbPlacementBlockage*>(blockage);

      fprintf(file_write, "    - PLACEMENT ");

      if (placement_blockage->is_pushdown() == true) {
        fprintf(file_write, "+ PUSHDOWN ");
      }

      if (placement_blockage->get_instance() != nullptr) {
        fprintf(file_write, "+ COMPONENT %s ", placement_blockage->get_instance_name().c_str());
      }

      for (IdbRect* rect : placement_blockage->get_rect_list()) {
        fprintf(file_write, "RECT ( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
      }
    }

    fprintf(file_write, ";\n");
  }

  fprintf(file_write, "END BLOCKAGES\n \n");

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
    fprintf(file_write, " %s%s %d %s ( %d %d ) ( * %d )\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
            segment->get_route_width(), shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(),
            segment->get_point_second()->get_y());
  } else if (segment->get_point_start()->get_y() == segment->get_point_second()->get_y()) {
    fprintf(file_write, " %s%s %d %s ( %d %d ) ( %d * )\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
            segment->get_route_width(), shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(),
            segment->get_point_second()->get_x());
  } else {
    fprintf(file_write, " %s%s %d %s ( %d %d ) ( %d %d )\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
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
      fprintf(file_write, " %s%s %d %s ( %d %d ) ( * %d ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
              segment->get_route_width(), shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(),
              segment->get_point_second()->get_y(), segment->get_via()->get_name().c_str());
    } else if (segment->get_point_start()->get_y() == segment->get_point_second()->get_y()) {
      fprintf(file_write, " %s%s %d %s ( %d %d ) ( %d * ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
              segment->get_route_width(), shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(),
              segment->get_point_second()->get_x(), segment->get_via()->get_name().c_str());
    } else {
      fprintf(file_write, " %s%s %d %s ( %d %d ) ( %d %d ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
              segment->get_route_width(), shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(),
              segment->get_point_second()->get_x(), segment->get_point_second()->get_y(), segment->get_via()->get_name().c_str());
    }
  } else {
    fprintf(file_write, " %s%s %d %s ( %d %d ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
            segment->get_route_width(), shape.c_str(), segment->get_point_start()->get_x(), segment->get_point_start()->get_y(),
            segment->get_via()->get_name().c_str());
  }

  return kDbSuccess;
}

int32_t DefWrite::write_specialnet_wire_segment(IdbSpecialWireSegment* segment, string& wire_new_str)
{
  if (segment->is_via()) {
    return write_specialnet_wire_segment_via(segment, wire_new_str);
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

  fprintf(file_write, "SPECIALNETS %ld ;\n", special_net_list->get_num());

  for (IdbSpecialNet* special_net : special_net_list->get_net_list()) {
    fprintf(file_write, "- %s ", special_net->get_net_name().c_str());

    if (special_net->get_pin_string_list().size() > 0) {
      for (string pin_string : special_net->get_pin_string_list()) {
        fprintf(file_write, "( * %s ) ", pin_string.c_str());
      }
    } else {
      for (IdbPin* pin_io : special_net->get_io_pin_list()->get_pin_list()) {
        fprintf(file_write, "( PIN %s ) ", pin_io->get_pin_name().c_str());
      }

      for (IdbPin* pin_instance : special_net->get_instance_pin_list()->get_pin_list()) {
        fprintf(file_write, "( %s %s ) ", pin_instance->get_instance()->get_name().c_str(), pin_instance->get_pin_name().c_str());
      }
    }

    fprintf(file_write, "\n");

    string connect_str = IdbEnum::GetInstance()->get_connect_property()->get_type_name(special_net->get_connect_type());
    fprintf(file_write, "  + USE %s \n", connect_str.c_str());

    for (IdbSpecialWire* wire : special_net->get_wire_list()->get_wire_list()) {
      write_specialnet_wire(wire);
    }

    fprintf(file_write, " ;\n");
  }

  fprintf(file_write, "END SPECIALNETS\n \n");

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

  fprintf(file_write, "NETS %ld ;\n", net_list->get_num());

  for (IdbNet* net : net_list->get_net_list()) {
    // std::string net_name = net->get_net_name();
    // std::string net_name_new = ieda::Str::addBackslash(net_name);
    fprintf(file_write, "- %s", net->get_net_name().c_str());

    if (net->get_io_pin() != nullptr) {
      fprintf(file_write, " ( PIN %s )", net->get_io_pin()->get_pin_name().c_str());
    }

    for (IdbPin* instance : net->get_instance_pin_list()->get_pin_list()) {
      fprintf(file_write, " ( %s %s )", instance->get_instance()->get_name().c_str(), instance->get_pin_name().c_str());
    }

    fprintf(file_write, "\n");

    if (IdbConnectType::kNone < net->get_connect_type() && IdbConnectType::kMax > net->get_connect_type()) {
      string use = IdbEnum::GetInstance()->get_connect_property()->get_type_name(net->get_connect_type());
      fprintf(file_write, "  + USE %s \n", use.c_str());
    } else {
    }

    if (net->get_wire_list()->get_num() > 0) {
      for (IdbRegularWire* wire : net->get_wire_list()->get_wire_list()) {
        write_net_wire(wire);
      }
    }

    fprintf(file_write, " ;\n");
  }

  fprintf(file_write, "END NETS\n \n");

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
    fprintf(file_write, "%s %s ( %d %d ) %s( * %d )\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
            segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), virtual_str, segment->get_point_second()->get_y());
  } else if (segment->get_point_start()->get_y() == segment->get_point_second()->get_y()) {
    fprintf(file_write, "%s %s ( %d %d ) %s( %d * )\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
            segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), virtual_str, segment->get_point_second()->get_x());
  } else {
    fprintf(file_write, "%s %s ( %d %d ) %s( %d %d )\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
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
      fprintf(file_write, "%s %s ( %d %d ) ( * %d ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
              segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), segment->get_point_second()->get_y(),
              segment->get_via_list().at(_POINT_START_)->get_name().c_str());
    } else if (segment->get_point_start()->get_y() == segment->get_point_second()->get_y()) {
      fprintf(file_write, "%s %s ( %d %d ) ( %d * ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
              segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), segment->get_point_second()->get_x(),
              segment->get_via_list().at(_POINT_START_)->get_name().c_str());
    } else {
      fprintf(file_write, "%s %s ( %d %d ) ( %d %d ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
              segment->get_point_start()->get_x(), segment->get_point_start()->get_y(), segment->get_point_second()->get_x(),
              segment->get_point_second()->get_y(), segment->get_via_list().at(0)->get_name().c_str());
    }
  } else {
    fprintf(file_write, "%s %s ( %d %d ) %s\n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
            segment->get_point_start()->get_x(), segment->get_point_start()->get_y(),
            segment->get_via_list().at(_POINT_START_)->get_name().c_str());
  }

  return kDbSuccess;
}

int32_t DefWrite::write_net_wire_segment_rect(IdbRegularWireSegment* segment, string& wire_new_str)
{
  if (segment->get_point_list().size() <= 0 || segment->get_layer() == nullptr || segment->get_delta_rect() == nullptr) {
    std::cout << "Error net wire segment rect..." << std::endl;
    return kDbFail;
  }

  fprintf(file_write, "%s %s ( %d %d ) RECT ( %d %d %d %d ) \n", wire_new_str.c_str(), segment->get_layer()->get_name().c_str(),
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

  //   fprintf(file_write, "GCELLGRID\n");

  for (IdbGCellGrid* gcell_grid : gcell_grid_list->get_gcell_grid_list()) {
    string direction_str = gcell_grid->get_direction() == IdbTrackDirection::kDirectionX ? "X" : "Y";

    fprintf(file_write, "GCELLGRID %s %d DO %d STEP %d ;\n", direction_str.c_str(), gcell_grid->get_start(), gcell_grid->get_num(),
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

  fprintf(file_write, "REGIONS %d ;\n", region_list->get_num());

  for (IdbRegion* region : region_list->get_region_list()) {
    fprintf(file_write, "    - %s ", region->get_name().c_str());

    for (IdbRect* rect : region->get_boundary()) {
      fprintf(file_write, "( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
    }

    string type = IdbEnum::GetInstance()->get_region_property()->get_name(region->get_type());
    fprintf(file_write, "+ TYPE %s ", type.c_str());

    fprintf(file_write, ";\n");
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

  fprintf(file_write, "SLOTS %d ;\n", slot_list->get_num());

  for (IdbSlot* slot : slot_list->get_slot_list()) {
    fprintf(file_write, "    - LAYER %s ", slot->get_layer_name().c_str());

    for (IdbRect* rect : slot->get_rect_list()) {
      fprintf(file_write, "RECT ( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
    }

    fprintf(file_write, ";\n");
  }

  fprintf(file_write, "END SLOTS\n");

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

  fprintf(file_write, "GROUPS %d ;\n", group_list->get_num());

  for (IdbGroup* group : group_list->get_group_list()) {
    fprintf(file_write, "    - %s ", group->get_group_name().c_str());

    for (IdbInstance* instance : group->get_instance_list()->get_instance_list()) {
      fprintf(file_write, "%s ", instance->get_name().c_str());
    }

    fprintf(file_write, "+ REGION %s ", group->get_region()->get_name().c_str());

    fprintf(file_write, ";\n");
  }

  fprintf(file_write, "END GROUPS\n");

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

  fprintf(file_write, "FILLS %d ;\n", fill_list->get_num_fill());

  for (IdbFill* fill : fill_list->get_fill_list()) {
    fprintf(file_write, "    - LAYER %s ", fill->get_layer()->get_layer()->get_name().c_str());

    for (IdbRect* rect : fill->get_layer()->get_rect_list()) {
      fprintf(file_write, "RECT ( %d %d ) ( %d %d ) ", rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
    }

    fprintf(file_write, ";\n");

    fprintf(file_write, "    - VIA %s ", fill->get_via()->get_via()->get_name().c_str());

    for (IdbCoordinate<int32_t>* point : fill->get_via()->get_coordinate_list()) {
      fprintf(file_write, "( %d %d ) ", point->get_x(), point->get_y());
    }

    fprintf(file_write, ";\n");
  }

  cout << "Write FILLS success..." << endl;
  return kDbSuccess;
}

}  // namespace idb
