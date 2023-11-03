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

#include "json_write.h"
#include <fstream>
#include <string>
#include <sstream>
#include "../../../data/design/IdbDesign.h"
#include "omp.h"

vector<string> discard;
namespace idb {

Gds2JsonWrite::Gds2JsonWrite(IdbDefService* def_service)
{
  _def_service = def_service;
  file_write = nullptr;
}

Gds2JsonWrite::~Gds2JsonWrite()
{
}

void set_discard(string option){
  
  std::istringstream iss(option);
  std::string element;
  
  while (std::getline(iss, element, '/')) {
      discard.push_back(element);
  }
  
  // for (const auto& ele : discard) {
  //     std::cout << ele << std::endl;
  // }
}

bool Gds2JsonWrite::writeDb(const char* file,string option)
{
  _writer.init(file, &_json);

  set_units();

  set_discard(option);

  _writer.begin(0);

  writeChip();

  return _writer.finish(0,discard);
}

bool Gds2JsonWrite::writeChip()
{
  write_version();
  write_design();
  write_die();
  write_layer();
  // write_layer();
  //   write_blockage();
  write_pin();
  write_component();
  write_fill();
  write_special_net();
  write_net();

  /// no need to print
  //   write_row();
  //   write_track_grid();
  //   write_gcell_grid();
  //   write_via();
  //   write_region();
  //   write_slot();
  //   write_group();
  return true;
}

std::map<string, int> activated_layers;

void Gds2JsonWrite::activate_layer(IdbLayers* layer_list){

  for (auto element : discard) {
    if(element == "met"){
      for (unsigned int i = 0; i < layer_list->get_layers().size(); i++) {
        if (layer_list->get_layers()[i]->get_name().find("met") == 0)
          activated_layers[layer_list->get_layers()[i]->get_name()] = i;
      }
    }
    else if(element == "via"){
      for (unsigned int i = 0; i < layer_list->get_layers().size(); i++) {
        if (layer_list->get_layers()[i]->get_name().find("via") == 0)
          activated_layers[layer_list->get_layers()[i]->get_name()] = i;
      }
    }
    else if(element == "li"){
      for (unsigned int i = 0; i < layer_list->get_layers().size(); i++) {
        if (layer_list->get_layers()[i]->get_name().find("li") == 0)
          activated_layers[layer_list->get_layers()[i]->get_name()] = i;
      }
    }
    else if(element == "mcon"){
      for (unsigned int i = 0; i < layer_list->get_layers().size(); i++) {
        if (layer_list->get_layers()[i]->get_name().find("mcon") == 0)
          activated_layers[layer_list->get_layers()[i]->get_name()] = i;
      }
    }
    else if(element == "nwell"){
      for (unsigned int i = 0; i < layer_list->get_layers().size(); i++) {
        if (layer_list->get_layers()[i]->get_name().find("nwell") == 0)
          activated_layers[layer_list->get_layers()[i]->get_name()] = i;
      }
    }
    else if(element == "pwell"){
      for (unsigned int i = 0; i < layer_list->get_layers().size(); i++) {
        if (layer_list->get_layers()[i]->get_name().find("pwell") == 0)
          activated_layers[layer_list->get_layers()[i]->get_name()] = i;
      }
    }
    else if(element == "null"){
      activated_layers["null"] = 0;
    }
  }
}

bool Gds2JsonWrite::write_layer()
{
  unsigned int i=0;
  IdbLayout* layout = _def_service->get_layout();
  auto layer_list = layout->get_layers();
  vector<string> layer_name;
  
  activate_layer(layer_list);
  
  for (; i < layer_list->get_layers().size(); i++) {
    IdbLayer* lay = (layer_list->get_layers())[i];
    // if(activated_layers.find(lay->get_name())!=activated_layers.end()){
      layer_name.push_back(lay->get_name());
      // num++;
    // }
  }
  _writer.write_layerinfo(layer_name, 1,i); 
  return true;
}

void Gds2JsonWrite::addStruct(JsonStruct* json_struct)
{
  _json.add_struct(json_struct);

  if (_json.is_full()) {
    _writer.writeStruct(1);
  }
}

void Gds2JsonWrite::writeStruct()
{
  _writer.writeStruct(1);
}

void Gds2JsonWrite::write_diearea()
{
  _writer.write_diearea(1);
}

int32_t Gds2JsonWrite::set_units()
{
  IdbDesign* design = _def_service->get_design();
  IdbUnits* def_units = design->get_units();
  IdbUnits* lef_units = design->get_layout()->get_units();
  if (def_units == nullptr && lef_units == nullptr) {
    std::cout << "Write UNITS failed..." << std::endl;

    return kDbFail;
  }

  _unit_microns = def_units->get_micron_dbu() > 0 ? def_units->get_micron_dbu() : lef_units->get_micron_dbu();
  if (_unit_microns <= 0) {
    std::cout << "Write UNITS failed..." << std::endl;

    return kDbFail;
  }

  _json.set_unit(1.0 / _unit_microns, 1.0 * 1e-6 / _unit_microns);

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_version()
{
  IdbDesign* design = _def_service->get_design();
  /// support version 5.8
  string version = design->get_version().empty() ? "5.8" : design->get_version();

  /// @brief json format
  JsonStruct* json_struct = new JsonStruct("VERSION");

  JsonText json_text;

  json_text.str = version;
  json_text.width = 2;
  json_text.presentation = JsonPresentation::kCenter;
  json_text.add_coord(0, 0);

  json_struct->add_element(json_text);

  /// @brief add to json
  addStruct(json_struct);

  writeStruct();

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_design()
{
  IdbDesign* design = _def_service->get_design();
  string design_name = design->get_design_name();

  /// @brief json format
  JsonStruct* json_struct = new JsonStruct("Design Name");

  JsonText json_text;

  json_text.str = design_name;
  json_text.width = 2;
  json_text.presentation = JsonPresentation::kCenter;
  json_text.add_coord(0, -5);

  json_struct->add_element(json_text);

  /// @brief add to json
  addStruct(json_struct);
  writeStruct();

  return kDbSuccess;
}
// int32_t Gds2JsonWrite::write_layer()
// {
//   IdbDesign* design = _def_service->get_design();
//   string design_name = design->get_design_name();

//   /// @brief json format
//   JsonStruct* json_struct = new JsonStruct("Design Name");

//   JsonText json_text;

//   json_text.str = design_name;
//   json_text.width = 2;
//   json_text.presentation = JsonPresentation::kCenter;
//   json_text.add_coord(0, -5);

//   json_struct->add_element(json_text);

//   /// @brief add to json
//   addStruct(json_struct);
//   writeStruct();

//   return kDbSuccess;
// }
/**
 * @brief write die as the top struct
 *
 * @return int32_t
 */
int32_t Gds2JsonWrite::write_die()
{
  IdbLayout* layout = _def_service->get_layout();
  IdbDie* die = layout->get_die();
  if (die == nullptr) {
    std::cout << "Write DIE failed..." << std::endl;

    return kDbFail;
  }

  //   /// @brief json format
  _top_struct = new JsonStruct("DIEAREA");

  JsonPath* path = new JsonPath();
  path->layer = 0;
  path->data_type = 2;
  path->path_type = JsonPathType::kRoundEnd;
  path->width = 2;

  path->add_coord(transDB2Unit(die->get_llx()), transDB2Unit(die->get_lly()));
  path->add_coord(transDB2Unit(die->get_urx()), transDB2Unit(die->get_lly()));
  path->add_coord(transDB2Unit(die->get_urx()), transDB2Unit(die->get_ury()));
  path->add_coord(transDB2Unit(die->get_llx()), transDB2Unit(die->get_ury()));
  path->add_coord(transDB2Unit(die->get_llx()), transDB2Unit(die->get_lly()));

  _top_struct->add_element(path);

  /// @brief set top struct
  _json.set_top_struct(_top_struct);

  //   /// @brief write top struct
  //   _writer.writeTopStruct();
  write_diearea();

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_track_grid()
{
  IdbLayout* layout = _def_service->get_layout();
  IdbTrackGridList* track_grid_list = layout->get_track_grid_list();
  if (track_grid_list == nullptr) {
    std::cout << "Write Track Grid failed..." << std::endl;
    return kDbFail;
  }

  int32_t width = transDB2Unit(layout->get_die()->get_width());
  int32_t height = transDB2Unit(layout->get_die()->get_height());

  /// @brief json format
  JsonStruct* json_struct = new JsonStruct("TRACKS");

  for (IdbTrackGrid* track_grid : track_grid_list->get_track_grid_list()) {
    IdbTrack* track = track_grid->get_track();
    int32_t start = track->get_start();
    int32_t pitch = track->get_pitch();

    for (IdbLayer* layer : track_grid->get_layer_list()) {
      if (track->is_track_direction_y()) {
        for (uint i = 0; i < track_grid->get_track_num(); ++i) {
          int32_t y = start + pitch * i;

          JsonPath* path = new JsonPath();
          path->layer = layer->get_order();
          path->data_type = 2;
          path->path_type = JsonPathType::kRoundEnd;
          path->width = 1;
          path->add_coord(0, transDB2Unit(y));
          path->add_coord(width, transDB2Unit(y));

          json_struct->add_element(path);
        }
      } else {
        for (uint i = 0; i < track_grid->get_track_num(); ++i) {
          int32_t x = start + pitch * i;

          JsonPath* path = new JsonPath();
          path->layer = layer->get_order();
          path->data_type = 2;
          path->path_type = JsonPathType::kRoundEnd;
          path->width = 1;
          path->add_coord(transDB2Unit(x), 0);
          path->add_coord(transDB2Unit(x), height);

          json_struct->add_element(path);
        }
      }
    }

    /// @brief add to json
    addStruct(json_struct);
  }

  writeStruct();

  return kDbSuccess;
}

void Gds2JsonWrite::packLayerShape(JsonStruct* json_struct, IdbLayerShape* layer_shape)
{
  for (auto rect : layer_shape->get_rect_list()) {
    packRect(json_struct, rect, layer_shape->get_layer());
  }
}

void Gds2JsonWrite::packRect(JsonStruct* json_struct, IdbRect* rect, int32_t layer_id)
{
  JsonBox box;
  box.add_coord(transDB2Unit(rect->get_low_x()), transDB2Unit(rect->get_low_y()));
  box.add_coord(transDB2Unit(rect->get_high_x()), transDB2Unit(rect->get_low_y()));
  box.add_coord(transDB2Unit(rect->get_high_x()), transDB2Unit(rect->get_high_y()));
  box.add_coord(transDB2Unit(rect->get_low_x()), transDB2Unit(rect->get_high_y()));
  box.add_coord(transDB2Unit(rect->get_low_x()), transDB2Unit(rect->get_low_y()));
  box.layer = layer_id;
  json_struct->add_element(box);
}

void Gds2JsonWrite::packRect(JsonStruct* json_struct, IdbRect* rect, IdbLayer* layer)
{
  IdbLayout* layout = _def_service->get_layout();
  auto layer_list = layout->get_layers();
  int32_t order = layer == nullptr ? layer_list->get_bottom_routing_layer()->get_order() : layer->get_order();
  packRect(json_struct, rect, order);

}

void Gds2JsonWrite::packRect(JsonStruct* json_struct, int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y, IdbLayer* layer)
{
  IdbLayout* layout = _def_service->get_layout();
  auto layer_list = layout->get_layers();

  JsonBox box;
  box.add_coord(transDB2Unit(ll_x), transDB2Unit(ll_y));
  box.add_coord(transDB2Unit(ur_x), transDB2Unit(ll_y));
  box.add_coord(transDB2Unit(ur_x), transDB2Unit(ur_y));
  box.add_coord(transDB2Unit(ll_x), transDB2Unit(ur_y));
  box.add_coord(transDB2Unit(ll_x), transDB2Unit(ll_y));
  box.layer = (layer == nullptr ? layer_list->get_bottom_routing_layer()->get_order() : layer->get_order());
  json_struct->add_element(box);
}

void Gds2JsonWrite::packVia(JsonStruct* json_struct, IdbVia* via)
{
  /// top
  auto top_layer_shape = via->get_top_layer_shape();
  packLayerShape(json_struct, &top_layer_shape);

  /// cut
  auto cut_layer_shape = via->get_cut_layer_shape();
  packLayerShape(json_struct, &cut_layer_shape);

  /// bottom
  auto bottom_layer_shape = via->get_bottom_layer_shape();
  packLayerShape(json_struct, &bottom_layer_shape);
}

void Gds2JsonWrite::packPin(JsonStruct* json_struct, IdbPin* pin)
{
  IdbTerm* term = pin->get_term();
  if (term->is_port_exist()) {
    /// there are "port" key word
    for (auto layer_shape : pin->get_port_box_list()) {
      packLayerShape(json_struct, layer_shape);
    }

    for (auto via : pin->get_via_list()) {
      packVia(json_struct, via);
    }
  }
}

/// @brief transfer segment of 2 points data to json format
/// @param json_struct
/// @param routing_layer
/// @param point_1
/// @param point_2
void Gds2JsonWrite::packSegment(JsonStruct* json_struct, IdbLayerRouting* routing_layer, IdbCoordinate<int32_t>* point_1,
                               IdbCoordinate<int32_t>* point_2, int32_t width)
{
  int32_t routing_width = width > 0 ? width : routing_layer->get_width();

  int32_t ll_x = 0;
  int32_t ll_y = 0;
  int32_t ur_x = 0;
  int32_t ur_y = 0;
  if (point_1->get_y() == point_2->get_y()) {
    // horizontal
    ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width / 2;
    ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width / 2;
    ur_x = std::max(point_1->get_x(), point_2->get_x()) + routing_width / 2;
    ur_y = ll_y + routing_width;
  } else if (point_1->get_x() == point_2->get_x()) {
    // vertical
    ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width / 2;
    ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width / 2;
    ur_x = ll_x + routing_width;
    ur_y = std::max(point_1->get_y(), point_2->get_y()) + routing_width / 2;
  } else {
    // only support horizontal & vertical direction
    std::cout << "Error...Regular segment only support horizontal & "
                 "vertical direction... "
              << std::endl;
  }

  packRect(json_struct, ll_x, ll_y, ur_x, ur_y, routing_layer);
}

int32_t Gds2JsonWrite::write_via()
{
  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_row()
{
  IdbLayout* layout = _def_service->get_layout();
  IdbRows* rows = layout->get_rows();
  if (rows == nullptr) {
    std::cout << "Write ROWS failed..." << std::endl;
    return kDbFail;
  }

  addSRefDefault("Rows");
  JsonStruct* json_struct = new JsonStruct("Rows");
  for (IdbRow* row : rows->get_row_list()) {
    packRect(json_struct, row->get_bounding_box(), 0);
  }

  addStruct(json_struct);
  writeStruct();

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_component()
{
  static int num=0;
  IdbDesign* design = _def_service->get_design();  // Def
  IdbInstanceList* instance_list = design->get_instance_list();
  if (instance_list == nullptr || instance_list->get_num() == 0) {
    std::cout << "Write COMPONENTS failed..." << std::endl;
    return kDbFail;
  }

  int x = 0;
  int max_num = instance_list->get_num();

  omp_lock_t lck;
  omp_init_lock(&lck);
// #pragma omp parallel for schedule(dynamic)

  for (IdbInstance* instance : instance_list->get_instance_list()) {
    string name = "Instance_" + instance->get_name();

    //support sref expension
    srefHash.emplace(name,num++);

    omp_set_lock(&lck);

    addSRefDefault(name);

    omp_unset_lock(&lck);

    JsonStruct* json_struct = new JsonStruct(name);

    /// instance boundingbox
    packRect(json_struct, instance->get_bounding_box(), 0);

    /// pins
    for (auto pin : instance->get_pin_list()->get_pin_list()) {
      packPin(json_struct, pin);
    }

    /// obs
 for (auto obs_shape : instance->get_obs_box_list()) {
      packLayerShape(json_struct, obs_shape);  
  }
    omp_set_lock(&lck);

    /// @brief add to json
    addStruct(json_struct);
    

    omp_unset_lock(&lck);

    x++;
    if (x % 1000 == 0) {
      std::cout << "Write COMPONENTS. " << x << " / " << max_num << std::endl;
    }
  }

  omp_destroy_lock(&lck);

  writeStruct();

  std::cout << "Write COMPONENTS success. " << max_num << " / " << max_num << std::endl;

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_pin()
{
  IdbDesign* design = _def_service->get_design();
  IdbPins* pin_list = design->get_io_pin_list();
  if (pin_list == nullptr) {
    std::cout << "Write PINS failed..." << std::endl;
    return kDbFail;
  }

  /// @brief json format
  JsonStruct* json_struct = new JsonStruct("PINS");

  for (IdbPin* pin : pin_list->get_pin_list()) {
    packPin(json_struct, pin);
  }

  /// @brief add to json
  addStruct(json_struct);

  writeStruct();

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_blockage()
{
  IdbDesign* design = _def_service->get_design();
  IdbBlockageList* blockage_list = design->get_blockage_list();
  if (blockage_list == nullptr || blockage_list->get_num() == 0) {
    std::cout << "Write blocakge failed..." << std::endl;
    return kDbFail;
  }

  for (IdbBlockage* blockage : blockage_list->get_blockage_list()) {
    /// @brief json format
    string name = "Blockage_" + blockage->get_instance_name();
    addSRefDefault(name);
    JsonStruct* json_struct = new JsonStruct(name);

    if (blockage->is_palcement_blockage()) {
      for (auto idb_rect : blockage->get_rect_list()) {
        packRect(json_struct, idb_rect, ((IdbPlacementBlockage*) blockage)->get_layer());
      }
    } else {
      for (auto idb_rect : blockage->get_rect_list()) {
        packRect(json_struct, idb_rect, ((IdbRoutingBlockage*) blockage)->get_layer());
      }
    }

    addStruct(json_struct);
  }

  writeStruct();

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_specialnet_wire_segment_points(JsonStruct* json_struct, IdbSpecialWireSegment* segment)
{
  if (segment->get_point_list().size() < _POINT_MAX_) {
    std::cout << "Specialnet wire points are less than 2..." << std::endl;
    return kDbFail;
  }

  if (segment->get_point_num() >= _POINT_MAX_) {
    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(segment->get_layer());
    int32_t routing_width = segment->get_route_width() == 0 ? routing_layer->get_width() : segment->get_route_width();

    IdbCoordinate<int32_t>* point_1 = segment->get_point_start();
    IdbCoordinate<int32_t>* point_2 = segment->get_point_second();

    packSegment(json_struct, routing_layer, point_1, point_2, routing_width);
  }

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_specialnet_wire_segment_via(JsonStruct* json_struct, IdbSpecialWireSegment* segment)
{
  if (segment->get_point_list().size() <= 0 || segment->get_layer() == nullptr || segment->get_via() == nullptr) {
    std::cout << "No net wire segment via..." << std::endl;
    return kDbFail;
  }

  packVia(json_struct, segment->get_via());

  if (segment->get_point_list().size() >= _POINT_MAX_) {
    return write_specialnet_wire_segment_points(json_struct, segment);
  }

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_specialnet_wire_segment(JsonStruct* json_struct, IdbSpecialWireSegment* segment)
{
  if (segment->is_via()) {
    return write_specialnet_wire_segment_via(json_struct, segment);
  } else {
    return write_specialnet_wire_segment_points(json_struct, segment);
  }

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_specialnet_wire(JsonStruct* json_struct, IdbSpecialWire* wire)
{
  for (IdbSpecialWireSegment* segment : wire->get_segment_list()) {
    write_specialnet_wire_segment(json_struct, segment);
  }

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_special_net()
{
  IdbSpecialNetList* special_net_list = _def_service->get_design()->get_special_net_list();
  if (special_net_list == nullptr || special_net_list->get_num() == 0) {
    std::cout << "No SPECIALNETS..." << std::endl;
    return kDbFail;
  }

  omp_lock_t lck;
  omp_init_lock(&lck);

// #pragma omp parallel for schedule(dynamic)

  for (IdbSpecialNet* special_net : special_net_list->get_net_list()) {
    omp_set_lock(&lck);
    /// @brief json format
    addSRefDefault(special_net->get_net_name());
    omp_unset_lock(&lck);

    JsonStruct* json_struct = new JsonStruct(special_net->get_net_name());

    for (IdbSpecialWire* wire : special_net->get_wire_list()->get_wire_list()) {
      write_specialnet_wire(json_struct, wire);
    }

    omp_set_lock(&lck);

    addStruct(json_struct);

    omp_unset_lock(&lck);
  }

  omp_destroy_lock(&lck);

  writeStruct();

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_net()
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

  int x = 0;
  int max_num = net_list->get_num();

  omp_lock_t lck;
  omp_init_lock(&lck);

// #pragma omp parallel for schedule(dynamic)

  for (IdbNet* net : net_list->get_net_list()) {
    omp_set_lock(&lck);
    /// @brief json format
    addSRefDefault(net->get_net_name());
    omp_unset_lock(&lck);

    JsonStruct* json_struct = new JsonStruct(net->get_net_name());

    if (net->get_wire_list()->get_num() > 0) {
      for (IdbRegularWire* wire : net->get_wire_list()->get_wire_list()) {
        write_net_wire(json_struct, wire);
      }
    }

    omp_set_lock(&lck);

    addStruct(json_struct);

    omp_unset_lock(&lck);

    x++;
    if (x % 1000 == 0) {
      std::cout << "Write NETS. " << x << " / " << max_num << std::endl;
    }
  }

  omp_destroy_lock(&lck);

  writeStruct();
  std::cout << "Write NETS success. " << max_num << " / " << max_num << std::endl;

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_net_wire(JsonStruct* json_struct, IdbRegularWire* wire)
{
  for (IdbRegularWireSegment* segment : wire->get_segment_list()) {
    write_net_wire_segment(json_struct, segment);
  }

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_net_wire_segment(JsonStruct* json_struct, IdbRegularWireSegment* segment)
{
  if (segment->is_rect()) {
    return write_net_wire_segment_rect(json_struct, segment);

  } else if (segment->is_via()) {
    return write_net_wire_segment_via(json_struct, segment);

  } else {
    // two points
    return write_net_wire_segment_points(json_struct, segment);
  }

  return kDbFail;
}

int32_t Gds2JsonWrite::write_net_wire_segment_points(JsonStruct* json_struct, IdbRegularWireSegment* segment)
{
  if (segment->get_point_list().size() < _POINT_MAX_ || segment->get_layer() == nullptr) {
    // std::cout << "Error net wire point..." << std::endl;
    return kDbFail;
  }

  IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(segment->get_layer());

  IdbCoordinate<int32_t>* point_1 = segment->get_point_start();
  IdbCoordinate<int32_t>* point_2 = segment->get_point_second();

  packSegment(json_struct, routing_layer, point_1, point_2);

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_net_wire_segment_via(JsonStruct* json_struct, IdbRegularWireSegment* segment)
{
  if (segment->get_point_list().size() <= 0 || segment->get_layer() == nullptr || segment->get_via_list().size() <= 0) {
    std::cout << "No net wire segment via..." << std::endl;
    return kDbFail;
  }

  packVia(json_struct, segment->get_via_list().at(_POINT_START_));

  if (segment->get_point_number() >= _POINT_MAX_) {
    return write_net_wire_segment_points(json_struct, segment);
  }

  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_net_wire_segment_rect(JsonStruct* json_struct, IdbRegularWireSegment* segment)
{
  if (segment->get_point_list().size() <= 0 || segment->get_layer() == nullptr || segment->get_delta_rect() == nullptr) {
    std::cout << "No net wire segment rect..." << std::endl;
    return kDbFail;
  }

  IdbCoordinate<int32_t>* coordinate = segment->get_point_start();
  IdbRect* rect_delta = segment->get_delta_rect();

  if (coordinate->get_x() < 0 || coordinate->get_y() < 0) {
    std::cout << "Error...Coordinate error...x = " << coordinate->get_x() << " y = " << coordinate->get_y() << std::endl;
  }

  IdbLayer* layer = segment->get_layer();
  if (layer == nullptr) {
    std::cout << "Error...createNetRect : Layer not exist :  " << std::endl;
    return kDbFail;
  }

  IdbRect* rect = new IdbRect(rect_delta);
  rect->moveByStep(coordinate->get_x(), coordinate->get_y());

  packRect(json_struct, rect, layer);

  delete rect;
  rect = nullptr;

  return kDbSuccess;
}

/**
 * @brief Write IO pins, create each IO Term in IdbPin
 *
 */

int32_t Gds2JsonWrite::write_gcell_grid()
{
  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_region()
{
  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_slot()
{
  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_group()
{
  return kDbSuccess;
}

int32_t Gds2JsonWrite::write_fill()
{
  IdbDesign* design = _def_service->get_design();  // def
  IdbFillList* fill_list = design->get_fill_list();
  if (fill_list == nullptr || fill_list->get_num_fill() == 0) {
    std::cout << "No FILLS ..." << std::endl;
    return kDbFail;
  }

  addSRefDefault("Fills");
  JsonStruct* json_struct = new JsonStruct("Fills");

  for (IdbFill* fill : fill_list->get_fill_list()) {
    for (IdbRect* rect : fill->get_layer()->get_rect_list()) {
      packRect(json_struct, rect, fill->get_layer()->get_layer());
    }

    if (fill->get_via() != nullptr) {
      IdbVia* via = fill->get_via()->get_via()->clone();
      for (IdbCoordinate<int32_t>* point : fill->get_via()->get_coordinate_list()) {
        via->set_coordinate(point);

        packVia(json_struct, via);
      }

      delete via;
      via = nullptr;
    }
  }

  addStruct(json_struct);

  writeStruct();

  return kDbSuccess;
}

}  // namespace idb
