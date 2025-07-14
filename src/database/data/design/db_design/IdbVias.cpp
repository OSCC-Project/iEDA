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
 * @file		IdbVias.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Vias information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbVias.h"

#include <limits.h>

#include <algorithm>
#include <cstdio>

#include "../db_layout/IdbLayer.h"
#include "IdbViaMaster.h"
#include "Str.hh"

namespace idb {

IdbVia::IdbVia()
{
  _name = "";
  _coordinate = new IdbCoordinate<int32_t>();
  _master_instance = nullptr;
}

IdbVia::~IdbVia()
{
  clear();
}

void IdbVia::clear()
{
  if (_master_instance != nullptr && _b_master_clone == false) {
    delete _master_instance;
    _master_instance = nullptr;
  }

  if (_coordinate != nullptr) {
    delete _coordinate;
    _coordinate = nullptr;
  }
}

IdbViaMaster* IdbVia::get_instance()
{
  if (_master_instance == nullptr) {
    _master_instance = new IdbViaMaster();
    _b_master_clone = false;
  }
  return _master_instance;
}

IdbVia* IdbVia::clone()
{
  IdbVia* via_new = new IdbVia();
  via_new->_name = _name;
  via_new->_master_instance = _master_instance;  //_via_instance->clone();
  via_new->_b_master_clone = true;
  via_new->_coordinate->set_xy(_coordinate->get_x(), _coordinate->get_y());

  return via_new;
}

void IdbVia::set_instance(IdbViaMaster* instance)
{
  if (_master_instance != nullptr && _b_master_clone == false) {
    delete _master_instance;
    _master_instance = nullptr;
  }
  _master_instance = instance;
  _name = instance->get_name();
}

void IdbVia::reset_instance(IdbViaMaster* instance)
{
  set_instance(instance);
}

void IdbVia::set_coordinate(IdbCoordinate<int32_t>* point)
{
  _coordinate->set_xy(point->get_x(), point->get_y());
  //   set_bounding_box();
}

// bool IdbVia::set_bounding_box() {
//   if (_master_instance->is_fix()) {
//     IdbRect* rect = get_bounding_box();

//     IdbRect* master_rect = _master_instance->get_cut_rect();
//     rect->set_rect(_coordinate->get_x() + master_rect->get_low_x(), _coordinate->get_y() + master_rect->get_low_y(),
//                    _coordinate->get_x() + master_rect->get_high_x(), _coordinate->get_y() +
//                    master_rect->get_high_y());
//   } else if (_master_instance->is_generate()) {
//     IdbRect* rect     = get_bounding_box();
//     IdbRect* fix_rect = _master_instance->get_cut_rect();
//     rect->set_rect(_coordinate->get_x() + fix_rect->get_low_x(), _coordinate->get_y() + fix_rect->get_low_y(),
//                    _coordinate->get_x() + fix_rect->get_high_x(), _coordinate->get_y() + fix_rect->get_high_y());
//   } else {
//     std::cout << "Error set via bounding box... name = " << _name << std::endl;
//   }

//   return true;
// }

IdbLayerShape IdbVia::get_bottom_layer_shape()
{
  IdbLayerShape* layer_shape = _master_instance->get_bottom_layer_shape();

  IdbLayerShape via_shape;
  layer_shape->clone(via_shape);
  via_shape.moveToLocation(_coordinate);

  return via_shape;
}

IdbRect IdbVia::get_bottom_bounding_box()
{
  IdbLayerShape layer_shape = get_bottom_layer_shape();
  return layer_shape.get_bounding_box();
}

IdbLayerShape IdbVia::get_top_layer_shape()
{
  IdbLayerShape* layer_shape = _master_instance->get_top_layer_shape();

  IdbLayerShape via_shape;
  layer_shape->clone(via_shape);
  via_shape.moveToLocation(_coordinate);

  return via_shape;
}

IdbRect IdbVia::get_top_bounding_box()
{
  IdbLayerShape layer_shape = get_top_layer_shape();
  return layer_shape.get_bounding_box();
}

IdbLayerShape IdbVia::get_cut_layer_shape()
{
  IdbLayerShape* layer_shape = _master_instance->get_cut_layer_shape();

  IdbLayerShape via_shape;
  layer_shape->clone(via_shape);
  via_shape.moveToLocation(_coordinate);

  return via_shape;
}

IdbRect IdbVia::get_cut_bounding_box()
{
  IdbLayerShape layer_shape = get_cut_layer_shape();
  return layer_shape.get_bounding_box();
}

bool IdbVia::isIntersection(IdbRect rect, IdbLayer* layer)
{
  IdbLayerShape layer_bootom = get_bottom_layer_shape();
  if (layer->compareLayer(layer_bootom.get_layer()) && rect.isIntersection(layer_bootom.get_bounding_box())) {
    return true;
  }
  IdbLayerShape layer_top = get_top_layer_shape();
  if (layer->compareLayer(layer_top.get_layer()) && rect.isIntersection(layer_top.get_bounding_box())) {
    return true;
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbVias::IdbVias()
{
  _num_vias = 0;
}

IdbVias::~IdbVias()
{
  for (IdbVia* via : _via_list) {
    if (via != nullptr) {
      delete via;
      via = nullptr;
    }
  }
}

IdbVia* IdbVias::find_via(const string& name)
{
  for (IdbVia* via : _via_list) {
    if (via->get_name() == name) {
      return via;
    }
  }

  return nullptr;
}

IdbVia* IdbVias::find_via(size_t index)
{
  if (_num_vias > index) {
    return _via_list.at(index);
  }

  return nullptr;
}

IdbVia* IdbVias::find_via_generate(IdbLayerCut* layer_cut, int32_t width, int32_t height, string via_name)
{
  IdbVia* via_find = nullptr;
  /// step 1 : find the same name in VIAS,
  if (!via_name.empty()) {
    via_find = find_via(via_name);
    if (via_find != nullptr) {
      return via_find;
    }
  }
  /// step 2 : find the matched via rule between cut layer and via list
  if (layer_cut == nullptr) {
    std::cout << "Error: Cut layer illegal." << std::endl;
    return nullptr;
  }

  for (IdbVia* via : _via_list) {
    IdbViaMaster* via_master = via->get_instance();
    if (via_master != nullptr && via_master->is_generate()) {
      IdbViaMasterGenerate* master_generate = via_master->get_master_generate();
      if (master_generate != nullptr && master_generate->get_rule_generate() == layer_cut->get_via_rule()) {
        return via;
      }
    }
  }

  /// step 3 : if find none, create via as via_name
  if (width == 0 || height == 0) {
    std::cout << "Error: width and height must be set." << std::endl;
    return nullptr;
  }

  if (via_name.empty()) {
    via_name = layer_cut->get_name() + "_" + std::to_string(width) + "x" + std::to_string(height);
  }

  return createVia(via_name, layer_cut, width, height);
  ;
}

IdbVia* IdbVias::add_via(IdbVia* via)
{
  IdbVia* pVia = via;
  if (pVia == nullptr) {
    pVia = new IdbVia();
  }
  _via_list.emplace_back(pVia);
  _num_vias++;

  return pVia;
}

IdbVia* IdbVias::add_via(string name)
{
  IdbVia* pVia = find_via(name);
  if (pVia == nullptr) {
    pVia = new IdbVia();
    pVia->set_name(name);
    _via_list.emplace_back(pVia);
    _num_vias++;
  }

  return pVia;
}
/// ----tbd---------
///
/// via_name : via name created in VIAS of def file
/// layer_cut : pointer of cut layer, contains the viarule for the cut layer
/// row :
/// col :
/// width_design : width that set by the designer
/// height_design : height that set by the designer

IdbVia* IdbVias::createVia(string via_name, IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design)
{
  IdbViaRuleGenerate* via_rule = layer_cut->get_via_rule();
  if (via_rule == nullptr)
    return nullptr;

  std::pair<int32_t, int32_t> row_col_pair = calculateRowsCols(layer_cut, width_design, height_design);
  int32_t rows = row_col_pair.first;
  int32_t cols = row_col_pair.second;

  IdbLayerRouting* layer_bottom = via_rule->get_layer_bottom();
  IdbLayerRouting* layer_top = via_rule->get_layer_top();

  IdbVia* via_new = add_via(via_name);

  IdbViaMaster* master_instance = via_new->get_instance();
  master_instance->set_type_generate();
  IdbViaMasterGenerate* master_generate = master_instance->get_master_generate();
  master_generate->set_rule_name(via_rule->get_name());
  master_generate->set_rule_generate(via_rule);

  master_generate->set_layer_bottom(layer_bottom);
  master_generate->set_layer_cut(layer_cut);
  master_generate->set_layer_top(layer_top);

  int32_t cutsize_x = via_rule->get_cut_rect()->get_width();
  int32_t cutsize_y = via_rule->get_cut_rect()->get_height();
  master_generate->set_cut_size(cutsize_x, cutsize_y);

  int32_t spacing_x = via_rule->get_spacing_x();
  int32_t spacing_y = via_rule->get_spacing_y();
  if (!layer_cut->get_spacings().empty()) {
    int layer_cut_spacing = layer_cut->get_spacings().at(0)->get_spacing();
    spacing_x = std::max(spacing_x, cutsize_x + layer_cut_spacing);
    spacing_y = std::max(spacing_y, cutsize_y + layer_cut_spacing);
  }

  int32_t cut_spacing_x = spacing_x - cutsize_x;
  int32_t cut_spacing_y = spacing_y - cutsize_y;
  master_generate->set_cut_spacing(cut_spacing_x, cut_spacing_y);

  master_generate->set_cut_row_col(rows, cols);
  master_instance->set_cut_row_col(rows, cols);
  /// generate pattern string if rows > cutarray size and cols > cutarray size
  string pattern_string = createViaPatternString(rows, cols, layer_cut->get_array_spacing());
  if (!pattern_string.empty()) {
    master_generate->set_patttern(pattern_string);
  }

  int32_t bottom_enclosure_x = 0;
  int32_t bottom_enclosure_y = 0;
  int32_t top_enclosure_x = 0;
  int32_t top_enclosure_y = 0;

  if (layer_bottom->is_horizontal()) {
    int32_t rule_bottom_enclosure_x = via_rule->get_enclosure_bottom()->get_overhang_1();
    int32_t rule_bottom_enclosure_y = via_rule->get_enclosure_bottom()->get_overhang_2();

    /// calculate enclosure y
    int32_t height = height_design > 0 ? height_design
                                       : (layer_bottom->get_power_segment_width() == 0 ? layer_bottom->get_width()
                                                                                       : layer_bottom->get_power_segment_width());

    bottom_enclosure_y = (height - (rows * cutsize_y + (rows - kMinRowColNum) * cut_spacing_y)) / 2;
    bottom_enclosure_y = std::max(bottom_enclosure_y, rule_bottom_enclosure_y);
    bottom_enclosure_y = rule_bottom_enclosure_y;  /// set enclosure y as default

    /// caculate bottom width
    int32_t bottom_height = rows * cutsize_y + (rows - kMinRowColNum) * cut_spacing_y + bottom_enclosure_y * 2;
    int32_t area = layer_bottom->get_area();
    int32_t bottom_width = area % bottom_height == 0 ? area / bottom_height : area / bottom_height + 1;
    bottom_width = std::max(bottom_width, width_design);

    /// calculate enclosure x
    bottom_enclosure_x = (bottom_width - (cols * cutsize_x + (cols - kMinRowColNum) * cut_spacing_x)) / 2;
    bottom_enclosure_x = std::max(bottom_enclosure_x, rule_bottom_enclosure_x);

  } else {
    int32_t rule_bottom_enclosure_x = via_rule->get_enclosure_bottom()->get_overhang_2();
    int32_t rule_bottom_enclosure_y = via_rule->get_enclosure_bottom()->get_overhang_1();

    /// calculate enclosure x
    int32_t width = width_design > 0 ? width_design
                                     : (layer_bottom->get_power_segment_width() == 0 ? layer_bottom->get_width()
                                                                                     : layer_bottom->get_power_segment_width());
    bottom_enclosure_x = (width - (cols * cutsize_x + (cols - kMinRowColNum) * cut_spacing_x)) / 2;
    bottom_enclosure_x = std::max(bottom_enclosure_x, rule_bottom_enclosure_x);
    bottom_enclosure_x = rule_bottom_enclosure_x;  /// set enclosure x as default

    /// caculate bottom height
    int32_t bottom_width = cols * cutsize_x + (cols - kMinRowColNum) * cut_spacing_x + bottom_enclosure_x * 2;
    int32_t area = layer_bottom->get_area();
    int32_t bottom_height = area % bottom_width == 0 ? area / bottom_width : area / bottom_width + 1;
    bottom_height = std::max(bottom_height, height_design);

    /// calculate enclosure y
    bottom_enclosure_y = (bottom_height - (rows * cutsize_y + (rows - kMinRowColNum) * cut_spacing_y)) / 2;
    bottom_enclosure_y = std::max(bottom_enclosure_y, rule_bottom_enclosure_y);
  }

  if (layer_top->is_horizontal()) {
    int32_t rule_top_enclosure_x = via_rule->get_enclosure_top()->get_overhang_1();
    int32_t rule_top_enclosure_y = via_rule->get_enclosure_top()->get_overhang_2();

    /// calculate enclosure y
    int32_t height = height_design > 0
                         ? height_design
                         : (layer_top->get_power_segment_width() == 0 ? layer_top->get_width() : layer_top->get_power_segment_width());
    top_enclosure_y = (height - (rows * cutsize_y + (rows - kMinRowColNum) * cut_spacing_y)) / 2;
    top_enclosure_y = std::max(top_enclosure_y, rule_top_enclosure_y);
    top_enclosure_y = rule_top_enclosure_y;  /// set enclosure y as default

    /// caculate top width
    int32_t top_height = rows * cutsize_y + (rows - kMinRowColNum) * cut_spacing_y + top_enclosure_y * 2;
    int32_t area = layer_top->get_area();
    int32_t top_width = area % top_height == 0 ? area / top_height : area / top_height + 1;
    top_width = std::max(top_width, width_design);

    /// calculate enclosure x
    top_enclosure_x = (top_width - (cols * cutsize_x + (cols - kMinRowColNum) * cut_spacing_x)) / 2;
    top_enclosure_x = std::max(top_enclosure_x, rule_top_enclosure_x);
  } else {
    int32_t rule_top_enclosure_x = via_rule->get_enclosure_top()->get_overhang_2();
    int32_t rule_top_enclosure_y = via_rule->get_enclosure_top()->get_overhang_1();

    /// calculate enclosure x
    int32_t width = width_design > 0
                        ? width_design
                        : (layer_top->get_power_segment_width() == 0 ? layer_top->get_width() : layer_top->get_power_segment_width());
    top_enclosure_x = (width - (cols * cutsize_x + (cols - kMinRowColNum) * cut_spacing_x)) / 2;
    top_enclosure_x = std::max(top_enclosure_x, rule_top_enclosure_x);
    top_enclosure_x = rule_top_enclosure_x;  /// set enclosure x as default

    /// caculate top height
    int32_t top_width = cols * cutsize_x + (cols - kMinRowColNum) * cut_spacing_x + top_enclosure_x * 2;
    int32_t area = layer_top->get_area();
    int32_t top_height = area % top_width == 0 ? area / top_width : area / top_width + 1;
    top_height = std::max(top_height, height_design);

    /// calculate enclosure y
    top_enclosure_y = (top_height - (rows * cutsize_y + (rows - kMinRowColNum) * cut_spacing_y)) / 2;
    top_enclosure_y = std::max(top_enclosure_y, rule_top_enclosure_y);
  }

  master_generate->set_enclosure_bottom(bottom_enclosure_x, bottom_enclosure_y);
  master_generate->set_enclosure_top(top_enclosure_x, top_enclosure_y);

  // build core cut shape
  vector<IdbRect*> cut_rect_list = master_generate->get_cut_rect_list();

  int32_t cut_width_total = cols * cutsize_x + (cols - 1) * cut_spacing_x;
  int32_t cut_height_total = rows * cutsize_y + (rows - 1) * cut_spacing_y;

  int32_t original_offset_x = master_generate->get_original_offset_x();
  int32_t original_offset_y = master_generate->get_original_offset_y();
  int32_t ll_x_min = (-cut_width_total / 2) + original_offset_x;
  int32_t ll_y_min = (-cut_height_total / 2) + original_offset_y;
  for (int32_t i = 0; i < rows; ++i) {
    for (int32_t j = 0; j < cols; j++) {
      if (nullptr != master_generate->get_patttern() && !master_generate->is_pattern_cut_exist(i, j)) {
        continue;
      }

      int32_t ll_x = ll_x_min + j * (cutsize_x + cut_spacing_x);
      int32_t ll_y = ll_y_min + i * (cutsize_y + cut_spacing_y);
      int32_t ur_x = ll_x + cutsize_x;
      int32_t ur_y = ll_y + cutsize_y;
      master_generate->add_cut_rect(ll_x, ll_y, ur_x, ur_y);
    }
  }

  master_generate->set_cut_bouding_rect(ll_x_min, ll_y_min, ll_x_min + cut_width_total, ll_y_min + cut_height_total);
  master_instance->set_via_shape();
  return via_new;
}

string IdbVias::createViaPatternString(int row_num, int col_num, IdbLayerCutArraySpacing* _array_spacing)
{
  string pattern_string = "";

  if (_array_spacing != nullptr) {
    int array_cut_min = _array_spacing->get_array_cut_min_num();

    if (array_cut_min > 0 && row_num > array_cut_min && col_num > array_cut_min) {
      /// construct seperater
      string row_seperate_string = "1_R" + transIntToHexBy4Bits((col_num / array_cut_min) + 1) + "0";

      int row_index = 0;
      while (row_index < row_num) {
        if (row_index % array_cut_min == 0) {
          /// construct module string
          int array_cut_num = std::min((row_num - row_index), array_cut_min - 1);
          string row_module = transIntToHexBy4Bits(array_cut_num) + "_" + constructRowPattern(col_num, array_cut_min);

          /// module string
          pattern_string += row_module;

          row_index += array_cut_num;
        } else {
          /// cut seperate string, described as 0
          pattern_string += row_seperate_string;

          row_index += 1;
        }

        if (row_index < row_num) {
          pattern_string += "_";
        }
      }
    }
  }

  return pattern_string;
}

// string IdbVias::constructRowPattern(int col_num, int array_cut_num) {
//   string row_pattern = "";

//   uint8_t value = 0xFF;
//   for (int i = 0; i < col_num; ++i) {
//     if (0 == i % 8) {
//       /// restore value
//       value = 0xFF;
//     }
//     /// set data bit as 0 while (0 == (i+1) % (array_cut_num+1))
//     if (0 == ((i + 1) % (array_cut_num + 1))) {
//       uint8_t mask = ~(0x80 >> (i % 8));
//       value &= mask;
//     }

//     if ((i == (col_num - 1)) || (0 == (i + 1) % 8)) {
//       /// save value
//       row_pattern += transInt8ToHex(value);
//     }
//   }

//   return row_pattern;
// }

string IdbVias::constructRowPattern(int col_num, int array_cut_num)
{
  string row_pattern = "";

  uint8_t value = 0;
  for (int i = 0; i < col_num; ++i) {
    if (0 == i % 4) {
      /// restore value
      value = 0;
    }

    if (0 != ((i + 1) % array_cut_num)) {
      uint8_t mask = 0x08 >> (i % 4);
      value |= mask;
    }

    if ((i == (col_num - 1)) || (0 == (i + 1) % 4)) {
      /// save value
      row_pattern += transIntToHexBy4Bits(value);
    }
  }

  return row_pattern;
}

string IdbVias::transIntToHexBy4Bits(uint8_t value)
{
  string result = "";
  switch (value) {
    case 0:
      result = "0";
      break;
    case 1:
      result = "1";
      break;
    case 2:
      result = "2";
      break;
    case 3:
      result = "3";
      break;
    case 4:
      result = "4";
      break;
    case 5:
      result = "5";
      break;
    case 6:
      result = "6";
      break;
    case 7:
      result = "7";
      break;
    case 8:
      result = "8";
      break;
    case 9:
      result = "9";
      break;
    case 10:
      result = "A";
      break;
    case 11:
      result = "B";
      break;
    case 12:
      result = "C";
      break;
    case 13:
      result = "D";
      break;
    case 14:
      result = "E";
      break;
    case 15:
      result = "F";
      break;

    default:
      break;
  }
  return result;
}

string IdbVias::transInt8ToHex(uint8_t value)
{
  uint8_t low_bits = value & 0x0F;
  uint8_t high_bits = value >> 4;

  return transIntToHexBy4Bits(high_bits) + transIntToHexBy4Bits(low_bits);
}

// IdbVia* IdbVias::createViaDefault(string via_name, IdbLayerCut* layer_cut)
// {
//   IdbViaRuleGenerate* via_rule = layer_cut->get_via_rule();
//   if (via_rule == nullptr)
//     return nullptr;

//   int width = 0;
//   int height = 0;

//   if (layer_cut->get_name() == "VIA12") {
//     width = 820;
//     height = 240;
//   } else if (layer_cut->get_name() == "VIA23") {
//     width = 820;
//     height = 240;
//   } else if (layer_cut->get_name() == "VIA34") {
//     width = 820;
//     height = 240;
//   } else if (layer_cut->get_name() == "VIA45") {
//     width = 820;
//     height = 1640;
//   } else if (layer_cut->get_name() == "VIA56") {
//     width = 1640;
//     height = 1640;
//   } else if (layer_cut->get_name() == "VIA67") {
//     width = 1640;
//     height = 4100;
//   } else if (layer_cut->get_name() == "VIA78") {
//     width = 16400;
//     height = 4100;
//   } else if (layer_cut->get_name() == "PA") {
//     width = 16400;
//     height = 20000;
//   } else {
//   }

//   //   std::pair<int32_t, int32_t> row_col_pair = calculateRowsCols(layer_cut, width, height);

//   return createVia(via_name, layer_cut, width, height);
// }

std::pair<int32_t, int32_t> IdbVias::calculateRowsCols(IdbLayerCut* layer_cut, int32_t width, int32_t height)
{
  int32_t num_rows = kMinRowColNum;
  int32_t num_cols = kMinRowColNum;

  IdbViaRuleGenerate* via_rule = layer_cut->get_via_rule();
  if (via_rule == nullptr) {
    return std::make_pair(num_rows, num_cols);
  }

  /// calculate width and height
  IdbLayerRouting* layer_bottom = via_rule->get_layer_bottom();
  IdbLayerRouting* layer_top = via_rule->get_layer_top();
  int32_t cutsize_x = via_rule->get_cut_rect()->get_width();
  int32_t cutsize_y = via_rule->get_cut_rect()->get_height();
  int32_t spacing_x = via_rule->get_spacing_x();
  int32_t spacing_y = via_rule->get_spacing_y();

  if (!layer_cut->get_spacings().empty()) {
    int layer_cut_spacing = layer_cut->get_spacings().at(0)->get_spacing();
    spacing_x = std::max(spacing_x, cutsize_x + layer_cut_spacing);
    spacing_y = std::max(spacing_y, cutsize_y + layer_cut_spacing);
  }

  if (width > 0 && height > 0) {
    num_rows = std::max(kMinRowColNum + (height - cutsize_y) / spacing_y, kMinRowColNum);
    num_cols = std::max(kMinRowColNum + (width - cutsize_x) / spacing_x, kMinRowColNum);
  } else {
    if (layer_bottom->is_horizontal() && layer_top->is_vertical()) {
      width = layer_top->get_power_segment_width() == 0 ? layer_top->get_width() : layer_top->get_power_segment_width();
      height = layer_bottom->get_power_segment_width() == 0 ? layer_bottom->get_width() : layer_bottom->get_power_segment_width();
      int32_t rule_bottom_enclosure_y = via_rule->get_enclosure_bottom()->get_overhang_2();
      int32_t rule_top_enclosure_x = via_rule->get_enclosure_top()->get_overhang_1();
      num_rows = std::max(kMinRowColNum + (height - cutsize_y - rule_bottom_enclosure_y * 2) / spacing_y, kMinRowColNum);
      num_cols = std::max(kMinRowColNum + (width - cutsize_x - rule_top_enclosure_x * 2) / spacing_x, kMinRowColNum);
    } else if (layer_top->is_horizontal() && layer_bottom->is_vertical()) {
      width = layer_bottom->get_power_segment_width() == 0 ? layer_bottom->get_width() : layer_bottom->get_power_segment_width();
      height = layer_top->get_power_segment_width() == 0 ? layer_top->get_width() : layer_top->get_power_segment_width();
      int32_t rule_top_enclosure_y = via_rule->get_enclosure_top()->get_overhang_2();
      int32_t rule_bottom_enclosure_x = via_rule->get_enclosure_bottom()->get_overhang_1();
      num_rows = std::max(kMinRowColNum + (height - cutsize_y - rule_top_enclosure_y * 2) / spacing_y, kMinRowColNum);
      num_cols = std::max(kMinRowColNum + (width - cutsize_x - rule_bottom_enclosure_x * 2) / spacing_x, kMinRowColNum);
    } else if (layer_top->is_horizontal() && layer_bottom->is_horizontal()) {
      int32_t area = layer_bottom->get_area();  /// assert area of bottom < area of top
      height = layer_bottom->get_power_segment_width() == 0
                   ? layer_bottom->get_width()
                   : layer_bottom->get_power_segment_width();  /// assert width of bottom < width of top
      width = area % height == 0 ? area / height : area / height + kMinRowColNum;

      int32_t bottom_enclosure_x = via_rule->get_enclosure_bottom()->get_overhang_1();
      int32_t bottom_enclosure_y = via_rule->get_enclosure_bottom()->get_overhang_2();

      num_rows = std::max(kMinRowColNum + (height - cutsize_y - bottom_enclosure_y * 2) / spacing_y, kMinRowColNum);
      num_cols = std::max(kMinRowColNum + (width - cutsize_x - bottom_enclosure_x * 2) / spacing_x, kMinRowColNum);
    } else if (layer_top->is_vertical() && layer_bottom->is_vertical()) {
      int32_t area = layer_bottom->get_area();  /// assert area of bottom < area of top
      width = layer_bottom->get_power_segment_width() == 0
                  ? layer_bottom->get_width()
                  : layer_bottom->get_power_segment_width();  /// assert width of bottom < width of top
      height = area % width == 0 ? area / width : area / width + kMinRowColNum;

      int32_t bottom_enclosure_x = via_rule->get_enclosure_bottom()->get_overhang_1();
      int32_t bottom_enclosure_y = via_rule->get_enclosure_bottom()->get_overhang_2();

      num_rows = std::max(kMinRowColNum + (height - cutsize_y - bottom_enclosure_y * 2) / spacing_y, kMinRowColNum);
      num_cols = std::max(kMinRowColNum + (width - cutsize_x - bottom_enclosure_x * 2) / spacing_x, kMinRowColNum);
    } else {
      /// do nothing
    }
  }

  return std::make_pair(num_rows, num_cols);
}

}  // namespace idb
