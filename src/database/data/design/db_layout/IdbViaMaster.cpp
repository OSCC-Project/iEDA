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
 * @file		IdbViaMaster.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Via master information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbViaMaster.h"

#include <string>

#include "Str.hh"

namespace idb {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void IdbViaMasterRulePattern::parse_pattern(size_t row, size_t col)
{
  if (row <= 0 || col <= 0) {
    return;
  }

  _pattern_state.resize(row);
  for (size_t i = 0; i < row; ++i) {
    _pattern_state[i].resize(col);
    for (size_t j = 0; j < col; ++j) {
      _pattern_state[i][j] = 0;
    }
  }

  _row_num = row;
  _col_num = col;

  vector<pair<string, string>> pattern_array;
  parse_pattern_array(pattern_array);

  size_t row_index = 0;
  for (auto& pattern_pair : pattern_array) {
    int row_num = hexString2Int(pattern_pair.first);
    for (int i = 0; i < row_num; ++i) {
      /// parse pattern state to pattern state array
      parse_pattern_row_value(row_index, pattern_pair.second);

      ++row_index;
    }
  }
}
/**
 * @Brief : parse partern string to string row and row pattern vaule
 * @param  pattern_array : first=row number, second=row pattern vaule, both are hex string
 */
void IdbViaMasterRulePattern::parse_pattern_array(vector<pair<string, string>>& pattern_array)
{
  const char* sep = "_";

  string pattern = _pattern;
  vector<string> strs = ieda::Str::split(pattern.c_str(), sep);
  for (size_t i = 0; i < strs.size(); i += 2) {
    pattern_array.push_back(std::make_pair(strs[i], strs[i + 1]));
  }
}

/**
 * @Brief : parse pattern value for _pattern_state[row_index]
 * @param  row_index row index
 * @param  value value of row
 */
void IdbViaMasterRulePattern::parse_pattern_row_value(size_t row_index, string value)
{
  if (row_index >= _pattern_state.size()) {
    std::cout << "Error : pattern index error, pattern size = " << _pattern_state.size() << " row index = " << row_index << std::endl;
    return;
  }

  int col_index = 0;
  size_t str_index = 0;
  bool b_result = true;
  while (b_result && str_index < value.length()) {
    char char_value = value.at(str_index);
    /// repeat value
    if (repeat_flag == char_value) {
      /// get number
      str_index++;
      char char_number = value.at(str_index);
      int repeat_num = hexString2Int(string(1, char_number));

      /// get value
      str_index++;
      char repeat_value = value.at(str_index);
      for (int i = 0; i < repeat_num; i++) {
        b_result &= save_pattern_value(repeat_value, row_index, col_index);
      }

      str_index++;
    } else {
      b_result &= save_pattern_value(char_value, row_index, col_index);
      str_index++;
    }
  }

  //int a = 0;
}

bool IdbViaMasterRulePattern::save_pattern_value(char value, int row_index, int& col_index)
{
  bool b_result = true;

  int8_t bits_value = hexString2Int(string(1, value));
  b_result &= set_pattern_value(row_index, col_index++, getBitVaule(bits_value, 3));
  b_result &= set_pattern_value(row_index, col_index++, getBitVaule(bits_value, 2));
  b_result &= set_pattern_value(row_index, col_index++, getBitVaule(bits_value, 1));
  b_result &= set_pattern_value(row_index, col_index++, getBitVaule(bits_value, 0));

  return b_result;
}

/**
 * @Brief : transfer hex string to int
 * @param  hex_string
 * @return int
 */
int IdbViaMasterRulePattern::hexString2Int(string hex_string)
{
  return std::stoi(hex_string, nullptr, 16);
}
/**
 * @Brief : get bit value in index
 * @param  value
 * @param  index
 * @return int32_t
 */
int8_t IdbViaMasterRulePattern::getBitVaule(int8_t& value, int32_t index)
{
  int8_t bit_value = 0;
  switch (index) {
    case 0:
      bit_value = value & 0x01;
      break;
    case 1:
      bit_value = (value & 0x02) >> 1;
      break;
    case 2:
      bit_value = (value & 0x04) >> 2;
      break;
    case 3:
      bit_value = (value & 0x08) >> 3;
      break;

    default:
      break;
  }

  return bit_value;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbViaMasterGenerate::IdbViaMasterGenerate()
{
  _rule_name = "";
  _rule_generate = nullptr;
  _cut_size_x = -1;
  _cut_size_y = -1;
  _layer_bottom = nullptr;
  _layer_cut = nullptr;
  _layer_top = nullptr;
  _cut_spacing_x = -1;
  _cut_spacing_y = -1;
  _enclosure_bottom_x = -1;
  _enclosure_bottom_y = -1;
  _enclosure_top_x = -1;
  _enclosure_top_y = -1;
  _num_cut_rows = 1;
  _num_cut_cols = 1;
  _original_offset_x = 0;
  _original_offset_y = 0;
  _offset_bottom_x = 0;
  _offset_bottom_y = 0;
  _offset_top_x = 0;
  _offset_top_y = 0;
  _patttern = nullptr;
  _cut_bouding_rect = new IdbRect();
}

IdbViaMasterGenerate::~IdbViaMasterGenerate()
{
  clear();
}

void IdbViaMasterGenerate::clear()
{
  for (auto& rect : _cut_rect_list) {
    if (rect != nullptr) {
      delete rect;
      rect = nullptr;
    }
  }

  if (_patttern != nullptr) {
    delete _patttern;
    _patttern = nullptr;
  }

  if (_cut_bouding_rect != nullptr) {
    delete _cut_bouding_rect;
    _cut_bouding_rect = nullptr;
  }
}

IdbViaMasterGenerate* IdbViaMasterGenerate::clone()
{
  IdbViaMasterGenerate* generate_new = new IdbViaMasterGenerate();
  generate_new->_rule_name = _rule_name;
  generate_new->_rule_generate = _rule_generate;
  generate_new->_cut_size_x = _cut_size_x;
  generate_new->_cut_size_y = _cut_size_y;
  generate_new->_layer_bottom = _layer_bottom;
  generate_new->_layer_cut = _layer_cut;
  generate_new->_layer_top = _layer_top;
  generate_new->_cut_spacing_x = _cut_spacing_x;
  generate_new->_cut_spacing_y = _cut_spacing_y;
  generate_new->_enclosure_bottom_x = _enclosure_bottom_x;
  generate_new->_enclosure_bottom_y = _enclosure_bottom_y;
  generate_new->_enclosure_top_x = _enclosure_top_x;
  generate_new->_enclosure_top_y = _enclosure_top_y;
  generate_new->_num_cut_rows = _num_cut_rows;
  generate_new->_num_cut_cols = _num_cut_cols;
  generate_new->_original_offset_x = _original_offset_x;
  generate_new->_original_offset_y = _original_offset_y;
  generate_new->_offset_bottom_x = _offset_bottom_x;
  generate_new->_offset_bottom_y = _offset_bottom_y;
  generate_new->_offset_top_x = _offset_top_x;
  generate_new->_offset_top_y = _offset_top_y;

  for (IdbRect* rect : _cut_rect_list) {
    generate_new->add_cut_rect(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
  }

  generate_new->_cut_bouding_rect->set_rect(_cut_bouding_rect);

  if (_patttern != nullptr) {
    generate_new->set_patttern(_patttern->get_pattern_string());
  }

  return generate_new;
}

IdbRect* IdbViaMasterGenerate::add_cut_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y)
{
  IdbRect* rect = new IdbRect(ll_x, ll_y, ur_x, ur_y);
  _cut_rect_list.emplace_back(rect);

  return rect;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbViaMasterFixed::IdbViaMasterFixed()
{
  _layer_shape = new IdbLayerShape();
}

IdbViaMasterFixed::~IdbViaMasterFixed()
{
  clear();
}

void IdbViaMasterFixed::clear()
{
  if (_layer_shape != nullptr) {
    delete _layer_shape;
    _layer_shape = nullptr;
  }
}

IdbRect* IdbViaMasterFixed::get_rect(int32_t index)
{
  if (index < static_cast<int>(_layer_shape->get_rect_list().size()) && _layer_shape->get_rect_list().size() > 0) {
    return _layer_shape->get_rect_list().at(index);
  }

  return nullptr;
}

IdbRect* IdbViaMasterFixed::add_rect()
{
  IdbRect* rect = new IdbRect();

  _layer_shape->get_rect_list().emplace_back(rect);

  return rect;
}

void IdbViaMasterFixed::add_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y)
{
  IdbRect* rect = new IdbRect(ll_x, ll_y, ur_x, ur_y);
  _layer_shape->get_rect_list().emplace_back(rect);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbViaMaster::IdbViaMaster()
{
  _name = "";
  _type = IdbViaMasterType::kNone;
  _is_default = false;
  _master_generate = new IdbViaMasterGenerate();
  _cut_rect = new IdbRect();
  _layer_shape_bottom = new IdbLayerShape(IdbLayerShapeType::kRect);
  _layer_shape_cut = new IdbLayerShape(IdbLayerShapeType::kVia);
  _layer_shape_top = new IdbLayerShape(IdbLayerShapeType::kRect);
}

IdbViaMaster::~IdbViaMaster()
{
  clear();
}

void IdbViaMaster::clear()
{
  if (_master_generate != nullptr) {
    delete _master_generate;
    _master_generate = nullptr;
  }

  for (auto& master_fixed : _master_fixed_list) {
    if (master_fixed != nullptr) {
      delete master_fixed;
      master_fixed = nullptr;
    }
  }
  _master_fixed_list.clear();

  if (_cut_rect != nullptr) {
    delete _cut_rect;
    _cut_rect = nullptr;
  }

  if (_layer_shape_bottom != nullptr) {
    delete _layer_shape_bottom;
    _layer_shape_bottom = nullptr;
  }

  if (_layer_shape_cut != nullptr) {
    delete _layer_shape_cut;
    _layer_shape_cut = nullptr;
  }

  if (_layer_shape_top != nullptr) {
    delete _layer_shape_top;
    _layer_shape_top = nullptr;
  }
}

IdbViaMasterFixed* IdbViaMaster::add_fixed(string layer_name)
{
  /// juge if layer exist
  for (auto fix_item : _master_fixed_list) {
    if (fix_item->get_layer()->compareLayer(layer_name)) {
      return fix_item;
    }
  }

  /// if not exist, create one
  IdbViaMasterFixed* master_fixed = new IdbViaMasterFixed();
  _master_fixed_list.emplace_back(master_fixed);

  return master_fixed;
}

IdbViaMaster* IdbViaMaster::clone()
{
  IdbViaMaster* master_new = new IdbViaMaster();
  master_new->_name = _name;
  master_new->_is_default = _is_default;
  master_new->_type = _type;
  master_new->_master_generate = _master_generate->clone();
  master_new->_master_fixed_list = _master_fixed_list;
  master_new->_cut_rect->set_rect(_cut_rect);

  return master_new;
}

void IdbViaMaster::set_cut_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y)
{
  _cut_rect->set_rect(ll_x, ll_y, ur_x, ur_y);
}

void IdbViaMaster::set_top_layer_shape()
{
  set_layer_shape(IdbViaLayerIndex::kLayerTop);
}

void IdbViaMaster::set_cut_layer_shape()
{
  set_layer_shape(IdbViaLayerIndex::kLayerCut);
}

void IdbViaMaster::set_bottom_layer_shape()
{
  set_layer_shape(IdbViaLayerIndex::kLayerBottom);
}

void IdbViaMaster::set_layer_shape(IdbViaLayerIndex layer_index)
{
  if (is_generate()) {
    IdbViaMasterGenerate* master_generate = get_master_generate();

    int32_t enclosure_x = 0;
    int32_t enclosure_y = 0;

    if (layer_index == IdbViaLayerIndex::kLayerBottom) {
      _layer_shape_bottom->set_layer(master_generate->get_layer_bottom());
      _layer_shape_bottom->set_type_rect();
      enclosure_x = master_generate->get_enclosure_bottom_x();
      enclosure_y = master_generate->get_enclosure_bottom_y();

      IdbRect* cut_bouding_rect = master_generate->get_cut_bouding_rect();
      int ll_x = cut_bouding_rect->get_low_x() - enclosure_x;
      int ll_y = cut_bouding_rect->get_low_y() - enclosure_y;
      int ur_x = cut_bouding_rect->get_high_x() + enclosure_x;
      int ur_y = cut_bouding_rect->get_high_y() + enclosure_y;
      _layer_shape_bottom->add_rect(IdbRect(ll_x, ll_y, ur_x, ur_y));

    } else if (layer_index == IdbViaLayerIndex::kLayerTop) {
      _layer_shape_top->set_layer(master_generate->get_layer_top());
      _layer_shape_top->set_type_rect();
      enclosure_x = master_generate->get_enclosure_top_x();
      enclosure_y = master_generate->get_enclosure_top_y();

      IdbRect* cut_bouding_rect = master_generate->get_cut_bouding_rect();
      int ll_x = cut_bouding_rect->get_low_x() - enclosure_x;
      int ll_y = cut_bouding_rect->get_low_y() - enclosure_y;
      int ur_x = cut_bouding_rect->get_high_x() + enclosure_x;
      int ur_y = cut_bouding_rect->get_high_y() + enclosure_y;
      _layer_shape_top->add_rect(IdbRect(ll_x, ll_y, ur_x, ur_y));
    } else {
      _layer_shape_cut->set_layer(master_generate->get_layer_cut());
      _layer_shape_cut->set_type_via();
      for (IdbRect* rect : master_generate->get_cut_rect_list()) {
        _layer_shape_cut->add_rect(*rect);
      }
    }

  } else if (is_fix()) {
    IdbLayerShape* layer_shape = nullptr;
    switch (layer_index) {
      case IdbViaLayerIndex::kLayerBottom: {
        layer_shape = _layer_shape_bottom;
        layer_shape->set_type_rect();
        break;
      }
      case IdbViaLayerIndex::kLayerCut: {
        layer_shape = _layer_shape_cut;
        layer_shape->set_type_via();
        break;
      }
      case IdbViaLayerIndex::kLayerTop: {
        layer_shape = _layer_shape_top;
        layer_shape->set_type_rect();
        break;
      }
      default:
        break;
    }
    if (_master_fixed_list.size() != IdbViaLayerIndex::kMax) {
      std::cout << "Error fix via : fix layer number must be ==3..." << std::endl;
    } else {
      //   if (layer_index == IdbViaLayerIndex::kLayerBottom || layer_index == IdbViaLayerIndex::kLayerTop) {
      //     IdbViaMasterFixed* master_fix_find = get_master_fixed(layer_index);
      //     layer_shape->set_layer(master_fix_find->get_layer());
      //     for (IdbRect* rect : master_fix_find->get_rect_list())
      //       layer_shape->add_rect(*rect);
      //   } else {
      //     IdbViaMasterFixed* master_fix_find = get_master_fixed(layer_index);
      //     layer_shape->set_layer(master_fix_find->get_layer_shape()->get_layer());
      //     ////tbd------need to confirmed that there are more than 1 cut master existed in 1 via???
      //     for (IdbViaMasterFixed* master_fix : _master_fixed_list) {
      //       if (master_fix->get_layer()->is_cut()) {
      //         for (IdbRect* rect : master_fix->get_rect_list())
      //           layer_shape->add_rect(*rect);
      //       }
      //     }
      //   }

      IdbViaMasterFixed* master_fix_find = get_master_fixed(layer_index);
      layer_shape->set_layer(master_fix_find->get_layer());
      for (IdbRect* rect : master_fix_find->get_rect_list())
        layer_shape->add_rect(*rect);
    }

  } else {
    std::cout << "Error IdbViaMaster : No via master exist..." << std::endl;
  }
}
/// find top\bottom\cut fix master according to layer index
IdbViaMasterFixed* IdbViaMaster::get_master_fixed(IdbViaLayerIndex layer_index)
{
  IdbViaMasterFixed* master_fixed_find = nullptr;

  switch (layer_index) {
    case IdbViaLayerIndex::kLayerBottom:
      for (IdbViaMasterFixed* master_fixed : _master_fixed_list) {
        if (master_fixed->get_layer()->is_routing()) {
          if (master_fixed_find == nullptr) {
            master_fixed_find = master_fixed;
          } else {
            if (master_fixed_find->get_layer()->get_order() > master_fixed->get_layer()->get_order()) {
              master_fixed_find = master_fixed;
            }
          }
        }
      }
      break;
    case IdbViaLayerIndex::kLayerCut:
      for (IdbViaMasterFixed* master_fixed : _master_fixed_list) {
        if (master_fixed->get_layer()->is_cut()) {
          master_fixed_find = master_fixed;
        }
      }
      break;
    case IdbViaLayerIndex::kLayerTop:
      for (IdbViaMasterFixed* master_fixed : _master_fixed_list) {
        if (master_fixed->get_layer()->is_routing()) {
          if (master_fixed_find == nullptr) {
            master_fixed_find = master_fixed;
          } else {
            if (master_fixed_find->get_layer()->get_order() < master_fixed->get_layer()->get_order()) {
              master_fixed_find = master_fixed;
            }
          }
        }
      }
      break;

    default:
      break;
  }

  return master_fixed_find;
}

bool IdbViaMaster::isOneCut()
{
  bool is_one_cut = false;

  if (_num_cut_rows == 1 && _num_cut_cols == 1) {
    is_one_cut = true;
  }

  if (_layer_shape_cut->get_rect_list_num() == 1) {
    is_one_cut = true;
  }

  return is_one_cut;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbViaMasterList::IdbViaMasterList()
{
  _num_master = 0;
}

IdbViaMasterList::~IdbViaMasterList()
{
}

IdbViaMaster* IdbViaMasterList::find_via_master(string name)
{
  for (IdbViaMaster* via_master : _via_master_list) {
    if (via_master->get_name() == name) {
      return via_master;
    }
  }

  return nullptr;
}

IdbViaMaster* IdbViaMasterList::find_via_master(int32_t index)
{
  if ((static_cast<int>(_via_master_list.size()) > index) && (index >= 0)) {
    return _via_master_list.at(index);
  }

  return nullptr;
}

IdbViaMaster* IdbViaMasterList::add_via_master(IdbViaMaster* via_master)
{
  IdbViaMaster* pMaster = via_master;
  if (pMaster == nullptr) {
    pMaster = new IdbViaMaster();
  }
  _via_master_list.emplace_back(pMaster);
  _num_master++;

  return pMaster;
}

IdbViaMaster* IdbViaMasterList::add_via_master(string name)
{
  IdbViaMaster* pMaster = find_via_master(name);
  if (pMaster == nullptr) {
    pMaster = new IdbViaMaster();
    pMaster->set_name(name);
    _via_master_list.emplace_back(pMaster);
    _num_master++;
  }

  return pMaster;
}

}  // namespace idb
