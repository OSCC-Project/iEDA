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
#pragma once
/**
 * @project		iDB
 * @file		IdbBlockages.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Blockages information.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"

namespace idb {

// using std::vector;

class IdbLayer;
class IdbInstance;
class IdbBlockage
{
 public:
  IdbBlockage();
  virtual ~IdbBlockage();

  enum IdbBlockageType
  {
    kNone,
    kRoutingBlockage,
    kPlacementBlockage,
    kMax
  };

  // getter
  const IdbBlockageType get_type() const { return _type; }
  const std::string get_instance_name() const { return _instance_name; }
  IdbInstance* get_instance() { return _instance; }
  const bool is_pushdown() const { return _is_pushdown; }
  const bool is_routing_blockage() { return _type == IdbBlockageType::kRoutingBlockage ? true : false; }
  const bool is_palcement_blockage() { return _type == IdbBlockageType::kPlacementBlockage ? true : false; }

  std::vector<IdbRect*> get_rect_list() { return _rect_list; }
  int32_t get_rect_num() const { return _rect_list.size(); }
  IdbRect* get_rect(size_t index);

  // setter
  void set_type_routing() { _type = IdbBlockageType::kRoutingBlockage; }
  void set_type_placement() { _type = IdbBlockageType::kPlacementBlockage; }
  void set_instance_name(std::string name) { _instance_name = name; }
  void set_instance(IdbInstance* instance) { _instance = instance; }
  void set_pushdown(bool value) { _is_pushdown = value; }

  void set_rect_list(std::vector<IdbRect*> rect_list) { _rect_list = rect_list; }
  IdbRect* add_rect();
  void add_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y);
  void reset_rect();

  virtual void function(){};
  // function

 private:
  std::string _instance_name;
  IdbInstance* _instance;
  bool _is_pushdown;
  IdbBlockageType _type;
  std::vector<IdbRect*> _rect_list;
};

class IdbRoutingBlockage : public IdbBlockage
{
 public:
  IdbRoutingBlockage();
  ~IdbRoutingBlockage();

  // getter
  const std::string get_layer_name() const { return _layer_name; }
  IdbLayer* get_layer() { return _layer; }
  const bool is_slots() const { return _is_slots; }
  const bool is_fills() const { return _is_fills; }
  const bool is_except_pgnet() const { return _is_except_pgnet; }
  const int32_t get_min_spacing() const { return _min_spacing; }
  const int32_t get_effective_width() const { return _effective_width; }

  // setter
  void set_layer_name(std::string name) { _layer_name = name; }
  void set_layer(IdbLayer* layer) { _layer = layer; }
  void set_slots(bool value) { _is_slots = value; }
  void set_fills(bool value) { _is_fills = value; }
  void set_except_pgnet(bool value) { _is_except_pgnet = value; }
  void set_min_spacing(int32_t spacing) { _min_spacing = spacing; }
  void set_effective_width(int32_t width) { _effective_width = width; }

 private:
  std::string _layer_name;
  IdbLayer* _layer;
  int32_t _min_spacing;
  int32_t _effective_width;
  bool _is_slots;
  bool _is_fills;
  /*Indicates that the blockage only blocks signal net routing, and does not block power or ground net routing.*/
  bool _is_except_pgnet;
};

class IdbPlacementBlockage : public IdbBlockage
{
 public:
  IdbPlacementBlockage();
  ~IdbPlacementBlockage();

  // getter
  const bool is_soft() const { return _is_soft; }
  const bool is_partial() const { return _is_partial; }
  const double get_max_density() const { return _max_density; }
  IdbLayer* get_layer();

  // setter
  void set_soft(bool value) { _is_soft = value; }
  void set_fills(bool value) { _is_partial = value; }
  void set_max_density(double max_density) { _max_density = max_density; }

 private:
  double _max_density;
  bool _is_soft;
  bool _is_partial;
};

class IdbBlockageList
{
 public:
  IdbBlockageList();
  ~IdbBlockageList();

  // getter
  const int32_t get_num() const { return _blockage_list.size(); };
  std::vector<IdbBlockage*> get_blockage_list() { return _blockage_list; }

  // vector<IdbBlockage*> find_routing_blockage(string layer_name);
  // IdbBlockage* get_blockage(size_t index);

  // setter
  IdbRoutingBlockage* add_blockage_routing(std::string layer_name);
  IdbPlacementBlockage* add_blockage_placement();
  void removeExceptPgNetBlockageList();

  void reset();
  void clearPlacementBlockage();
  void clearRoutingBlockage();

  // operator
  // void init_blockage_list(int32_t size){_blockage_list.reserve(size);}

 private:
  int32_t _num;
  std::vector<IdbBlockage*> _blockage_list;
};

}  // namespace idb
