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

#include <string>
#include <vector>

#include "DensityRule.h"
#include "DrcEnum.h"
#include "DrcRect.h"
#include "DrcRules.hpp"
#include "IdbLayer.h"
#include "SpacingRangeRule.h"

namespace idrc {
class DrcLayer
{
 public:
  DrcLayer() : _layer_id(-1), _name(""), _layer_type(LayerType::kNone) {}
  virtual ~DrcLayer() {}
  // setter
  void set_name(const std::string& name) { _name = name; }
  void set_layer_type(const LayerType& type) { _layer_type = type; }
  void set_layer_id(const int layerId) { _layer_id = layerId; }
  // getter
  const std::string& get_name() const { return _name; }
  LayerType get_layer_type() const { return _layer_type; }
  int get_layer_id() const { return _layer_id; }

 private:
  int _layer_id;
  std::string _name;
  LayerType _layer_type;
};

class DrcRoutingLayer : public DrcLayer
{
 public:
  DrcRoutingLayer()
      : _default_width(-1),
        _min_area(-1),
        _min_width(-1),
        _min_spacing(-1),
        _min_enclosed_area(-1),
        _direction(LayerDirection::kNone),
        _density_rule(nullptr)
  {
  }
  ~DrcRoutingLayer()
  {
    clear_density_rule();
    clear_spacing_range_rule_list();
  }

  // setter
  void set_default_width(int width) { _default_width = width; }
  void set_min_area(int area) { _min_area = area; }
  void set_min_width(int width) { _min_width = width; }
  void set_min_spacing(int spacing) { _min_spacing = spacing; }
  void set_min_enclosed_area(int min_enclosed_area) { _min_enclosed_area = min_enclosed_area; }
  void set_direction(LayerDirection direction) { _direction = direction; }
  void set_density_rule(DensityRule* density_rule) { _density_rule = density_rule; }
  void set_lef58_eol_spacing_rule_list(const std::vector<std::shared_ptr<idb::routinglayer::Lef58SpacingEol>>& in)
  {
    _lef58_eol_spacing_rule_list = in;
  }
  void set_spacing_table(const std::shared_ptr<idb::IdbLayerSpacingTable> in) { _spacing_table = in; }
  void set_notch_spacing_rule(const idb::IdbLayerSpacingNotchLength& in)
  {
    _notch_spacing_rule.set_min_spacing(in.get_min_spacing());
    _notch_spacing_rule.set_notch_length(in.get_notch_length());
  }
  void set_lef58_notch_spacing_rule(const std::shared_ptr<idb::routinglayer::Lef58SpacingNotchlength> in)
  {
    _lef58_notch_spacing_rule = in;
  }
  void set_lef58_area_rule_list(const std::vector<std::shared_ptr<idb::routinglayer::Lef58Area>>& in) { _lef58_area_rule_list = in; }
  void set_lef58_min_step_rule(const std::vector<std::shared_ptr<idb::routinglayer::Lef58MinStep>>& in) { _lef58_min_step_rule = in; }
  void set_min_step_rule(const std::shared_ptr<idb::IdbMinStep> in) { _min_step_rule = in; }
  void set_lef58_jog_spacing_rule(const std::shared_ptr<idb::routinglayer::Lef58SpacingTableJogToJog> in) { _lef58_jog_spacing_rule = in; }
  void set_lef58_corner_fill_spacing_rule(const std::shared_ptr<idb::routinglayer::Lef58CornerFillSpacing> in)
  {
    _lef58_corner_fill_spacing_rule = in;
  }

  // SpacingRangeRule* add_spacing_range_rule();
  // getter
  int get_default_width() const { return _default_width; }
  int get_min_area() const { return _min_area; }
  int get_min_width() const { return _min_width; }
  int get_min_spacing() const { return _min_spacing; }
  int get_min_enclosed_area() const { return _min_enclosed_area; }
  LayerDirection get_direction() const { return _direction; }
  std::vector<SpacingRangeRule*>& get_spacing_range_rule_list() { return _spacing_range_rule_list; }
  // std::vector<MinimumCutRule*>& getMinimumCutRuleList() { return _minimum_cut_rule_list; }
  std::vector<std::shared_ptr<idb::routinglayer::Lef58SpacingEol>>& get_lef58_eol_spacing_rule_list()
  {
    return _lef58_eol_spacing_rule_list;
  }
  std::shared_ptr<idb::IdbLayerSpacingTable> get_spacing_table() { return _spacing_table; }
  idb::IdbLayerSpacingNotchLength& get_notch_spacing_rule() { return _notch_spacing_rule; }
  std::shared_ptr<idb::routinglayer::Lef58SpacingNotchlength> get_lef58_notch_spacing_rule() { return _lef58_notch_spacing_rule; }
  std::vector<std::shared_ptr<idb::routinglayer::Lef58Area>>& get_lef58_area_rule_list() { return _lef58_area_rule_list; }

  std::shared_ptr<idb::IdbMinStep> get_min_step_rule() { return _min_step_rule; }
  std::vector<std::shared_ptr<idb::routinglayer::Lef58MinStep>> get_lef58_min_step_rule() { return _lef58_min_step_rule; }
  std::shared_ptr<idb::routinglayer::Lef58SpacingTableJogToJog> get_lef58_jog_spacing_rule() { return _lef58_jog_spacing_rule; }
  std::shared_ptr<idb::routinglayer::Lef58CornerFillSpacing> get_lef58_corner_fill_spacing_rule()
  {
    return _lef58_corner_fill_spacing_rule;
  }

  // function
  int getRoutingSpacing(int width);
  int getLayerMaxRequireSpacing(DrcRect* target_rect);
  bool isSpacingTable() { return _spacing_table->get_parallel() != nullptr; }

  void clear_density_rule();
  void clear_spacing_range_rule_list();

 private:
  int _default_width;
  int _min_area;
  int _min_width;
  int _min_spacing;
  int _min_enclosed_area;
  LayerDirection _direction;
  DensityRule* _density_rule;
  std::vector<SpacingRangeRule*> _spacing_range_rule_list;
  // std::vector<MinimumCutRule*> _minimum_cut_rule_list;
  std::vector<std::shared_ptr<idb::routinglayer::Lef58SpacingEol>> _lef58_eol_spacing_rule_list;
  std::shared_ptr<idb::IdbLayerSpacingTable> _spacing_table;
  idb::IdbLayerSpacingNotchLength _notch_spacing_rule;
  std::shared_ptr<idb::routinglayer::Lef58SpacingNotchlength> _lef58_notch_spacing_rule;
  std::vector<std::shared_ptr<idb::routinglayer::Lef58Area>> _lef58_area_rule_list;
  std::shared_ptr<idb::IdbMinStep> _min_step_rule;
  std::vector<std::shared_ptr<idb::routinglayer::Lef58MinStep>> _lef58_min_step_rule;
  std::shared_ptr<idb::routinglayer::Lef58SpacingTableJogToJog> _lef58_jog_spacing_rule;
  std::shared_ptr<idb::routinglayer::Lef58CornerFillSpacing> _lef58_corner_fill_spacing_rule;
};

class DrcCutLayer : public DrcLayer
{
 public:
  DrcCutLayer() : _cut_spacing(-1), _width(-1) {}
  ~DrcCutLayer() {}

  // setter
  void set_cut_spacing(int cut_spacing) { _cut_spacing = cut_spacing; }
  void set_default_width(int in) { _width = in; }
  void setLEF58EnclosureRuleList(std::shared_ptr<std::vector<idb::cutlayer::Lef58Enclosure>> in) { _lef58_enclosure_rule_list = in; }

  void set_lef58_spacing_table_list(std::vector<std::shared_ptr<idb::cutlayer::Lef58SpacingTable>>& in) { _lef58_spacing_table_list = in; }
  void set_lef58_cut_class_list(std::vector<std::shared_ptr<idb::cutlayer::Lef58Cutclass>>& in) { _lef58_cut_class_list = in; }
  void set_lef58_cut_eol_spacing(std::shared_ptr<idb::cutlayer::Lef58EolSpacing> in) { _lef58_cut_eol_spacing = in; }
  void set_lef58_enclosure_list(std::vector<std::shared_ptr<idb::cutlayer::Lef58Enclosure>>& in) { _lef58_enclosure_list = in; }
  void set_lef58_enclosure_edge_list(std::vector<std::shared_ptr<idb::cutlayer::Lef58EnclosureEdge>>& in)
  {
    _lef58_enclosure_edge_list = in;
  }
  // getter
  int get_cut_spacing() const { return _cut_spacing; }
  int get_default_width() const { return _width; }

  std::vector<std::shared_ptr<idb::cutlayer::Lef58SpacingTable>>& get_lef58_spacing_table_list() { return _lef58_spacing_table_list; }
  std::vector<std::shared_ptr<idb::cutlayer::Lef58Cutclass>>& get_lef58_cut_class_list() { return _lef58_cut_class_list; }
  std::shared_ptr<idb::cutlayer::Lef58EolSpacing> get_lef58_cut_eol_spacing() { return _lef58_cut_eol_spacing; }
  std::vector<std::shared_ptr<idb::cutlayer::Lef58Enclosure>>& get_lef58_enclosure_list() { return _lef58_enclosure_list; }
  std::vector<std::shared_ptr<idb::cutlayer::Lef58EnclosureEdge>>& get_lef58_enclosure_edge_list() { return _lef58_enclosure_edge_list; }

  // std::vector<EnclosureRule*>& getEnclosureRuleList() { return _enclosure_rule_list; }
  // std::vector<EnclosureRule*>& getBelowEnclosureRuleList() { return _below_enclosure_rule_list; }
  // std::vector<EnclosureRule*>& getAboveEnclosureRuleList() { return _above_enclosure_rule_list; }

 private:
  int _cut_spacing;
  int _width;
  std::vector<std::shared_ptr<idb::cutlayer::Lef58SpacingTable>> _lef58_spacing_table_list;
  std::vector<std::shared_ptr<idb::cutlayer::Lef58Cutclass>> _lef58_cut_class_list;
  std::shared_ptr<idb::cutlayer::Lef58EolSpacing> _lef58_cut_eol_spacing;
  std::vector<std::shared_ptr<idb::cutlayer::Lef58Enclosure>> _lef58_enclosure_list;
  std::vector<std::shared_ptr<idb::cutlayer::Lef58EnclosureEdge>> _lef58_enclosure_edge_list;
  // std::vector<EnclosureRule*> _enclosure_rule_list;
  // std::vector<EnclosureRule*> _below_enclosure_rule_list;
  // std::vector<EnclosureRule*> _above_enclosure_rule_list;
  std::shared_ptr<std::vector<idb::cutlayer::Lef58Enclosure>> _lef58_enclosure_rule_list;
};
}  // namespace idrc
