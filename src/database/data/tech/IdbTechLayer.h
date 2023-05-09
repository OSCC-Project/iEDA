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
#ifndef _LAYER_H_
#define _LAYER_H_

#include <iostream>
#include <memory>
#include <string>

#include "../design/IdbEnum.h"
#include "IdbAntennaCheck.h"
#include "IdbArraySpacingCheck.h"
#include "IdbCutSpacingCheck.h"
#include "IdbDensityCheck.h"
#include "IdbEnclosurecheck.h"
#include "IdbMaxWidthCheck.h"
#include "IdbMinAreaCheck.h"
#include "IdbMinEnclosedAreaCheck.h"
#include "IdbMinStepCheck.h"
#include "IdbMinWidthCheck.h"
#include "IdbMinimumCutCheck.h"
#include "IdbSpacingCheck.h"
#include "IdbSpacingTableCheck.h"
#include "IdbTechEnum.h"

namespace idb {
  class IdbTechLayer {
   public:
    IdbTechLayer();
    virtual ~IdbTechLayer();

    // setter
    void add_antenna_check(std::unique_ptr<IdbAntennaCheck> &check) { _antenna_checks.push_back(std::move(check)); }
    void set_type(LayerTypeEnum type) { _type = type; }
    void set_layer_id(int layerId) { _layer_id = layerId; }
    void set_name(const std::string &name) { _name = name; }

    // getter
    IdbAntennaCheck *get_antenna_check() { return (_antenna_checks.begin())->get(); }
    LayerTypeEnum get_type() const { return _type; }
    int get_layer_id() const { return _layer_id; }
    const std::string &get_name() const { return _name; }

   private:
    LayerTypeEnum _type;
    int _layer_id;
    std::string _name;
    std::vector<std::unique_ptr<IdbAntennaCheck>> _antenna_checks;
  };

  class IdbTechRoutingLayer : public IdbTechLayer {
   public:
    IdbTechRoutingLayer();
    ~IdbTechRoutingLayer();
    // setter
    void set_direction(LayerDirEnum dir) { _direction = dir; };
    void set_direction(IdbLayerDirection dir);
    void set_pitch(int pitch) { _pitch = pitch; }
    void set_offset(int offset) { _offset = offset; }
    void set_thickness(int thickness) { _thickness = thickness; }
    void set_resistance(double resistance) { _resistance = resistance; }
    void set_capacitance(double capacitance) { _capacitance = capacitance; }
    void set_edge_capcitance(double edge_capcitance) { _edge_capcitance = edge_capcitance; }
    void set_width(int width) { _width = width; }
    void set_max_width_check(std::unique_ptr<IdbMaxWidthCheck> &check) { _max_width_check = std::move(check); }
    void set_min_width_check(std::unique_ptr<IdbMinWidthCheck> &check) { _min_width_check = std::move(check); }
    void set_min_area_check(std::unique_ptr<IdbMinAreaCheck> &check) { _min_area_check = std::move(check); }
    void set_density_check(std::unique_ptr<IdbDensityCheck> &check) { _density_check = std::move(check); }
    void set_min_step_check(std::unique_ptr<IdbMinStepCheck> &check) { _min_step_check = std::move(check); }
    void set_spacing_table_prl_Check(std::unique_ptr<IdbSpacingTableParallelRunLengthCheck> &check) {
      _spacing_table_prl_Check = std::move(check);
    }
    void set_spacing_table_two_widtth_check(std::unique_ptr<IdbSpacingTableTwoWidthCheck> &check) {
      _spacing_table_two_widtth_check = std::move(check);
    }
    void set_spacing_check(std::unique_ptr<IdbSpacingCheck> &check) { _spacing_check = std::move(check); }
    void set_min_enclosed_area_check(std::unique_ptr<IdbMinEnclosedAreaCheck> &check) {
      _min_enclosed_area_check = std::move(check);
    }
    void set_minimum_cut_check(std::unique_ptr<IdbMinimumCutCheck> &check) { _minimum_cut_check = std::move(check); }

    // getter
    LayerDirEnum &get_direction() { return _direction; }
    int get_width() { return _width; }
    int get_pitch() const { return _pitch; }
    int get_offset() const { return _offset; }
    int get_thickness() const { return _thickness; }
    double get_resistance() const { return _resistance; }
    double get_capacitance() const { return _capacitance; }
    double get_edge_capcitance() const { return _edge_capcitance; }
    IdbMaxWidthCheck *get_min_width_check() { return _max_width_check.get(); }
    // other
    void addSpacingRangeCheck(std::unique_ptr<IdbSpacingRangeCheck> &check) {
      _spacing_range_check_list->addSpacingRangeCheck(check);
    }
    void addSpacingEolCheck(std::unique_ptr<IdbSpacingEolCheck> &check) {
      _spacing_eol_check_list->addSpacingEolCheck(check);
    }
    void addSpacingSamenetCheck(std::unique_ptr<IdbSpacingSamenetCheck> &check) {
      _spacing_samenet_check_list->addSpacingSamenetCheck(check);
    }
    // interface
    std::string getRoutingLayerName() { return get_name(); }
    bool isHoritional() { return _direction == LayerDirEnum::kHORITIONAL; }
    bool isVertical() { return _direction == LayerDirEnum::kVERTICAL; }
    int getPitch() { return get_pitch(); }
    int getOffset() { return get_offset(); }
    int getWidthDefault() { return get_width(); }
    int getRoutingSpacing() { return _spacing_check->get_min_spacing(); }
    int getMaxWidth() { return _max_width_check->get_max_width(); }
    int getMinWidth() { return _min_width_check->get_min_width(); }
    std::vector<IdbSpacingRangeCheck *> getSpacingRangeCheckList();
    int getRoutingSpacing(int width);
    int getMinArea() { return _min_area_check->get_min_area(); }
    int getMinEnclosedArea() { return _min_enclosed_area_check->get_area(); }

    int getMinimumDensity() { return _density_check->get_min_density(); }
    int getMaximumDensity() { return _density_check->get_max_density(); }
    int getDensityCheckWindowWidth() { return _density_check->get_density_check_width(); }
    int getDensityCheckWindowLength() { return _density_check->get_density_check_length(); }
    int getDensityCheckWindowStep() { return _density_check->get_density_check_step(); }

    int getMinimumCutNum() { return _minimum_cut_check->get_num_cuts(); }
    int getMiniMumCutWidth() { return _minimum_cut_check->get_width(); }
    MinimumcutConnectionEnum getConnectionFrom() { return _minimum_cut_check->get_connection(); }

   private:
    LayerDirEnum _direction;
    int _pitch;
    int _offset;
    int _thickness;
    double _resistance;
    double _capacitance;
    double _edge_capcitance;
    int _width;

    std::unique_ptr<IdbMaxWidthCheck> _max_width_check;
    std::unique_ptr<IdbMinWidthCheck> _min_width_check;
    std::unique_ptr<IdbMinAreaCheck> _min_area_check;
    std::unique_ptr<IdbDensityCheck> _density_check;
    // only one minStep rule should be defined for a given layer. Only the last one is checked
    std::unique_ptr<IdbMinStepCheck> _min_step_check;
    // only one parallel run length and one influence spacing table for a layer
    std::unique_ptr<IdbSpacingTableParallelRunLengthCheck> _spacing_table_prl_Check;
    std::unique_ptr<IdbSpacingTableTwoWidthCheck> _spacing_table_two_widtth_check;
    // spacing rule
    std::unique_ptr<IdbSpacingCheck> _spacing_check;
    std::unique_ptr<IdbSpacingRangeCheckList> _spacing_range_check_list;
    std::unique_ptr<IdbSpacingEolCheckList> _spacing_eol_check_list;
    std::unique_ptr<IdbSpacingSamenetCheckList> _spacing_samenet_check_list;
    // MinEnclosedAreaCheck
    std::unique_ptr<IdbMinEnclosedAreaCheck> _min_enclosed_area_check;
    // MinimumCutCheck
    std::unique_ptr<IdbMinimumCutCheck> _minimum_cut_check;
  };

  class IdbTechCutLayer : public IdbTechLayer {
   public:
    IdbTechCutLayer();
    ~IdbTechCutLayer();
    // setter
    void set_width(int width) { _width = width; }
    void set_array_spacing_check(std::unique_ptr<IdbArraySpacingCheck> &check) { _array_spacing_check = std::move(check); }
    void set_enclosure_check(std::unique_ptr<IdbEnclosureCheck> &check) { _enclosure_check = std::move(check); }
    // getter
    int get_width() const { return _width; }
    // other
    void set_cut_spacing_check(std::unique_ptr<IdbCutSpacingCheck> &check) { _cut_spacing_check = std::move(check); }
    void addDccurrentDensityCheck(std::unique_ptr<IdbDccurrentDensityCheck> &check) {
      _dccurrent_density_check_list->addDccurrentDensityCheck(check);
    }
    // interface
    std::string getCutLayerName() { return get_name(); }
    int getCutSpacing() { return _cut_spacing_check->get_cut_spacing(); }
    int getCutWidth() { return get_width(); }

    int getArrayWithin() { return _array_spacing_check->get_cut_spacing(); }
    int getArrayCuts() { return _array_spacing_check->getArrayCuts(); }
    int getArraySpacing() { return _array_spacing_check->getArraySpacing(); }

    int getBelowEnclosureX() { return _enclosure_check->getBelowOverhang1(); }
    int getBelowEnclosureY() { return _enclosure_check->getBelowOverhang2(); }
    int getAboveEnclosureX() { return _enclosure_check->getAboveOverhang1(); }
    int getAboveEnclosureY() { return _enclosure_check->getAboveOverhang2(); }

   private:
    int _width;
    std::unique_ptr<IdbArraySpacingCheck> _array_spacing_check;
    std::unique_ptr<IdbEnclosureCheck> _enclosure_check;
    std::unique_ptr<IdbCutSpacingCheck> _cut_spacing_check;
    std::unique_ptr<IdbDccurrentDensityCheckList> _dccurrent_density_check_list;
  };

  class IdbTechLayerList {
   public:
    IdbTechLayerList() { }
    ~IdbTechLayerList() { }

    void addLayer(std::unique_ptr<IdbTechLayer> layer) { _tech_layers.push_back(std::move(layer)); }

   private:
    std::vector<std::unique_ptr<IdbTechLayer>> _tech_layers;
  };

  class IdbTechRoutingLayerList {
   public:
    IdbTechRoutingLayerList() { }
    ~IdbTechRoutingLayerList() { }

    IdbTechRoutingLayer *findRoutingLayer(const std::string &name);
    IdbTechRoutingLayer *findRoutingLayer(int layerId);
    void initRoutingLayerId();
    void printRoutingLayer();

    void addTechRoutingLayer(std::unique_ptr<IdbTechRoutingLayer> &layer) {
      _tech_routing_layers.push_back(std::move(layer));
    }

   private:
    std::vector<std::unique_ptr<IdbTechRoutingLayer>> _tech_routing_layers;
  };

  class IdbTechCutLayerList {
   public:
    IdbTechCutLayerList() { }
    ~IdbTechCutLayerList() { }

    IdbTechCutLayer *findCutLayer(const std::string &name);
    IdbTechCutLayer *findCutLayer(int layerId);
    void initCutLayerId();
    void printCutLayer();

    void addTechCutLayer(std::unique_ptr<IdbTechCutLayer> &layer) { _tech_cut_layers.push_back(std::move(layer)); }

   private:
    std::vector<std::unique_ptr<IdbTechCutLayer>> _tech_cut_layers;
  };

}  // namespace idb

#endif
