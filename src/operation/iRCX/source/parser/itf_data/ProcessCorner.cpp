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
#include <iomanip>
#include <string.h>

#include "ProcessCorner.hpp"
#include "magic_enum/magic_enum.hpp"
namespace itf
{

ProcessCorner::ProcessCorner()
: _technology()
, _global_temperature(25)
, _background_er(1)
, _half_node_scale_factor(1)
, _use_si_density()
, _drop_factor_lateral_spacing(.5)
, _layers(nullptr)
, _variation_params(nullptr)
{
  _layers = new Layers();
  _variation_params = new VariationParams();
}

ProcessCorner::~ProcessCorner()
{
  delete _layers;
  delete _variation_params;
}

std::string
ProcessCorner::get_technology() const
{
  return _technology;
}

double
ProcessCorner::get_global_temperature() const
{
  return _global_temperature;
}

double
ProcessCorner::get_background_er() const
{
  return _background_er;
}

double
ProcessCorner::get_half_node_scale_factor() const
{
  return _half_node_scale_factor;
}

bool
ProcessCorner::get_use_si_density() const
{
  return _use_si_density;
}

double
ProcessCorner::get_drop_factor_lateral_spacing() const
{
  return _drop_factor_lateral_spacing;
}

Layers*
ProcessCorner::get_layers()
{
  return _layers;
}

const Layers*
ProcessCorner::get_layers() const
{
  return _layers;
}

VariationParams*
ProcessCorner::get_variation_params()
{
  return _variation_params;
}

const VariationParams*
ProcessCorner::get_variation_params() const
{
  return _variation_params;
}

void
ProcessCorner::set_technology(const std::string v)
{
  _technology = v;
}
void
ProcessCorner::set_global_temperature(double v)
{
  _global_temperature = v;
}
void
ProcessCorner::set_background_er(double v)
{
  _background_er = v;
}
void
ProcessCorner::set_half_node_scale_factor(double v)
{
  _half_node_scale_factor = v;
}
void
ProcessCorner::set_use_si_density(bool v)
{
  _use_si_density = v;
}
void
ProcessCorner::set_drop_factor_lateral_spacing(double v)
{
  _drop_factor_lateral_spacing = v;
}

// @param name_order_map itf layer name -> layer order
void
ProcessCorner::set_layers_map(const std::map<std::string, uint8_t>& name_order_map)
{
  for (const auto& [name, layer_order] : name_order_map) {
    auto layer = _layers->find_layer(name);
    if (layer) {
      layer->set_order(layer_order);
    }
  }
}

void
ProcessCorner::update_layers_height()
{
  std::vector<Layer*> stack_layers;
  for (auto layer : _layers->get_layers()) {
    if (layer->get_type() == LayerType::kVia) continue;
    stack_layers.push_back(layer);
  }
  std::sort(stack_layers.begin(), stack_layers.end(), [](Layer* high, Layer* low){
    return high->get_id() > low->get_id();
  });

  LayerDielectric* last_diel = nullptr;
  for (auto layer : stack_layers) {
    switch (layer->get_type()) {
      case LayerType::kConductor:
        update_conductor_layer_height(dynamic_cast<LayerConductor*>(layer), last_diel);
        break;
      case LayerType::kDielectric:
        update_dielectric_layer_height(last_diel, dynamic_cast<LayerDielectric*>(layer));
        last_diel = dynamic_cast<LayerDielectric*>(layer);
        break;
      default:
        std::cout << "Unhandled layer type: " << magic_enum::enum_name(layer->get_type()) << std::endl;
        break;
    }
  }

  // via layer
  for (auto via : _layers->get_via_layers()) {
    auto bot = _layers->find_layer(via->get_from());
    if (bot) {
      via->set_height(bot->get_height() + bot->get_layer_thickness());
    } else if (strncasecmp("SUBSTRATE", via->get_from(), 9)) {
      std::cout << "fail to find layer: " << via->get_from() << std::endl;   
    }

    auto top = _layers->find_layer(via->get_to());
    if (top) {
      via->set_top_height(top->get_height());
    } else {
      std::cout << "fail to find layer: " << via->get_to() << std::endl;   
    }
  }

  // debug
  // show_layers();  
}

void
ProcessCorner::update_dielectric_layer_height(LayerDielectric* last_diel, LayerDielectric* cur_diel)
{
  if (!cur_diel) return;

  if (cur_diel->has_associated_conductor()) {
    auto cdt = _layers->find_conductor_layer(cur_diel->get_associated_conductor());
    if (cdt) {
      cur_diel->set_height(cdt->get_height());
    }
  } else if (cur_diel->has_measured_from()) {
    auto layer = _layers->find_layer(cur_diel->get_measured_from());
    if (layer) {
      cur_diel->set_height(layer->get_height() + layer->get_layer_thickness());
    } else if (strncasecmp("TOP_OF_CHIP", cur_diel->get_measured_from(), 11) == 0 && last_diel) {
      cur_diel->set_height(last_diel->get_height() + last_diel->get_layer_thickness());  
    }
  } else if (last_diel) {
    cur_diel->set_height(last_diel->get_height() + last_diel->get_layer_thickness());
  } else if (_layers->is_lowermost_diel(cur_diel->get_id())) {
    return;
  } else {
    std::cout << "fail to update layer height at: " << cur_diel->get_name() << std::endl;
  }
}

void
ProcessCorner::update_conductor_layer_height(LayerConductor* cdt, LayerDielectric* last_diel)
{
  if (!cdt) return;

  char* measured = cdt->get_measured_from();
  LayerDielectric* diel = nullptr;
  if (measured) {
    diel = _layers->find_diel(measured);
  } else {
    diel = last_diel;
  }

  if (diel) {
    cdt->set_height(diel->get_height() + diel->get_thickness());
  } else {
    std:: cout << "fail to find diel, cdt_id = " << std::to_string(cdt->get_id()) << std::endl;
  }
}

void
ProcessCorner::show_layers() const
{
  auto show_er = [](Layer* layer) -> std::string {
    if (layer->get_type() != LayerType::kDielectric) return "";
    else return "| " + std::to_string(dynamic_cast<LayerDielectric*>(layer)->get_er());
  };
  auto show_layer_type = [](Layer* layer) -> std::string_view {
    return magic_enum::enum_name(layer->get_type());
  };
  std::cout << "  layer type   | layer name               | height | thickness | ER" << std::endl;
  for (auto layer : _layers->get_layers()) {
    std::cout 
      << std::left << std::setw(15) << show_layer_type(layer) << "| "
      << std::left << std::setw(25) << layer->get_name()  << "| " 
      << std::setw(7) << layer->get_height() << "| "
      << std::setw(10) << layer->get_layer_thickness()
      << show_er(layer) << ""
      << std::endl; 
  }
}

} // namespace itf
