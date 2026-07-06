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

#include <cstdint>
#include <map>
#include <string>

#include "Layer.hpp"
#include "VariationParams.hpp"
namespace itf
{

class ProcessCorner {
 public:
  // constructor
  ProcessCorner();
  ~ProcessCorner();

  // getter
  std::string get_technology() const;
  double get_global_temperature() const;
  double get_background_er() const;
  double get_half_node_scale_factor() const;
  bool get_use_si_density() const;
  double get_drop_factor_lateral_spacing() const;
  Layers* get_layers();
  const Layers* get_layers() const;
  VariationParams* get_variation_params();
  const VariationParams* get_variation_params() const;

  // setter
  void set_technology(std::string);
  void set_global_temperature(double);
  void set_background_er(double);
  void set_half_node_scale_factor(double);
  void set_use_si_density(bool);
  void set_drop_factor_lateral_spacing(double);
  void set_layers_map(const std::map<std::string, uint8_t>&);

  // function
  void update_layers_height();
  void show_layers() const;

 private:
  // function
  void update_dielectric_layer_height(LayerDielectric*, LayerDielectric*);
  void update_conductor_layer_height(LayerConductor*, LayerDielectric*);

  // members
  std::string _technology;
  double _global_temperature;
  double _background_er;
  double _half_node_scale_factor;
  bool _use_si_density;
  double _drop_factor_lateral_spacing;
  Layers* _layers;
  VariationParams* _variation_params;
};

} // namespace itf
