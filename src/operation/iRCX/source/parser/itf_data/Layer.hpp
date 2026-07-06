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
#include <optional>
#include <string>
#include <vector>

#include "itfiConductor.hpp"
#include "itfiDielectric.hpp"
#include "itfiVia.hpp"
namespace itf
{


enum class LayerType { kDielectric, kConductor, kVia };

class Layer {
 public:
  // constructor
  explicit Layer(LayerType);
  virtual ~Layer() = default;

  // getter
  LayerType get_type() const;
  int16_t get_id() const;
  std::optional<uint8_t> get_order() const;
  double get_height() const;
  virtual std::string get_name() const = 0;
  virtual double get_layer_thickness() const = 0;

  // setter
  void set_id(int16_t);
  void set_order(uint8_t);
  void set_height(double);

  // function

 private:
  LayerType _type;
  int16_t _id; // refer to the order in .itf file
  std::optional<uint8_t> _order; // refer to the order in .tlef file
  double _height; // Units: micron
};

class LayerConductor : public Layer, public itfiConductor {
 public:
  // constructor
  LayerConductor();
  explicit LayerConductor(const itfiConductor&);
  ~LayerConductor();

  // getter
  virtual std::string get_name() const override;
  virtual double get_layer_thickness() const override;
  int32_t get_width() const;

  // setter
  void set_width(int32_t);

  // function
  void query_crt_vs_si_width(double, std::optional<double>&, std::optional<double>&) const;

 private:
  // members
  int32_t _width;
};

class LayerDielectric : public Layer, public itfiDielectric {
 public:
  // constructor
  LayerDielectric();
  explicit LayerDielectric(const itfiDielectric&);
  ~LayerDielectric();

  // getter
  virtual std::string get_name() const override;
  virtual double get_layer_thickness() const override;
  
  // setter

  // function

 private:
  // members
};

class LayerVia : public Layer, public itfiVia {
 public:
  // constructor
  LayerVia();
  explicit LayerVia(const itfiVia&);
  ~LayerVia();

  // getter
  virtual std::string get_name() const override;
  virtual double get_layer_thickness() const override;
  double get_bot_height() const;
  double get_top_height() const;
  
  // setter
  void set_top_height(double);

  // function
  double query_rpv_vs_area(double) const;
  void query_crt_vs_area(double, std::optional<double>&, std::optional<double>&) const;
  
 private:
  // members
  double _top_height;
};

class Layers {
 public:
  // constructor
  Layers() = default;
  ~Layers();

  // getter
  std::vector<Layer*>& get_layers();
  const std::vector<Layer*>& get_layers() const;
  std::vector<LayerConductor*>& get_conductor_layers();
  const std::vector<LayerConductor*>& get_conductor_layers() const;
  std::vector<LayerDielectric*>& get_dielectric_layers();
  const std::vector<LayerDielectric*>& get_dielectric_layers() const;
  std::vector<LayerVia*>& get_via_layers();
  const std::vector<LayerVia*>& get_via_layers() const;
  Layer* get_uppermost_layer_by_order() const;
  Layer* get_lowermost_layer_by_order() const;
  
  // setter
  void add_conductor_layer(LayerConductor*);
  void add_dielectric_layer(LayerDielectric*);
  void add_via_layer(LayerVia*);

  // function
  void clear();
  Layer* find_layer(uint8_t) const;
  Layer* find_layer(const std::string&) const;
  LayerConductor* find_conductor_layer(uint8_t) const;
  LayerConductor* find_conductor_layer(const char*) const;
  LayerVia* find_via_layer(uint8_t) const;
  LayerDielectric* find_diel_below(LayerConductor*) const;
  LayerDielectric* find_diel_below(double) const;
  LayerDielectric* find_diel_above(double) const;
  LayerDielectric* find_diel(int16_t) const;
  LayerDielectric* find_diel(const char*) const;
  bool is_lowermost_diel(int16_t) const;

 private:
  // members
  std::vector<Layer*> _layers;
  std::vector<LayerConductor*> _conductor_layers;
  std::vector<LayerDielectric*> _dielectric_layers;
  std::vector<LayerVia*> _via_layers;
};

} // namespace itf
