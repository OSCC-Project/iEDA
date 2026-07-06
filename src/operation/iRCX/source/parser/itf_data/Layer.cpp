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
#include "Layer.hpp"
namespace itf
{

Layer::Layer(LayerType t)
: _type(t)
, _id(0)
, _order(std::nullopt)
, _height(0)
{ }

LayerType
Layer::get_type() const
{
  return _type;
}

int16_t
Layer::get_id() const
{
  return _id;
}

std::optional<uint8_t>
Layer::get_order() const
{
  return _order;
}

double
Layer::get_height() const 
{
  return _height;
}

void
Layer::set_id(int16_t id)
{
  _id = id;
}

void
Layer::set_order(uint8_t order)
{
  _order = order;
}

void
Layer::set_height(double height)
{
  _height = height;
}

LayerConductor::LayerConductor()
: Layer(LayerType::kConductor)
, _width(0)
{

}

LayerConductor::LayerConductor(const itfiConductor& cdt)
: Layer(LayerType::kConductor)
, _width(0)
{
  itfiConductor::operator=(cdt);
}

LayerConductor::~LayerConductor()
{ }

std::string
LayerConductor::get_name() const
{
  return get_conductor_name();
}

// Units: microns
double
LayerConductor::get_layer_thickness() const
{
  return get_thickness();
}

int32_t
LayerConductor::get_width() const
{
  return _width;
}

void
LayerConductor::set_width(int32_t w)
{
  _width = w;
}

// @param width Conductor silicon (post-etch) widths
// @param crt1 output Linear temperature coefficients
// @param crt2 output Quadratic temperature coefficients 
void
LayerConductor::query_crt_vs_si_width(
  double width,
  std::optional<double>& crt1,
  std::optional<double>& crt2
) const {
  crt1.reset();
  crt2.reset();

  auto& list = get_crt_vs_si_width();
  if (list.empty()) return;
  if (list.size() == 1) {
    crt1 = list.front().crt_1;
    crt2 = list.front().crt_2;
    return;
  }

  auto it = std::upper_bound(list.begin(), list.end(), width,
    [](double width, const itfiWidthCrt& e) {
      return width < e.si_width;
  });

  if (it == list.end()) {
    // upper extrapolation
    crt1 = list.back().crt_1;
    crt2 = list.back().crt_2;
  } else if (width <= list.front().si_width) {
    // lower extrapolation
    crt1 = list.front().crt_1;
    crt2 = list.front().crt_2;
  } else {
    // interpolation
    size_t id_h = std::distance(list.begin(), it);
    size_t id_l = id_h - 1;
    double t = (width - list.at(id_l).si_width) / (list.at(id_h).si_width - list.at(id_l).si_width);
    crt1 = std::lerp(list.at(id_l).crt_1, list.at(id_h).crt_1, t);
    crt2 = std::lerp(list.at(id_l).crt_2, list.at(id_h).crt_2, t);
  }
}

LayerDielectric::LayerDielectric()
: Layer(LayerType::kDielectric)
{ }

LayerDielectric::LayerDielectric(const itfiDielectric& d)
: Layer(LayerType::kDielectric)
{
  itfiDielectric::operator=(d);
}

LayerDielectric::~LayerDielectric()
{ }

std::string
LayerDielectric::get_name() const
{
  return get_dielectric_name();
}

// Units: microns
double
LayerDielectric::get_layer_thickness() const
{
  return get_thickness();
}

LayerVia::LayerVia()
: Layer(LayerType::kVia)
, _top_height(0)
{ }

LayerVia::LayerVia(const itfiVia& v)
: Layer(LayerType::kVia)
{
  itfiVia::operator=(v);
}

LayerVia::~LayerVia()
{

}

std::string
LayerVia::get_name() const 
{
  return get_via_name();
}

// Units: microns
double
LayerVia::get_layer_thickness() const
{
  return get_top_height() - get_bot_height();
}

double
LayerVia::get_bot_height() const
{
  return get_height();
}

double
LayerVia::get_top_height() const
{
  return _top_height;
}

void
LayerVia::set_top_height(double top)
{
  _top_height = top;
}

// StarRC UG Version U-2022.12, December 2022. P-1333
double
LayerVia::query_rpv_vs_area(double area) const
{
  auto& table = get_rpv_vs_area();
  if (table.empty()) return 0; // lack of information
  if (table.size() == 1) return table.front().rpv;

  auto it = std::upper_bound(table.begin(), table.end(), area,
    [](double area, const itfiAreaRpv& e) {
      return area < e.area;
  });
  
  if (it == table.end()) {
    // upper extrapolation
    return table.back().rpv;
  } else if (area <= table.front().area) {
    // lower extrapolation 
    return table.front().rpv;
  } else {
    // interpolation
    size_t id_l, id_h;
    id_h = std::distance(table.begin(), it);
    id_l = id_h - 1;
    double rec_rpv_l = 1 / table.at(id_l).rpv;
    double rec_rpv_h = 1 / table.at(id_h).rpv;
    double rec_rpv_interpolated = 
      std::lerp(rec_rpv_l, rec_rpv_h,
        (area - table.at(id_l).area) / (table.at(id_h).area - table.at(id_l).area ) );
    return 1 / rec_rpv_interpolated;
  }
}

void
LayerVia::query_crt_vs_area(
  double area,
  std::optional<double>& crt1,
  std::optional<double>& crt2
) const {
  crt1.reset();
  crt2.reset();

  auto& list = get_crt_vs_area();
  if (list.empty()) return;
  if (list.size() == 1) {
    crt1 = list.front().crt1;
    crt2 = list.front().crt2;
    return;
  }

  auto it = std::upper_bound(list.begin(), list.end(), area, 
    [](double area, const itfiAreaCrt& e){
      return area < e.area;
  });

  if (it == list.end()) {
    // upper extrapolation
    crt1 = list.back().crt1;
    crt2 = list.back().crt2;
  } else if (area <= list.front().area) {
    // lower extrapolation 
    crt1 = list.front().crt1;
    crt2 = list.front().crt2;
  } else {
    // interpolation
    size_t id_h = std::distance(list.begin(), it);
    size_t id_l = id_h - 1;
    double t = (area - list.at(id_l).area) / (list.at(id_h).area - list.at(id_l).area);
    crt1 = std::lerp(list.at(id_l).crt1, list.at(id_h).crt1, t);
    crt2 = std::lerp(list.at(id_l).crt2, list.at(id_h).crt2, t);
  }
}

Layers::~Layers()
{
  clear();
}

// @param order of layer
Layer*
Layers::find_layer(uint8_t order) const
{
  std::optional<uint8_t> lo;
  for (auto layer : _layers) {
    lo = layer->get_order();
    if (lo.has_value() && lo.value() == order) {
      return layer;
    }
  }

  return nullptr;
}

Layer*
Layers::find_layer(const std::string& name) const
{
  for (auto layer : _layers) {
    if (layer->get_name() == name) {
      return layer;
    } 
  }

  return nullptr;
}

// @param order layer order
LayerConductor*
Layers::find_conductor_layer(uint8_t order) const
{
  for (auto layer : _conductor_layers) {
    if (layer->get_order() == order) {
      return layer;
    }
  }

  return nullptr;
}

LayerConductor*
Layers::find_conductor_layer(const char* s) const
{
  if (!s) return nullptr;
  for (auto cdt : _conductor_layers) {
    if (cdt->get_name() == s) {
      return cdt;
    }
  }
  return nullptr;
}

LayerVia*
Layers::find_via_layer(uint8_t order) const
{
  for (auto layer : _via_layers) {
    if (layer->get_order() == order) {
      return layer;
    }
  }

  return nullptr;
}

void
Layers::clear()
{
  for (auto layer : _layers) {
    delete layer;
  }
  _layers.clear();
  _conductor_layers.clear();
  _dielectric_layers.clear();
  _via_layers.clear();
}

std::vector<Layer*>&
Layers::get_layers()
{
  return _layers;
}

const std::vector<Layer*>&
Layers::get_layers() const
{
  return _layers;
}

std::vector<LayerConductor*>&
Layers::get_conductor_layers()
{
  return _conductor_layers;
}

const std::vector<LayerConductor*>&
Layers::get_conductor_layers() const
{
  return _conductor_layers;
}

std::vector<LayerDielectric*>&
Layers::get_dielectric_layers()
{
  return _dielectric_layers;
}

const std::vector<LayerDielectric*>&
Layers::get_dielectric_layers() const
{
  return _dielectric_layers;
}

std::vector<LayerVia*>&
Layers::get_via_layers()
{
  return _via_layers;
}

const std::vector<LayerVia*>&
Layers::get_via_layers() const
{
  return _via_layers;
}

Layer*
Layers::get_uppermost_layer_by_order() const
{
  Layer* ret = nullptr;
  for (auto l : _layers) {
    if (!l->get_order().has_value())  continue;
    
    if ((ret == nullptr)
     || (ret->get_order().value() < l->get_order().value()) ) {
      ret = l;
    }
  }
  return ret;
}

Layer*
Layers::get_lowermost_layer_by_order() const
{
  Layer* ret = nullptr;
  for (auto l : _layers) {
    if (!l->get_order().has_value())  continue;
    
    if ((ret == nullptr)
     || (ret->get_order().value() > l->get_order().value()) ) {
      ret = l;
    }
  }
  return ret;
}

void
Layers::add_conductor_layer(LayerConductor* layer)
{
  if (layer) {
    layer->set_id(_layers.size());
    layer->set_order(static_cast<uint8_t>(_layers.size()));
    _conductor_layers.push_back(layer);
    _layers.push_back(layer);
  }
}

void
Layers::add_dielectric_layer(LayerDielectric* layer)
{
  if (layer) {
    layer->set_id(_layers.size());
    _dielectric_layers.push_back(layer);
    _layers.push_back(layer);
  } 
}

void
Layers::add_via_layer(LayerVia* layer)
{
  if (layer) {
    layer->set_id(_layers.size());
    layer->set_order(static_cast<uint8_t>(_layers.size()));
    _via_layers.push_back(layer);
    _layers.push_back(layer);
  }
}

// @brief find the first dielectric layer below the conductor layer
LayerDielectric*
Layers::find_diel_below(LayerConductor* cdt) const
{
  if (!cdt) return nullptr;
  
  auto it = std::upper_bound(_dielectric_layers.begin(), _dielectric_layers.end(), cdt->get_id(),
    [](int16_t id_high, LayerDielectric* d){
      return d->get_id() > id_high;
  });
  
  return it == _dielectric_layers.end() ? nullptr :
    _dielectric_layers.at(std::distance(_dielectric_layers.begin(), it));
}

LayerDielectric*
Layers::find_diel_below(double height) const
{
  auto it = std::upper_bound(_dielectric_layers.begin(), _dielectric_layers.end(), height, 
    [](double h, LayerDielectric* d){
      return d->get_height() < h; 
  });

  return it == _dielectric_layers.end() ? nullptr : *it;
}

// @brief find the first dielectric layer above the height
LayerDielectric*
Layers::find_diel_above(double height) const
{
  auto it = std::upper_bound(_dielectric_layers.rbegin(), _dielectric_layers.rend(), height,
    [](double h, LayerDielectric* d){
      return d->get_height() + d->get_layer_thickness() > h;
  });

  return it == _dielectric_layers.rend() ? nullptr : *it;
}

LayerDielectric*
Layers::find_diel(int16_t id) const
{
  for (auto diel : _dielectric_layers) {
    if (diel->get_id() == id) {
      return diel;
    }
  }

  return nullptr;
}

LayerDielectric*
Layers::find_diel(const char* s) const
{
  if (!s) return nullptr;
  for (auto d : _dielectric_layers) {
    if (d->get_name() == s) {
      return d;
    }
  }
  return nullptr; 
}

bool
Layers::is_lowermost_diel(int16_t id) const
{
  if (_dielectric_layers.size()) {
    auto diel = _dielectric_layers.at(_dielectric_layers.size() - 1);
    if (diel->get_id() == id) {
      return true;
    }
  }

  return false;
}

} // namespace itf
