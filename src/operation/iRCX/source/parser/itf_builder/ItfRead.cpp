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
#include <assert.h>
#include <iostream>

#include "ItfRead.hpp"
namespace itf
{


ItfRead::ItfRead(ItfService* itf_service)
: _itf_service(nullptr)
, _fname()
, _process_corner(nullptr)
{
  assert(itf_service);
  _itf_service = itf_service;
}

bool
ItfRead::createDb(const std::string& fname)
{
  itfrSetTechnologyCb(technologyCb);
  itfrSetGlobalTemperatureCb(globalTemperatureCb);
  itfrSetBackgroundErCb(backgroundErCb);
  itfrSetHalfNodeScaleFactorCb(halfNodeScaleFactorCb);
  itfrSetUseSiDensityCb(useSiDensityCb);
  itfrSetDropFactorLateralSpacingCb(dropFactorLateralSpacingCb);
  itfrSetDielectricCb(dielectricCb);
  itfrSetConductorCb(conductorCb);
  itfrSetViaCb(viaCb);
  itfrSetVariationParamCb(variationParamCb);

  itfrInit();

  FILE* file = fopen(fname.c_str(), "r");
  if (file == nullptr) {
    std::cout << "fail to open: " << fname << std::endl;
    return false;
  }

  _process_corner = std::make_unique<ProcessCorner>();

  _fname = fname;
  const int ret = itfrRead(file, fname.c_str(), this);
  fclose(file);

  itfrClear();
  if (ret != 0) {
    _process_corner.reset();
    return false;
  }

  _process_corner->update_layers_height();
  _itf_service->add_process_corner(std::move(_process_corner));

  return true;
}
int
ItfRead::technologyCb(itfCallBackType c, const char* string, itfiUserData user_data)
{
  if (c != itfCallBackType::kTechnologyCbType) {
    std::cout << "Callback type error, needs kTechnologyCbType" << std::endl;
    return 1;
  }

  if (string == nullptr) {
    std::cout << "string is nullptr" << std::endl;
    return 1;
  }

  auto itf_reader = static_cast<ItfRead*>(user_data);
  return itf_reader->parse_technology(string);
}

int
ItfRead::globalTemperatureCb(itfCallBackType c, double temperature, itfiUserData user_data)
{
  if (c != itfCallBackType::kGlobalTemperatureCbType) {
    std::cout << "Callback type error, needs kGlobalTemperatureCbType" << std::endl;
    return 1;
  } 
  auto itf_reader = static_cast<ItfRead*>(user_data);
  return itf_reader->parse_globalTemperature(temperature);
}

int
ItfRead::backgroundErCb(itfCallBackType c, double background_er, itfiUserData user_data)
{
  if (c != itfCallBackType::kBackgroundErCbType) {
    std::cout << "Callback type error, needs kBackgroundErCbType" << std::endl;
    return 1;
  }
  auto itf_reader = static_cast<ItfRead*>(user_data);
  return itf_reader->parse_backgroundEr(background_er);
}

int
ItfRead::halfNodeScaleFactorCb(itfCallBackType c, double factor, itfiUserData user_data)
{
  if (c != itfCallBackType::kHalfNodeScaleFactorCbType) {
    std::cout << "Callback type error, needs kHalfNodeScaleFactorCbType" << std::endl;
    return 1;
  }
  auto itf_reader = static_cast<ItfRead*>(user_data);
  return itf_reader->parse_halfNodeScaleFactor(factor);
}

int
ItfRead::useSiDensityCb(itfCallBackType c, int use, itfiUserData user_data)
{
  if (c != itfCallBackType::kUseSiDensityCbType) {
    std::cout << "Callback type error, needs kUseSiDensityCbType" << std::endl;
    return 1;
  }
  auto itf_reader = static_cast<ItfRead*>(user_data);
  return itf_reader->parse_useSiDensity(use);
}

int
ItfRead::dropFactorLateralSpacingCb(itfCallBackType c, double factor, itfiUserData user_data)
{
  if (c != itfCallBackType::kDropFactorLateralSpacingCbType) {
    std::cout << "Callback type error, needs kDropFactorLateralSpacingCbType" << std::endl;
    return 1;
  }
  auto itf_reader = static_cast<ItfRead*>(user_data);
  return itf_reader->parse_dropFactorLateralSpacing(factor);
}

int
ItfRead::dielectricCb(itfCallBackType c, itfiDielectric* dielectric, itfiUserData user_data)
{
  if (c != itfCallBackType::kDielectricCbType) {
    std::cout << "Callback type error, needs kDielectricCbType" << std::endl;
    return 1;
  }

  if (dielectric == nullptr) {
    std::cout << "itf Dielectric is nullptr" << std::endl;
    return 1;
  }

  auto itf_reader = static_cast<ItfRead*>(user_data);
  return itf_reader->parse_dielectric(*dielectric);
}

int
ItfRead::conductorCb(itfCallBackType c, itfiConductor* conductor, itfiUserData user_data)
{
  if (c != itfCallBackType::kConductorCbType) {
    std::cout << "Callback type error, needs kConductorCbType" << std::endl;
    return 1;
  } 

  if (conductor == nullptr) {
    std::cout << "itf Conductor is nullptr" << std::endl;
    return 1;
  }

  auto itf_reader = static_cast<ItfRead*>(user_data);
  return itf_reader->parse_conductor(*conductor);
}

int
ItfRead::viaCb(itfCallBackType c, itfiVia* via, itfiUserData user_data)
{
  if (c != itfCallBackType::kViaCbType) {
    std::cout << "Callback type error, needs kViaCbType" << std::endl;
    return 1;
  } 

  if (via == nullptr) {
    std::cout << "itf Via is nullptr" << std::endl;
    return 1;
  }

  auto itf_reader = static_cast<ItfRead*>(user_data);
  return itf_reader->parse_via(*via);
}

int
ItfRead::variationParamCb(itfCallBackType c, itfiVariationParam* vp, itfiUserData user_data)
{
  if (c != itfCallBackType::kVariationCbType) {
    std::cout << "Callback type error, needs kVariationCbType" << std::endl;
    return 1;
  } 

  if (vp == nullptr) {
    std::cout << "itf Variation Param is nullptr" << std::endl;
    return 1;
  }

  auto itf_reader = static_cast<ItfRead*>(user_data);
  return itf_reader->parse_variation_param(*vp);
}

int
ItfRead::parse_technology(const char* v)
{
  _process_corner->set_technology(v);
  return 0;
}
int
ItfRead::parse_globalTemperature(double v)
{
  _process_corner->set_global_temperature(v);
  return 0;
}
int
ItfRead::parse_backgroundEr(double v)
{
  _process_corner->set_background_er(v);
  return 0;
}
int
ItfRead::parse_halfNodeScaleFactor(double v)
{
  _process_corner->set_half_node_scale_factor(v);
  return 0;
}
int
ItfRead::parse_useSiDensity(int v)
{
  _process_corner->set_use_si_density(v);
  return 0;
}
int
ItfRead::parse_dropFactorLateralSpacing(double v)
{
  _process_corner->set_drop_factor_lateral_spacing(v);
  return 0;
}

int
ItfRead::parse_dielectric(const itfiDielectric& dielectric)
{
  auto diel = new LayerDielectric(dielectric);
  auto layers = _process_corner->get_layers();
  layers->add_dielectric_layer(diel);
  return 0;
}

int
ItfRead::parse_conductor(const itfiConductor& conductor)
{
  auto cdt = new LayerConductor(conductor);
  auto layers = _process_corner->get_layers();
  layers->add_conductor_layer(cdt);
  return 0;
}

int
ItfRead::parse_via(const itfiVia& via)
{
  auto v = new LayerVia(via);
  auto layers = _process_corner->get_layers();
  layers->add_via_layer(v);
  return 0;
}

int
ItfRead::parse_variation_param(const itfiVariationParam& vp)
{
  auto vps = _process_corner->get_variation_params();
  vps->add_variation_param(vp);
  return 0;
}

} // namespace itf
