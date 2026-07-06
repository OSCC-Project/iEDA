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

#include "itfrReader.hpp"
namespace itf
{
  
class itfrCallBacks {
 public:
  // constructor
  itfrCallBacks();

  // function
  static void reset();

  // members
  itfrStringCbFnType technology_cb;
  itfrStringCbFnType process_foundry_cb;
  itfrIntegerCbFnType process_node_cb;
  itfrStringCbFnType process_type_cb;
  itfrDoubleCbFnType process_version_cb;
  itfrStringCbFnType process_corner_cb;
  itfrStringCbFnType reference_direction_cb;
  itfrDoubleCbFnType global_temperature_cb;
  itfrDoubleCbFnType background_er_cb;
  itfrDoubleCbFnType half_node_scale_factor_cb;
  itfrIntegerCbFnType use_si_density_cb;
  itfrDoubleCbFnType drop_factor_lateral_spacing_cb;
  itfrConductorCbFnType conductor_cb;
  itfrDielectricCbFnType dielectric_cb;
  itfrViaCbFnType via_cb;
  itfrVariationParamCbFnType variation_cb;
};

extern itfrCallBacks* itfCallbacks;

} // namespace itf
