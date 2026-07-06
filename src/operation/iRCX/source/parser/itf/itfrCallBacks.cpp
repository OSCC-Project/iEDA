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
#include "itfrCallBacks.hpp"
namespace itf
{
  
itfrCallBacks* itfCallbacks = nullptr;

itfrCallBacks::itfrCallBacks()
: technology_cb(nullptr)
, process_foundry_cb(nullptr)
, process_node_cb(0)
, process_type_cb(nullptr)
, process_version_cb(0)
, process_corner_cb(nullptr)
, reference_direction_cb(nullptr)
, global_temperature_cb(nullptr)
, background_er_cb(nullptr)
, half_node_scale_factor_cb(nullptr)
, use_si_density_cb(nullptr)
, drop_factor_lateral_spacing_cb(nullptr)
, conductor_cb(nullptr)
, dielectric_cb(nullptr)
, via_cb(nullptr)
, variation_cb(nullptr)
{

}

void
itfrCallBacks::reset()
{
  if (itfCallbacks) {
    delete itfCallbacks;
  }

  itfCallbacks = new itfrCallBacks();
}

} // namespace itf
