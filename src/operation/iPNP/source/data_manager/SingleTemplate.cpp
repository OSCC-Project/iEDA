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
/**
 * @file SingleTemplate.cpp
 * @author Jianrong Su
 * @brief Implementation of SingleTemplate class
 * @version 1.0
 * @date 2025-06-23
 */

#include "SingleTemplate.hh"

namespace ipnp {

SingleTemplate::SingleTemplate(StripeDirection direction, PowerType first_stripe_power_type, 
                               double width, double pg_offset, double space, double offset)
    : _direction(direction),
      _width(width),
      _pg_offset(pg_offset),
      _space(space),
      _offset(offset)
{

}

}  // namespace ipnp 