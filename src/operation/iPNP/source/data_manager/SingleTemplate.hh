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
 * @file SingleTemplate.hh
 * @author Jianrong Su
 * @brief Definition of SingleTemplate class and related enums
 * @version 1.0
 * @date 2025-06-23
 */

#pragma once

namespace ipnp {

/**
 * @brief Direction of power stripes
 */
enum class StripeDirection
{
  kHorizontal,
  kVertical
};

/**
 * @brief Type of power net
 */
enum class PowerType
{
  kVDD,
  kVSS
};

/**
 * @brief A single layer Template in 3D Template block
 */
class SingleTemplate
{
 public:
  SingleTemplate(StripeDirection direction = StripeDirection::kHorizontal,
    PowerType first_stripe_power_type = PowerType::kVSS,
    double width = -1.0, 
    double pg_offset = -1.0, 
    double space = -1.0, 
    double offset = -1.0);
  
  ~SingleTemplate() = default;

  // Getters
  StripeDirection get_direction() const { return _direction; }
  double get_width() const { return _width; }
  double get_pg_offset() const { return _pg_offset; }
  double get_space() const { return _space; }
  double get_offset() const { return _offset; }

  // Setters
  void set_direction(StripeDirection direction) { _direction = direction; }
  void set_width(double width) { _width = width; }
  void set_pg_offset(double pg_offset) { _pg_offset = pg_offset; }
  void set_space(double space) { _space = space; }
  void set_offset(double offset) { _offset = offset; }

 private:
  StripeDirection _direction;
  double _width;
  double _pg_offset;  // offset between the first power and ground wire
  double _space;      // distance between edges of two VDD wire
  double _offset;     // if direction is horizontal, offset from bottom; if direction is vertical, offset from left.
  /**
   * @attention DRC: width + pg_offset < space
   */
};

}  // namespace ipnp 