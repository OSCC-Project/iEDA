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
#ifndef IDRC_SRC_DB_DENSITYRULE_H_
#define IDRC_SRC_DB_DENSITYRULE_H_

#include <algorithm>

namespace idrc {
class DensityRule
{
 public:
  DensityRule() : _min_density(0), _max_density(0), _density_check_length(-1), _density_check_width(-1), _density_check_step(-1) {}
  ~DensityRule() {}
  DensityRule(const DensityRule& other)
  {
    _min_density = other._min_density;
    _max_density = other._max_density;
    _density_check_length = other._density_check_length;
    _density_check_width = other._density_check_width;
    _density_check_step = other._density_check_step;
  }
  DensityRule(DensityRule&& other)
  {
    _min_density = std::move(other._min_density);
    _max_density = std::move(other._max_density);
    _density_check_length = std::move(other._density_check_length);
    _density_check_width = std::move(other._density_check_width);
    _density_check_step = std::move(other._density_check_step);
  }
  DensityRule& operator=(const DensityRule& other)
  {
    _min_density = other._min_density;
    _max_density = other._max_density;
    _density_check_length = other._density_check_length;
    _density_check_width = other._density_check_width;
    _density_check_step = other._density_check_step;
    return *this;
  }
  DensityRule& operator=(DensityRule&& other)
  {
    _min_density = std::move(other._min_density);
    _max_density = std::move(other._max_density);
    _density_check_length = std::move(other._density_check_length);
    _density_check_width = std::move(other._density_check_width);
    _density_check_step = std::move(other._density_check_step);
    return *this;
  }

  // getter
  double get_min_density() const { return _min_density; }
  double get_max_density() const { return _max_density; }
  int get_density_check_length() const { return _density_check_length; }
  int get_density_check_width() const { return _density_check_width; }
  int get_density_check_step() const { return _density_check_step; }
  // setter
  void set_min_density(double min_density) { _min_density = min_density; }
  void set_max_density(double max_density) { _max_density = max_density; }
  void set_density_check_length(int density_check_length) { _density_check_length = density_check_length; }
  void set_density_check_width(int density_check_width) { _density_check_width = density_check_width; }
  void set_density_check_step(int density_check_step) { _density_check_step = density_check_step; }

 private:
  double _min_density;
  double _max_density;
  int _density_check_length;
  int _density_check_width;
  int _density_check_step;
};
}  // namespace idrc

#endif