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

#include "PlanarRect.hpp"

namespace irt {

class EXTPlanarRect
{
 public:
  EXTPlanarRect() = default;
  ~EXTPlanarRect() = default;
  // getter
  PlanarRect& get_grid_rect() { return _grid_rect; }
  PlanarCoord& get_grid_lb() { return _grid_rect.get_lb(); }
  PlanarCoord& get_grid_rt() { return _grid_rect.get_rt(); }
  irt_int get_grid_lb_x() const { return _grid_rect.get_lb_x(); }
  irt_int get_grid_lb_y() const { return _grid_rect.get_lb_y(); }
  irt_int get_grid_rt_x() const { return _grid_rect.get_rt_x(); }
  irt_int get_grid_rt_y() const { return _grid_rect.get_rt_y(); }
  PlanarRect& get_real_rect() { return _real_rect; }
  PlanarCoord& get_real_lb() { return _real_rect.get_lb(); }
  PlanarCoord& get_real_rt() { return _real_rect.get_rt(); }
  irt_int get_real_lb_x() const { return _real_rect.get_lb_x(); }
  irt_int get_real_lb_y() const { return _real_rect.get_lb_y(); }
  irt_int get_real_rt_x() const { return _real_rect.get_rt_x(); }
  irt_int get_real_rt_y() const { return _real_rect.get_rt_y(); }
  // const getter
  const PlanarRect& get_grid_rect() const { return _grid_rect; }
  const PlanarRect& get_real_rect() const { return _real_rect; }
  // setter
  void set_grid_rect(const PlanarRect& grid_rect) { _grid_rect = grid_rect; }
  void set_grid_lb(const PlanarCoord& grid_lb) { _grid_rect.set_lb(grid_lb); }
  void set_grid_rt(const PlanarCoord& grid_rt) { _grid_rect.set_rt(grid_rt); }
  void set_grid_lb(const irt_int x, const irt_int y) { _grid_rect.set_lb(x, y); }
  void set_grid_rt(const irt_int x, const irt_int y) { _grid_rect.set_rt(x, y); }
  void set_real_rect(const PlanarRect& real_rect) { _real_rect = real_rect; }
  void set_real_lb(const PlanarCoord& real_lb) { _real_rect.set_lb(real_lb); }
  void set_real_rt(const PlanarCoord& real_rt) { _real_rect.set_rt(real_rt); }
  void set_real_lb(const irt_int x, const irt_int y) { _real_rect.set_lb(x, y); }
  void set_real_rt(const irt_int x, const irt_int y) { _real_rect.set_rt(x, y); }
  // function
  inline irt_int getXSize() const;
  inline irt_int getYSize() const;
  inline irt_int getTotalSize() const;
  inline irt_int getRealLength() const;
  inline irt_int getRealWidth() const;
  inline irt_int getRealHalfPerimeter() const;
  inline irt_int getRealPerimeter() const;
  inline irt_int getRealArea() const;

 private:
  PlanarRect _grid_rect;
  PlanarRect _real_rect;
};

inline irt_int EXTPlanarRect::getXSize() const
{
  return get_grid_rt_x() - get_grid_lb_x() + 1;
}

inline irt_int EXTPlanarRect::getYSize() const
{
  return get_grid_rt_y() - get_grid_lb_y() + 1;
}

inline irt_int EXTPlanarRect::getTotalSize() const
{
  return getXSize() * getYSize();
}

inline irt_int EXTPlanarRect::getRealLength() const
{
  return get_real_rt_x() - get_real_lb_x();
}

inline irt_int EXTPlanarRect::getRealWidth() const
{
  return get_real_rt_y() - get_real_lb_y();
}

inline irt_int EXTPlanarRect::getRealHalfPerimeter() const
{
  return getRealLength() + getRealWidth();
}

inline irt_int EXTPlanarRect::getRealPerimeter() const
{
  return getRealHalfPerimeter() * 2;
}

inline irt_int EXTPlanarRect::getRealArea() const
{
  return getRealLength() * getRealWidth();
}

}  // namespace irt
