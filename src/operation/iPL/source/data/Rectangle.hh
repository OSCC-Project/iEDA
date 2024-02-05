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
/*
 * @Author: S.J Chen
 * @Date: 2022-01-20 19:30:26
 * @LastEditTime: 2022-09-08 10:55:43
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/database/Rectangle.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_RECTANGLE_H
#define IPL_RECTANGLE_H

#include "Point.hh"

namespace ipl {

template <class T>
class Rectangle
{
 public:
  Rectangle() = default;
  Rectangle(T ll_x, T ll_y, T ur_x, T ur_y)
  {
    this->set_lower_left(ll_x, ll_y);
    this->set_upper_right(ur_x, ur_y);
  }
  Rectangle(Point<T> ll, Point<T> ur)
  {
    _lower_left  = std::move(ll);
    _upper_right = std::move(ur);
  }
  Rectangle(const Rectangle& other)
  {
    _lower_left  = other._lower_left;
    _upper_right = other._upper_right;
  }
  Rectangle(Rectangle&& other)
  {
    _lower_left  = std::move(other._lower_left);
    _upper_right = std::move(other._upper_right);
  }
  ~Rectangle() = default;

  Rectangle& operator=(const Rectangle& other)
  {
    _lower_left  = other._lower_left;
    _upper_right = other._upper_right;
    return (*this);
  }

  Rectangle& operator=(Rectangle&& other)
  {
    _lower_left  = std::move(other._lower_left);
    _upper_right = std::move(other._upper_right);
    return (*this);
  }

  bool operator==(const Rectangle& other) { return (_lower_left == other._lower_left && _upper_right == other._upper_right); }
  bool operator!=(const Rectangle& other) { return !((*this) == other); }

  Rectangle get_intersetion(const Rectangle& rect) const{
    T lx = std::max(this->get_ll_x(), rect.get_ll_x());
    T ly = std::max(this->get_ll_y(), rect.get_ll_y());
    T ux = std::min(this->get_ur_x(), rect.get_ur_x());
    T uy = std::min(this->get_ur_y(), rect.get_ur_y());

    if (lx < ux && ly < uy) {
      Rectangle intersection_rect(lx, ly, ux, uy);
      return intersection_rect;
    } else {
      // Return an invalid rectangle or a special value
      return Rectangle(0,0,0,0);
    }
  }

  // getter.
  Point<T> get_lower_left() const { return _lower_left; }
  Point<T> get_upper_right() const { return _upper_right; }
  Point<T> get_center() const;

  T get_ll_x() const { return _lower_left.get_x(); }
  T get_ll_y() const { return _lower_left.get_y(); }
  T get_ur_x() const { return _upper_right.get_x(); }
  T get_ur_y() const { return _upper_right.get_y(); }

  // setter.
  void set_rectangle(T ll_x, T ll_y, T ur_x, T ur_y)
  {
    this->set_lower_left(ll_x, ll_y);
    this->set_upper_right(ur_x, ur_y);
  }

  void set_center(const T x, const T y)
  {
    T width  = this->get_width();
    T height = this->get_height();
    this->set_lower_left(x - width / 2, y - height / 2);
    this->set_upper_right(x + width / 2, y + height / 2);
  }

  void set_lower_left(const T x, const T y)
  {
    _lower_left.set_x(x);
    _lower_left.set_y(y);
  }

  void set_upper_right(const T x, const T y)
  {
    _upper_right.set_x(x);
    _upper_right.set_y(y);
  }

  // function.
  T get_width() const;
  T get_height() const;
  T get_half_perimeter();
  T get_perimeter();
  T get_area();

  bool isUnlegal() { return _lower_left.isUnLegal() || _upper_right.isUnLegal(); }

 private:
  Point<T> _lower_left;
  Point<T> _upper_right;
};

template <typename T>
inline Point<T> Rectangle<T>::get_center() const
{
  T center_x = this->get_ll_x() + (this->get_ur_x() - this->get_ll_x()) * 0.5;
  T center_y = this->get_ll_y() + (this->get_ur_y() - this->get_ll_y()) * 0.5;
  return Point<T>(center_x, center_y);
}

template <typename T>
inline T Rectangle<T>::get_width() const
{
  return this->get_ur_x() - this->get_ll_x();
}

template <typename T>
inline T Rectangle<T>::get_height() const
{
  return this->get_ur_y() - this->get_ll_y();
}

template <typename T>
inline T Rectangle<T>::get_half_perimeter()
{
  return this->get_width() + this->get_height();
}

template <typename T>
inline T Rectangle<T>::get_perimeter()
{
  return this->get_half_perimeter() * 2;
}

}  // namespace ipl

#endif