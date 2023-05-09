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
#ifndef _IDB_TECH_SHAPE_H
#define _IDB_TECH_SHAPE_H
#include <memory>
#include <vector>

#include "IdbTechEnum.h"
#include "IdbTechLayer.h"
#include "IdbTechPoint.h"

namespace idb {
  class IdbTechBox {
   public:
    IdbTechBox() : _lower_left(), _upper_right() { }
    IdbTechBox(int llx, int lly, int urx, int ury) {
      _lower_left.setCoordinate((llx > urx) ? urx : llx, (lly > ury) ? ury : lly);
      _upper_right.setCoordinate((llx > urx) ? llx : urx, (lly > ury) ? lly : ury);
    }
    IdbTechBox(const IdbTechPoint &lowerLeft, const IdbTechPoint &upperRight)
        : IdbTechBox(lowerLeft.get_coordinate_x(), lowerLeft.get_coordinate_y(), upperRight.get_coordinate_x(),
                     upperRight.get_coordinate_y()) { }
    ~IdbTechBox() { }
    // setter
    void setPoint(const IdbTechBox &box) {
      _lower_left  = box._lower_left;
      _upper_right = box._upper_right;
    }
    void setPoint(int llx, int lly, int urx, int ury) {
      _lower_left.setCoordinate((llx > urx) ? urx : llx, (lly > ury) ? ury : lly);
      _upper_right.setCoordinate((llx > urx) ? llx : urx, (lly > ury) ? lly : ury);
    }
    void setPoint(const IdbTechPoint &lowerLeft, const IdbTechPoint &upperRight) {
      setPoint(lowerLeft.get_coordinate_x(), lowerLeft.get_coordinate_y(), upperRight.get_coordinate_x(),
               upperRight.get_coordinate_y());
    }

    // getter
    IdbTechPoint &get_lower_left() { return _lower_left; }
    IdbTechPoint &get_upper_right() { return _upper_right; }
    int get_left() const { return _lower_left.get_coordinate_x(); }
    int get_bottom() const { return _lower_left.get_coordinate_y(); }
    int get_right() const { return _upper_right.get_coordinate_x(); }
    int get_top() const { return _upper_right.get_coordinate_y(); }
    int get_width() const {
      int xSpan = get_right() - get_left();
      int ySpan = get_top() - get_bottom();
      return (xSpan > ySpan) ? ySpan : xSpan;
    }
    int get_length() const {
      int xSpan = get_right() - get_left();
      int ySpan = get_top() - get_bottom();
      return (xSpan > ySpan) ? xSpan : ySpan;
    }

    // others
    IdbTechBox &operator=(const IdbTechBox &box) {
      _lower_left  = box._lower_left;
      _upper_right = box._upper_right;
      return *this;
    }
    IdbTechBox &operator=(IdbTechBox &&box) {
      _lower_left  = std::move(box._lower_left);
      _upper_right = std::move(box._upper_right);
      return *this;
    }
    bool operator==(const IdbTechBox &box) { return (_lower_left == box._lower_left) && (_upper_right == box._upper_right); }
    bool operator<(const IdbTechBox &box) {
      return (_lower_left == box._lower_left) ? _upper_right < box._upper_right : _lower_left < box._lower_left;
    }
    bool contains(const IdbTechBox &box) const {
      return (box.get_right() <= _upper_right.get_coordinate_x() && box.get_left() >= _lower_left.get_coordinate_x() &&
              box.get_top() <= _upper_right.get_coordinate_y() && box.get_bottom() >= _lower_left.get_coordinate_y());
    }
    bool contains(const IdbTechPoint &point) const {
      return (point.get_coordinate_x() <= _upper_right.get_coordinate_x() &&
              point.get_coordinate_x() >= _lower_left.get_coordinate_x() &&
              point.get_coordinate_y() <= _upper_right.get_coordinate_y() &&
              point.get_coordinate_y() >= _lower_left.get_coordinate_y());
    }
    void coordinateConversion(IdbTechPoint &newOrigin) {
      int newBottom = newOrigin.get_coordinate_y() + _lower_left.get_coordinate_y();
      int newLeft   = newOrigin.get_coordinate_x() + _lower_left.get_coordinate_x();
      int newTop    = newOrigin.get_coordinate_y() + _upper_right.get_coordinate_y();
      int newRight  = newOrigin.get_coordinate_x() + _upper_right.get_coordinate_x();
      _lower_left.setCoordinate(newLeft, newBottom);
      _upper_right.setCoordinate(newRight, newTop);
    }

    void coordinateConversion(int x, int y, int offsetMidX, int offsetMidY) {
      int newBottom = y + _lower_left.get_coordinate_y() - offsetMidY;
      int newLeft   = x + _lower_left.get_coordinate_x() - offsetMidX;
      int newTop    = y + _upper_right.get_coordinate_y() - offsetMidY;
      int newRight  = x + _upper_right.get_coordinate_x() - offsetMidX;
      _lower_left.setCoordinate(newLeft, newBottom);
      _upper_right.setCoordinate(newRight, newTop);
    }

   private:
    IdbTechPoint _lower_left;
    IdbTechPoint _upper_right;
  };

  class IdbTechRect {
   public:
    IdbTechRect() : _box() { }
    IdbTechRect(int llx, int lly, int urx, int ury) : _box(llx, lly, urx, ury) { }
    IdbTechRect(const IdbTechRect &rect) { _box = rect._box; }
    IdbTechRect(IdbTechRect &&rect) { _box = std::move(rect._box); }
    ~IdbTechRect() { }

    IdbTechRect &operator=(const IdbTechRect &rect) {
      _box = rect._box;
      return *this;
    }
    IdbTechRect &operator=(IdbTechRect &&rect) {
      _box = std::move(rect._box);
      return *this;
    }
    // getter
    IdbTechBox &get_box() { return _box; }
    // setter
    void set_box(const IdbTechBox &box) { _box = box; }
    // others
    bool isHorizontial() {
      int xSpan = _box.get_right() - _box.get_left();
      int ySpan = _box.get_top() - _box.get_bottom();
      return (xSpan >= ySpan) ? true : false;
    }
    bool isVertical() {
      int xSpan = _box.get_right() - _box.get_left();
      int ySpan = _box.get_top() - _box.get_bottom();
      return (xSpan <= ySpan) ? true : false;
    }
    int getWidth() const { return _box.get_width(); }
    int getLength() const { return _box.get_length(); }
    int getLowerLeftX() { return _box.get_left(); }
    int getLowerLeftY() { return _box.get_bottom(); }
    int getUpperRightX() { return _box.get_right(); }
    int getUpperRightY() { return _box.get_top(); }
    void setRectPoint(int llx, int lly, int urx, int ury) { _box.setPoint(llx, lly, urx, ury); }
    void coordinateConversion(IdbTechPoint &newOrigin) { _box.coordinateConversion(newOrigin); }
    void coordinateConversion(int x, int y, int offsetMidX, int offsetMidY) {
      _box.coordinateConversion(x, y, offsetMidX, offsetMidY);
    }

   private:
    IdbTechBox _box;
  };

  class IdbTechRectList {
   public:
    IdbTechRectList() { }
    ~IdbTechRectList() { }

    int get_rect_list_num() { return _rect_list_num; }

    void addRect(int llx, int lly, int urx, int ury) {
      std::unique_ptr<IdbTechRect> rect = std::make_unique<IdbTechRect>(llx, lly, urx, ury);
      _rect_list.push_back(std::move(rect));
    }
    IdbTechRect *getRect(int index) {
      if (index < static_cast<int>(_rect_list.size()) && _rect_list.size() > 0) {
        return _rect_list.at(index).get();
      }
      return nullptr;
    }
    void coordinateConversionForEveryVia(IdbTechPoint &newOrigin) {
      for (auto &rect : _rect_list) {
        rect->coordinateConversion(newOrigin);
      }
    }
    void coordinateConversionForEveryVia(int x, int y, int offsetMidX, int offsetMidY) {
      for (auto &rect : _rect_list) {
        rect->coordinateConversion(x, y, offsetMidX, offsetMidY);
      }
    }

    IdbTechRect *getFirstRect() { return _rect_list.front().get(); }
    IdbTechRect *getLastRect() { return _rect_list.back().get(); }

   private:
    int _rect_list_num;
    std::vector<std::unique_ptr<IdbTechRect>> _rect_list;
  };

  class IdbTechShape {
   public:
    IdbTechShape() { }
    IdbTechShape(IdbTechShape &&other) { _rect_list = std::move(other._rect_list); }
    ~IdbTechShape() { }
    // getter
    // std::vector<std::unique_ptr<IdbTechRect>> &get_rect_list() { return _rect_list; }
    IdbTechCutLayer *get_layer() { return _cut_layer; }
    // setter
    void set_layer(IdbTechCutLayer *layer) { _cut_layer = layer; }
    // other
    IdbTechShape &operator=(IdbTechShape &&other) {
      _rect_list = std::move(other._rect_list);
      return *this;
    }
    void addRect(int llx, int lly, int urx, int ury) {
      std::unique_ptr<IdbTechRect> rect = std::make_unique<IdbTechRect>(llx, lly, urx, ury);
      _rect_list.push_back(std::move(rect));
    }

   private:
    // int _index = 0;
    IdbTechCutLayer *_cut_layer;
    std::vector<std::unique_ptr<IdbTechRect>> _rect_list;
  };

}  // namespace idb

#endif
