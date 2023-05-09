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
#ifndef IDB_TECH_VIA_H
#define IDB_TECH_VIA_H

#include "IdbTechShape.h"

namespace idb {
class IdbTechVia
{
 public:
  IdbTechVia();
  explicit IdbTechVia(const std::string& name);

  IdbTechVia(IdbTechVia&& other);
  ~IdbTechVia() {}

  IdbTechVia& operator=(IdbTechVia&& other);
  // getter
  std::string get_name() { return _name; }
  bool get_is_default() { return _is_default; }
  IdbTechRect* get_bottom_layer_shape();
  IdbTechRectList* get_cut_layer_shape_list() { return _cut_layer_shape_list.get(); }
  IdbTechRect* get_top_layer_shape();
  ViaCutNumEnum get_cut_num() { return _cut_num; }
  LayerDirEnum& get_bottom_layer_default_direction() { return _bottom_layer_default_direction; }
  LayerDirEnum& get_top_layer_default_direction() { return _top_layer_default_direction; }
  bool isOneCutVia() { return _cut_num == ViaCutNumEnum::k1CUTVIA ? true : false; }
  bool isTwoCutVia() { return _cut_num == ViaCutNumEnum::k2CUTVIA ? true : false; }
  bool isDefaultBottomHorizonTopVertical();
  bool isDefaultBottomVerticalTopHorizon();
  int get_cut_layer_id() { return _cut_layer_id; }
  CutArrayTypeEnum& get_cut_array_type() { return _cut_array_type; }
  bool isTwoCutTypeNorth() { return _cut_array_type == CutArrayTypeEnum::kNORTH; }
  bool isTwoCutTypeSouth() { return _cut_array_type == CutArrayTypeEnum::kSOUTH; }
  bool isTwoCutTypeWest() { return _cut_array_type == CutArrayTypeEnum::kWEST; }
  bool isTwoCutTypeEast() { return _cut_array_type == CutArrayTypeEnum::kEAST; }
  int get_origin_x() { return _origin->get_coordinate_x(); }
  int get_origin_y() { return _origin->get_coordinate_y(); }

  // setter
  void set_name(const std::string& name) { _name = name; }
  void set_is_default(bool is_default) { _is_default = is_default; }
  void set_bottom_layer_shape(std::unique_ptr<IdbTechRect>& shape) { _bottom_layer_shape = std::move(shape); }
  void set_cut_layer_shape_list(std::unique_ptr<IdbTechRectList>& shape) { _cut_layer_shape_list = std::move(shape); }
  void set_top_layer_shape(std::unique_ptr<IdbTechRect>& shape) { _top_layer_shape = std::move(shape); }
  void set_cut_num(ViaCutNumEnum cutNum) { _cut_num = cutNum; }
  void set_cut_layer_id(int id) { _cut_layer_id = id; }
  void set_bottom_layer_default_direction(LayerDirEnum& direction) { _bottom_layer_default_direction = direction; }
  void set_top_layer_default_direction(LayerDirEnum& direction) { _top_layer_default_direction = direction; }
  void set_cut_array_type(const CutArrayTypeEnum& type) { _cut_array_type = type; }
  void set_origin(int x, int y) { _origin->setCoordinate(x, y); }

  // other
  void setBottomRect(int llx, int lly, int urx, int ury);
  void addCutRectList(int llx, int lly, int urx, int ury) { _cut_layer_shape_list->addRect(llx, lly, urx, ury); }
  void setTopRect(int llx, int lly, int urx, int ury);
  // interface
  int getBottomRectLowerLeftX() { return get_origin_x() + _bottom_layer_shape->getLowerLeftX(); }
  int getBottomRectLowerLeftY() { return get_origin_y() + _bottom_layer_shape->getLowerLeftY(); }
  int getBottomRectUpperRightX() { return get_origin_x() + _bottom_layer_shape->getUpperRightX(); }
  int getBottomRectUpperRightY() { return get_origin_y() + _bottom_layer_shape->getUpperRightY(); }

  int getTopRectLowerLeftX() { return get_origin_x() + _top_layer_shape->getLowerLeftX(); }
  int getTopRectLowerLeftY() { return get_origin_y() + _top_layer_shape->getLowerLeftY(); }
  int getTopRectUpperRightX() { return get_origin_x() + _top_layer_shape->getUpperRightX(); }
  int getTopRectUpperRightY() { return get_origin_y() + _top_layer_shape->getUpperRightY(); }

  int getCutRectLowerLeftX(int index) { return get_origin_x() + _cut_layer_shape_list->getRect(index)->getLowerLeftX(); }
  int getCutRectLowerLeftY(int index) { return get_origin_y() + _cut_layer_shape_list->getRect(index)->getLowerLeftY(); }
  int getCutRectUpperRightX(int index) { return get_origin_x() + _cut_layer_shape_list->getRect(index)->getUpperRightX(); }
  int getCutRectUpperRightY(int index) { return get_origin_y() + _cut_layer_shape_list->getRect(index)->getUpperRightY(); }

  bool isBottomRectHorizon() { return _bottom_layer_shape->isHorizontial(); }
  bool isBottomRectVertical() { return _bottom_layer_shape->isVertical(); }
  bool isTopRectHorizon() { return _top_layer_shape->isHorizontial(); }
  bool isTopRectVertical() { return _top_layer_shape->isVertical(); }

  void cutViaCoordinateConversion(IdbTechPoint& newOrigin);
  void cutViaCoordinateConversion(int x, int y);

 private:
  std::string _name;
  ViaCutNumEnum _cut_num;
  CutArrayTypeEnum _cut_array_type;
  LayerDirEnum _bottom_layer_default_direction;
  LayerDirEnum _top_layer_default_direction;
  int _cut_layer_id;
  bool _is_default;
  std::unique_ptr<IdbTechRect> _bottom_layer_shape;
  std::unique_ptr<IdbTechRectList> _cut_layer_shape_list;
  std::unique_ptr<IdbTechRect> _top_layer_shape;
  std::unique_ptr<IdbTechPoint> _origin;
};

class IdbTechViaList
{
 public:
  IdbTechViaList() {}
  ~IdbTechViaList() {}

  void addVia(std::unique_ptr<IdbTechVia>& via) { _tech_vias.push_back(std::move(via)); }
  void printVia();

  IdbTechVia* get_tech_via(int index)
  {
    if (index > static_cast<int>(_tech_vias.size()) && _tech_vias.size() > 0) {
      return _tech_vias.at(index).get();
    }
    return nullptr;
  }
  std::vector<IdbTechVia*> get_tech_via_list();
  IdbTechVia* getOneCutVia(int bottomLayerId, int originX, int originY);

  std::vector<IdbTechVia*> getOneCutViaList(int cutLayerId, int originX, int originY);
  std::vector<IdbTechVia*> getTwoCutViaList(int cutLayerId, int originX, int originY);

 private:
  std::vector<std::unique_ptr<IdbTechVia>> _tech_vias;
};
}  // namespace idb

#endif
