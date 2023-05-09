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
#include "IdbTechVia.h"

namespace idb {
  void IdbTechViaList::printVia() {
    for (auto &via : _tech_vias) {
      if (via->get_cut_layer_id() == 1 && via->isBottomRectVertical() && via->isTopRectHorizon()) {
        if (via->get_cut_num() == ViaCutNumEnum::k1CUTVIA) {
          // IdbTechPoint point(200, 200);
          // via->oneCutViaCoordinateConversion(point);
          std::cout << "via name ::" << via->get_name() << std::endl;
          std::cout << "BottomRectLowerLeftX is :: " << via->getBottomRectLowerLeftX() << std::endl;
          std::cout << "BottomRectLowerLeftY is :: " << via->getBottomRectLowerLeftY() << std::endl;
        }
      }
    }
  }
  std::vector<IdbTechVia *> IdbTechViaList::get_tech_via_list() {
    std::vector<IdbTechVia *> vias;
    for (auto &via : _tech_vias) {
      vias.push_back(via.get());
    }
    return vias;
  }

  IdbTechVia *IdbTechViaList::getOneCutVia(int bottomLayerId, int originX, int originY) {
    IdbTechVia *techVia = nullptr;
    for (auto &via : _tech_vias) {
      if (via->get_cut_layer_id() == bottomLayerId && via->get_cut_num() == ViaCutNumEnum::k1CUTVIA) {
        if (via->isDefaultBottomHorizonTopVertical() && via->isBottomRectHorizon() && via->isTopRectVertical()) {
          techVia = via.get();
          break;
        }
        if (via->isDefaultBottomVerticalTopHorizon() && via->isBottomRectVertical() && via->isTopRectHorizon()) {
          techVia = via.get();
          break;
        }
      }
    }
    if (techVia) {
      techVia->set_origin(originX, originY);
    }
    return techVia;
  }

  std::vector<IdbTechVia *> IdbTechViaList::getOneCutViaList(int cutLayerId, int originX, int originY) {
    std::vector<IdbTechVia *> oneCutViaList;
    for (auto &via : _tech_vias) {
      if (via->get_cut_layer_id() == cutLayerId && via->get_cut_num() == ViaCutNumEnum::k1CUTVIA) {
        via->set_origin(originX, originY);
        oneCutViaList.push_back(via.get());
      }
    }
    return oneCutViaList;
  }
  std::vector<IdbTechVia *> IdbTechViaList::getTwoCutViaList(int cutLayerId, int originX, int originY) {
    std::vector<IdbTechVia *> twoCutViaList;
    for (auto &via : _tech_vias) {
      if (via->get_cut_layer_id() == cutLayerId && via->get_cut_num() == ViaCutNumEnum::k2CUTVIA) {
        via->set_origin(originX, originY);
        twoCutViaList.push_back(via.get());
      }
    }
    return twoCutViaList;
  }

  /********************************techVia***************************************/
  IdbTechVia::IdbTechVia()
      : _name(""),
        _cut_num(ViaCutNumEnum::kUNKOWN),
        _cut_array_type(CutArrayTypeEnum::kUNKNOWN),
        _bottom_layer_default_direction(LayerDirEnum::kUNKNOWN),
        _top_layer_default_direction(LayerDirEnum::kUNKNOWN),
        _cut_layer_id(-1),
        _is_default(false),
        _bottom_layer_shape(nullptr),
        _top_layer_shape(nullptr) {
    _cut_layer_shape_list = std::make_unique<IdbTechRectList>();
    _origin               = std::make_unique<IdbTechPoint>(0, 0);
  }
  IdbTechVia::IdbTechVia(const std::string &name)
      : _name(name),
        _cut_num(ViaCutNumEnum::kUNKOWN),
        _cut_array_type(CutArrayTypeEnum::kUNKNOWN),
        _bottom_layer_default_direction(LayerDirEnum::kUNKNOWN),
        _top_layer_default_direction(LayerDirEnum::kUNKNOWN),
        _cut_layer_id(-1),
        _is_default(false),
        _bottom_layer_shape(nullptr),
        _top_layer_shape(nullptr) {
    _cut_layer_shape_list = std::make_unique<IdbTechRectList>();
    _origin               = std::make_unique<IdbTechPoint>(0, 0);
  }
  IdbTechVia::IdbTechVia(IdbTechVia &&other) {
    _name                 = other._name;
    _cut_num              = other._cut_num;
    _cut_layer_id         = other._cut_layer_id;
    _is_default           = other._is_default;
    _bottom_layer_shape   = std::move(other._bottom_layer_shape);
    _cut_layer_shape_list = std::move(other._cut_layer_shape_list);
    _top_layer_shape      = std::move(other._top_layer_shape);
  }
  IdbTechVia &IdbTechVia::operator=(IdbTechVia &&other) {
    _name                 = other._name;
    _cut_num              = other._cut_num;
    _cut_layer_id         = other._cut_layer_id;
    _is_default           = other._is_default;
    _bottom_layer_shape   = std::move(other._bottom_layer_shape);
    _cut_layer_shape_list = std::move(other._cut_layer_shape_list);
    _top_layer_shape      = std::move(other._top_layer_shape);
    return *this;
  }
  IdbTechRect *IdbTechVia::get_bottom_layer_shape() {
    if (!_bottom_layer_shape) {
      _bottom_layer_shape = std::make_unique<IdbTechRect>();
    }
    return _bottom_layer_shape.get();
  }

  IdbTechRect *IdbTechVia::get_top_layer_shape() {
    if (!_top_layer_shape) {
      _top_layer_shape = std::make_unique<IdbTechRect>();
    }
    return _top_layer_shape.get();
  }
  void IdbTechVia::setBottomRect(int llx, int lly, int urx, int ury) {
    if (!_bottom_layer_shape) {
      _bottom_layer_shape = std::make_unique<IdbTechRect>();
    }
    _bottom_layer_shape->setRectPoint(llx, lly, urx, ury);
  }

  void IdbTechVia::setTopRect(int llx, int lly, int urx, int ury) {
    if (!_top_layer_shape) {
      _top_layer_shape = std::make_unique<IdbTechRect>();
    }
    _top_layer_shape->setRectPoint(llx, lly, urx, ury);
  }

  bool IdbTechVia::isDefaultBottomHorizonTopVertical() {
    return _bottom_layer_default_direction == LayerDirEnum::kHORITIONAL &&
           _top_layer_default_direction == LayerDirEnum::kVERTICAL;
  }
  bool IdbTechVia::isDefaultBottomVerticalTopHorizon() {
    return _bottom_layer_default_direction == LayerDirEnum::kVERTICAL &&
           _top_layer_default_direction == LayerDirEnum::kHORITIONAL;
  }

  void IdbTechVia::cutViaCoordinateConversion(IdbTechPoint &newOrigin) {
    _bottom_layer_shape->coordinateConversion(newOrigin);
    _top_layer_shape->coordinateConversion(newOrigin);
    _cut_layer_shape_list->coordinateConversionForEveryVia(newOrigin);
  }
  void IdbTechVia::cutViaCoordinateConversion(int x, int y) {
    IdbTechRect *firstRect = _cut_layer_shape_list->getFirstRect();
    IdbTechRect *lastRect  = _cut_layer_shape_list->getLastRect();
    int firstLLx           = firstRect->getLowerLeftX();
    int firstLLy           = firstRect->getLowerLeftY();
    int lastURx            = lastRect->getUpperRightX();
    int lastURy            = lastRect->getUpperRightY();
    int referenceLeft      = firstLLx < lastURx ? firstLLx : lastURx;
    int referenceBottom    = firstLLy < lastURy ? firstLLy : lastURy;
    int referrnceRight     = firstLLx > lastURx ? firstLLx : lastURx;
    int referenceTop       = firstLLy > lastURy ? firstLLy : lastURy;
    int viaOffsetMidX      = (referrnceRight - referenceLeft) / 2;
    int viaOffsetMidY      = (referenceTop = referenceBottom) / 2;
    _bottom_layer_shape->coordinateConversion(x, y, viaOffsetMidX, viaOffsetMidY);
    _top_layer_shape->coordinateConversion(x, y, viaOffsetMidX, viaOffsetMidY);
    _cut_layer_shape_list->coordinateConversionForEveryVia(x, y, viaOffsetMidX, viaOffsetMidY);
  }
}  // namespace idb
