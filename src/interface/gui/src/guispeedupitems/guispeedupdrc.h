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
 * @file guispeedupdrc.h
 * @author Yell

 * @brief
 * @version 0.1
 * @date 2021-11-29(V0.1)
 *
 *
 *
 */

#ifndef GUI_SPEEDUP_DRC
#define GUI_SPEEDUP_DRC
#include <map>
#include <vector>

#include "guispeedupitem.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class GuiSpeedupDrc : public GuiSpeedupItem {
 public:
  explicit GuiSpeedupDrc(int32_t z_order, GuiSpeedupItem* parent = nullptr,
                         GuiSpeedupItemType type = GuiSpeedupItemType::kNone)
      : GuiSpeedupItem(parent, type) {
    setPen(QColor(255, 255, 255));
    setBrush(QBrush(QColor(255, 255, 255), Qt::BrushStyle::SolidPattern));
    _brush = brush();
    _pen   = pen();
    setZValue(100);
    _z_order = z_order;
  }

  virtual ~GuiSpeedupDrc() { clear(); }

  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

  virtual bool is_visible() override;
  virtual bool has_capacity() override { return get_rect_number() < GUI_ITEM_INSTANCE_MAX ? true : false; }

  /// getter
  int32_t get_z_order() { return _z_order; }
  void set_z_order(int z_order) { _z_order = z_order; }

  /// setter

  /// operator

  virtual GuiSpeedupItem* clone() override {
    GuiSpeedupDrc* new_item =
        new GuiSpeedupDrc(this->get_z_order(), dynamic_cast<GuiSpeedupItem*>(this->parentItem()), get_type());
    new_item->set_bounding_rect(get_bounding_rect());
    return dynamic_cast<GuiSpeedupItem*>(new_item);
  }

  virtual void clear() { GuiSpeedupItem::clear(); }

  /// painter
  virtual void paintScaleTop(QPainter* painter, qreal lod);
  virtual void paintScale_1st(QPainter* painter, qreal lod);
  virtual void paintScale_2nd(QPainter* painter, qreal lod);
  virtual void paintScale_3rd(QPainter* painter, qreal lod);

 private:
  int _z_order = 0;
  QPen _pen;
  QBrush _brush;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class GuiSpeedupDrcList : public GuiSpeedupItemList {
 public:
  GuiSpeedupDrcList(GuiGraphicsScene* scene, GuiSpeedupItemType type) : GuiSpeedupItemList(scene, type){};
  ~GuiSpeedupDrcList() = default;

  /// getter
  QString get_layer_name() { return _layer_name; }
  /// find item by coordinate
  GuiSpeedupDrc* findItem(QPointF pt) { return dynamic_cast<GuiSpeedupDrc*>(get_item(pt)); }

  /// setter
  void set_layer_name(QString name) { _layer_name = name; }
  void set_order(int32_t order) { _z_order = order; }

  GuiSpeedupDrc* addCurrentItem(GuiSpeedupItem* parent = nullptr) {
    GuiSpeedupItem* new_item = new GuiSpeedupDrc(_z_order, parent, get_type());

    return dynamic_cast<GuiSpeedupDrc*>(add_current_item(new_item));
  }

  /// Operator
  void init(QRectF boundingbox, int z_order);
  virtual void addSceneItem(GuiSpeedupItem* item) {
    GuiSpeedupDrc* instance_item = dynamic_cast<GuiSpeedupDrc*>(item);

    GuiSpeedupItemList::addSceneItem(item);
  }

 private:
  int32_t _z_order;
  QString _layer_name;
};

class GuiSpeedupDrcContainer {
 public:
  GuiSpeedupDrcContainer(GuiGraphicsScene* scene, GuiSpeedupItemType type) : _scene(scene), _type(type) { }
  ~GuiSpeedupDrcContainer() = default;

  /// getter
  GuiSpeedupItemType get_type() { return _type; }

  GuiSpeedupDrcList* findDrcList(std::string layer) {
    QString layer_name = QString::fromStdString(layer);
    for (auto item : _drc_container) {
      if (item->get_layer_name().toUpper() == layer_name.toUpper()) {
        return item;
      }
    }
    // std::cout << "Error : can not find gui wire list for " << layer << std::endl;
    return nullptr;
  }

  /// setter
  GuiSpeedupDrcList* addDrcList(GuiSpeedupItemType type) {
    GuiSpeedupDrcList* drc_list = new GuiSpeedupDrcList(_scene, type);
    _drc_container.emplace_back(drc_list);

    return drc_list;
  }

  ////operator
  virtual void finishCreateItem() {
    for (auto drc_list : _drc_container) {
      drc_list->finishCreateItem();
    }
  }

  int32_t number_create() {
    int32_t number = 0;
    for (auto drc_list : _drc_container) {
      number += drc_list->get_number_create();
    }
    return number;
  }

  int32_t number_not_find() {
    int32_t number = 0;
    for (auto drc_list : _drc_container) {
      number += drc_list->get_number_not_find();
    }
    return number;
  }

  void clear() {
    for (auto drc_list : _drc_container) {
      if (drc_list != nullptr) {
        delete drc_list;
        drc_list = nullptr;
      }
    }

    _drc_container.clear();
  }

  void update() {
#pragma omp parallel for
    for (auto drc_list : _drc_container) {
      drc_list->update();
    }
  }

 private:
  GuiGraphicsScene* _scene;

  //   std::map<std::string, GuiSpeedupDrcList*> _drc_container;
  std::vector<GuiSpeedupDrcList*> _drc_container;
  GuiSpeedupItemType _type = GuiSpeedupItemType::kDrc;
};

#endif  // GUI_SPEEDUP_DRC
