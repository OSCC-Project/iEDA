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
 * @file GuiSpeedupVia.h
 * @author Yell

 * @brief
 * @version 0.1
 * @date 2021-11-29(V0.1)
 *
 *
 *
 */

#ifndef GUI_SPEEDUP_VIA
#define GUI_SPEEDUP_VIA

#include "guiattribute.h"
#include "guispeedupitem.h"
#include "guispeedupwire.h"
#include "omp.h"

class GuiSpeedupVia : public GuiSpeedupItem {
 public:
  explicit GuiSpeedupVia(QColor color, QColor color_top, QColor color_bottom, int32_t z_order,
                         GuiSpeedupItem* parent = nullptr)
      : GuiSpeedupItem(parent) {
    setPen(color);
    setBrush(QBrush(color, Qt::BrushStyle::DiagCrossPattern));
    _brush = brush();
    _pen   = pen();
    setZValue(z_order);
    _enclosure_top    = new GuiSpeedupWire(color_top, z_order + 1, this, GuiSpeedupItemType::kVia);
    _enclosure_bottom = new GuiSpeedupWire(color_bottom, z_order - 1, this, GuiSpeedupItemType::kVia);
  }

  virtual ~GuiSpeedupVia() { clear(); }

  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

  /// getter
  virtual bool has_capacity() override { return get_rect_number() < GUI_ITEM_VIA_MAX ? true : false; }
  virtual bool is_visible() override;
  GuiSpeedupWire* get_enclosure_top() { return _enclosure_top; }
  GuiSpeedupWire* get_enclosure_bottom() { return _enclosure_bottom; }

  /// setter

  /// opreator
  virtual GuiSpeedupItem* clone() override {
    GuiSpeedupVia* new_item = new GuiSpeedupVia(this->pen().color(), this->get_enclosure_top()->pen().color(),
                                                this->get_enclosure_bottom()->pen().color(), this->zValue(),
                                                dynamic_cast<GuiSpeedupItem*>(this->parentItem()));
    new_item->set_bounding_rect(get_bounding_rect());
    new_item->get_enclosure_top()->set_bounding_rect(get_bounding_rect());
    new_item->get_enclosure_bottom()->set_bounding_rect(get_bounding_rect());
    new_item->set_type(get_type());
    new_item->get_enclosure_top()->set_type(_enclosure_top->get_type());
    new_item->get_enclosure_bottom()->set_type(_enclosure_bottom->get_type());
    return dynamic_cast<GuiSpeedupItem*>(new_item);
  }

  virtual void clear() {
    if (_enclosure_top != nullptr) {
      delete _enclosure_top;
      _enclosure_top = nullptr;
    }

    if (_enclosure_bottom != nullptr) {
      delete _enclosure_bottom;
      _enclosure_bottom = nullptr;
    }
    GuiSpeedupItem::clear();
  }

  /// painter
  virtual void paintScaleTop(QPainter* painter, qreal lod);
  virtual void paintScale_1st(QPainter* painter, qreal lod);
  virtual void paintScale_2nd(QPainter* painter, qreal lod);
  virtual void paintScale_3rd(QPainter* painter, qreal lod);

 private:
  QPen _pen;
  QBrush _brush;
  GuiSpeedupWire* _enclosure_top;
  GuiSpeedupWire* _enclosure_bottom;
};

class GuiSpeedupViaList : public GuiSpeedupItemList {
 public:
  GuiSpeedupViaList(GuiGraphicsScene* scene) : GuiSpeedupItemList(scene, GuiSpeedupItemType::kVia){};
  ~GuiSpeedupViaList() = default;

  /// getter
  QString get_layer_name() { return _layer_name; }
  /// find item by coordinate
  GuiSpeedupVia* findItem(QPointF pt) { return dynamic_cast<GuiSpeedupVia*>(get_item(pt)); }

  /// setter
  void set_color(QColor color) { _color = color; }
  void set_layer_name(QString name) { _layer_name = name; }
  void set_order(int32_t order) { _z_order = order; }
  GuiSpeedupVia* addCurrentItem(GuiSpeedupItem* parent = nullptr) {
    GuiSpeedupItem* new_item = new GuiSpeedupVia(_color, attributeInst->getLayerColor(_z_order + 1),
                                                 attributeInst->getLayerColor(_z_order - 1), _z_order, parent);

    return dynamic_cast<GuiSpeedupVia*>(add_current_item(new_item));
  }

  /// operator
  virtual void init(QRectF boundingbox) override;
  virtual void finishCreateItem() override {
    for (GuiSpeedupItem* item : get_item_list()) {
      if (item != nullptr) {
        addSceneItem(item);
      }
    }
  }

  virtual void addSceneItem(GuiSpeedupItem* item) override {
    GuiSpeedupVia* via_item = dynamic_cast<GuiSpeedupVia*>(item);
    get_scene()->addItem(via_item);
    get_scene()->addItem(via_item->get_enclosure_top());
    get_scene()->addItem(via_item->get_enclosure_bottom());
  }

 private:
  QColor _color;
  QString _layer_name;
  int32_t _z_order;
};

class GuiSeedupViaContainer {
 public:
  GuiSeedupViaContainer(GuiGraphicsScene* scene) : _scene(scene) { }
  ~GuiSeedupViaContainer() { clear(); }

  /// getter
  GuiSpeedupViaList* findViaList(std::string layer) {
    QString layer_name = QString::fromStdString(layer);
    for (auto item : _via_container) {
      if (item->get_layer_name().toUpper() == layer_name.toUpper()) {
        return item;
      }
    }
    // std::cout << "Error : can not find gui via list for " << layer << std::endl;
    return nullptr;
  }

  /// setter
  GuiSpeedupViaList* addViaList() {
    GuiSpeedupViaList* via_list = new GuiSpeedupViaList(_scene);
    _via_container.emplace_back(via_list);

    return via_list;
  }

  /// operator
  virtual void finishCreateItem() {
    for (GuiSpeedupViaList* via_list : _via_container) {
      via_list->finishCreateItem();
    }
  }

  void clear() {
    for (auto via_list : _via_container) {
      if (via_list != nullptr) {
        delete via_list;
        via_list = nullptr;
      }
    }
  }

  void update() {
#pragma omp parallel for
    for (auto via_list : _via_container) {
      via_list->update();
    }
  }

  /// test
  int32_t number_create() {
    int32_t number = 0;
    for (GuiSpeedupViaList* item_list : _via_container) {
      number += item_list->get_number_create();
    }
    return number;
  }

  int32_t number_not_find() {
    int32_t number = 0;
    for (GuiSpeedupViaList* item_list : _via_container) {
      number += item_list->get_number_not_find();
    }
    return number;
  }

 private:
  GuiGraphicsScene* _scene;
  std::vector<GuiSpeedupViaList*> _via_container;
};

#endif  // GUI_SPEEDUP_VIA
