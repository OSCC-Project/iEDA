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
 * @file GuiSpeedupWire.h
 * @author Yell

 * @brief
 * @version 0.1
 * @date 2021-11-29(V0.1)
 *
 *
 *
 */

#ifndef GUI_SPEEDUP_WIRE
#define GUI_SPEEDUP_WIRE

#include "guispeedupitem.h"
#include "omp.h"

class GuiSpeedupWire : public GuiSpeedupItem {
 public:
  explicit GuiSpeedupWire(QColor color, int32_t z_order, GuiSpeedupItem* parent = nullptr,
                          GuiSpeedupItemType type = GuiSpeedupItemType::kNone)
      : GuiSpeedupItem(parent, type) {
    setPen(color);

    switch (type) {
      case GuiSpeedupItemType::kVia: {
        setBrush(QBrush(color, Qt::BrushStyle::DiagCrossPattern));
      } break;
      case GuiSpeedupItemType::kSignalPower:
      case GuiSpeedupItemType::kSignalGround:
      case GuiSpeedupItemType::kPdn:
      case GuiSpeedupItemType::kPdnPower:
      case GuiSpeedupItemType::kPdnGround: {
        setBrush(QBrush(color, Qt::BrushStyle::CrossPattern));
      } break;
      case GuiSpeedupItemType::kInstance:
      case GuiSpeedupItemType::kInstStandarCell:
      case GuiSpeedupItemType::kInstIoCell:
      case GuiSpeedupItemType::kInstBlock:
      case GuiSpeedupItemType::kInstPad: {
        setBrush(QBrush(color, Qt::BrushStyle::Dense6Pattern));
      } break;
      default: {
        if (z_order % 2 == 0) {
          setBrush(QBrush(color, Qt::BrushStyle::BDiagPattern));
        } else {
          setBrush(QBrush(color, Qt::BrushStyle::FDiagPattern));
        }
      } break;
    }

    _brush = brush();
    _pen   = pen();
    setZValue(z_order);
  }

  virtual ~GuiSpeedupWire() = default;

  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

  /// getter
  virtual bool has_capacity() override { return get_rect_number() + get_points_number() < GUI_ITEM_WIRE_MAX ? true : false; }
  virtual bool is_visible() override;
  bool is_port() { return _b_port; }
  bool is_pdn();

  /// setter
  void set_type_net() { set_type(GuiSpeedupItemType::kNet); }
  void set_type_pdn() { set_type(GuiSpeedupItemType::kPdn); }
  void set_type_signal() { set_type(GuiSpeedupItemType::kSignal); }
  void set_type_clock() { set_type(GuiSpeedupItemType::kSignalClock); }
  void set_as_port() { _b_port = true; }

  /// opreator
  virtual GuiSpeedupItem* clone() override {
    GuiSpeedupWire* new_item = new GuiSpeedupWire(this->pen().color(), this->zValue(),
                                                  dynamic_cast<GuiSpeedupItem*>(this->parentItem()), get_type());
    new_item->set_bounding_rect(get_bounding_rect());
    return dynamic_cast<GuiSpeedupItem*>(new_item);
  }

  /// painter
  virtual void paintScaleTop(QPainter* painter, qreal lod);
  virtual void paintScale_1st(QPainter* painter, qreal lod);
  virtual void paintScale_2nd(QPainter* painter, qreal lod);
  virtual void paintScale_3rd(QPainter* painter, qreal lod);

 private:
  QPen _pen;
  QBrush _brush;
  bool _b_port = false;
};

class GuiSpeedupWireList : public GuiSpeedupItemList {
 public:
  GuiSpeedupWireList(GuiGraphicsScene* scene, GuiSpeedupItemType type)
      : GuiSpeedupItemList(scene, type){

        };
  ~GuiSpeedupWireList() { clear(); }

  /// getter
  QString get_layer_name() { return _layer_name; }
  GuiSpeedupWire* findItem(QPointF pt) { return dynamic_cast<GuiSpeedupWire*>(get_item(pt)); }
  GuiSpeedupWire* findItem(QPointF pt1, QPointF pt2) { return dynamic_cast<GuiSpeedupWire*>(get_item(pt1, pt2)); }
  //   GuiSpeedupWire* findItem(QPointF pt1, QPointF pt2) {
  //     return findItem(QPointF((pt1.x() + pt2.x()) / 2, (pt1.y() + pt2.y()) / 2));
  //   }

  /// setter
  void set_color(QColor color) { _color = color; }
  void set_layer_name(QString name) { _layer_name = name; }
  void set_order(int32_t order) { _z_order = order; }
  GuiSpeedupWire* addCurrentItem(GuiSpeedupItem* parent = nullptr) {
    GuiSpeedupItem* new_item = new GuiSpeedupWire(_color, _z_order, parent, get_type());

    return dynamic_cast<GuiSpeedupWire*>(add_current_item(new_item));
  }

  /// operator
  void init(QRectF boundingbox, IdbLayerDirection direction);
  void initPanel(QRectF boundingbox, IdbLayerDirection direction);

 private:
  QColor _color;
  QString _layer_name;
  int32_t _z_order;
};

class GuiSpeedupWireContainer {
 public:
  GuiSpeedupWireContainer(GuiGraphicsScene* scene, GuiSpeedupItemType type) : _scene(scene), _type(type) { }
  ~GuiSpeedupWireContainer() = default;

  /// getter
  GuiSpeedupWireList* findWireList(std::string layer) {
    QString layer_name = QString::fromStdString(layer);
    for (auto item : _wire_container) {
      if (item->get_layer_name().toUpper() == layer_name.toUpper()) {
        return item;
      }
    }
    // std::cout << "Error : can not find gui wire list for " << layer << std::endl;
    return nullptr;
  }

  /// setter
  GuiSpeedupWireList* addWireList(GuiSpeedupItemType type) {
    GuiSpeedupWireList* wire_list = new GuiSpeedupWireList(_scene, type);
    _wire_container.emplace_back(wire_list);

    return wire_list;
  }

  ////operator
  virtual void finishCreateItem() {
    for (GuiSpeedupWireList* wire_list : _wire_container) {
      wire_list->finishCreateItem();
    }
  }

  int32_t number_create() {
    int32_t number = 0;
    for (GuiSpeedupWireList* wire_list : _wire_container) {
      number += wire_list->get_number_create();
    }
    return number;
  }

  int32_t number_not_find() {
    int32_t number = 0;
    for (GuiSpeedupWireList* wire_list : _wire_container) {
      number += wire_list->get_number_not_find();
    }
    return number;
  }

  void clear() {
    for (auto wire_list : _wire_container) {
      if (wire_list != nullptr) {
        delete wire_list;
        wire_list = nullptr;
      }
    }
  }

  void update() {
#pragma omp parallel for
    for (auto wire_list : _wire_container) {
      wire_list->update();
    }
  }

 private:
  GuiGraphicsScene* _scene;
  std::vector<GuiSpeedupWireList*> _wire_container;
  GuiSpeedupItemType _type;
};

#endif  // GUI_SPEEDUP_WIRE
