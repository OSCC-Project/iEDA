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
 * @file GuiSpeedupClockTreeItem.h
 * @author Yell

 * @brief
 * @version 0.1
 * @date 2023-04-19(V0.1)
 *
 *
 *
 */

#ifndef GUI_SPEEDUP_CLK_ITEM
#define GUI_SPEEDUP_CLK_ITEM
#include <QBrush>
#include <QColor>
#include <QGraphicsItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsPolygonItem>
#include <QGraphicsRectItem>
#include <QPainter>
#include <QPen>
#include <QString>
#include <QStyleOptionGraphicsItem>
#include <set>

#include "IdbEnum.h"
#include "IdbGeometry.h"
#include "guiattribute.h"
#include "guigraphicsscene.h"
#include "omp.h"

enum class GuiClockType { kNone, kRoot, kLeaf, kNode, kMax };

class GuiSpeedupClockTreeItem : public QGraphicsRectItem {
 public:
  explicit GuiSpeedupClockTreeItem(GuiClockType type, QGraphicsRectItem* parent = nullptr)
      : QGraphicsRectItem(parent), _type(type) {
    switch (type) {
      case GuiClockType::kRoot: setBrush(QBrush(QColor(0, 0, 255), Qt::BrushStyle::SolidPattern)); break;
      case GuiClockType::kLeaf: setBrush(QBrush(QColor(255, 0, 0), Qt::BrushStyle::SolidPattern)); break;
      case GuiClockType::kNode: setBrush(QBrush(QColor(0, 255, 0), Qt::BrushStyle::SolidPattern)); break;
      default: setBrush(QBrush(QColor(255, 255, 255), Qt::BrushStyle::SolidPattern)); break;
    }
    _brush = brush();
    _pen   = pen();
    setZValue(0);
  }
  ~GuiSpeedupClockTreeItem() { clear(); }
  QRectF boundingRect() const override;
  QPainterPath shape() const override;
  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

  /// getter
  QRectF get_bounding_rect() { return boundingRect(); }
  bool is_visible();
  bool has_capacity() { return (_point_list.size() + _polygon_list.size() + _leaf_list.size()) > 200 ? false : true; }
  GuiClockType get_type() { return _type; }
  qreal get_poly_width() { return _poly_width; }

  /// setter
  void set_poly_width(qreal poly_width) { _poly_width = poly_width; }
  void add_point(QPointF point_1, QPointF point_2) {
    _point_list.emplace_back(std::make_pair(point_1, point_2));

    updateBoundingBox(point_1);
    updateBoundingBox(point_2);
  }
  void add_point(qreal ll_x, qreal ll_y, qreal ur_x, qreal ur_y) {
    _point_list.emplace_back(std::make_pair(QPointF(ll_x, ll_y), QPointF(ur_x, ur_y)));

    updateBoundingBox(ll_x, ll_y);
    updateBoundingBox(ur_x, ur_y);
  }

  void add_polygon(QPointF pt) {
    qreal width = _poly_width * 10;
    /// construct 3 points
    // QPolygonF polygon;
    // polygon.push_back(QPointF(pt.x() - width, pt.y() - width));
    // polygon.push_back(QPointF(pt.x() + width, pt.y() - width));
    // polygon.push_back(QPointF(pt.x(), pt.y() + width));
    // _polygon_list.push_back(polygon);
    _polygon_list.emplace_back(pt);

    updateBoundingBox(QPointF(pt.x() - width, pt.y() - width));
    updateBoundingBox(QPointF(pt.x() + width, pt.y() - width));
    updateBoundingBox(QPointF(pt.x(), pt.y() + width));
  }

  void add_leaf(QPointF pt) {
    /// construct 3 points
    // _leaf_list.emplace_back(QRectF(pt.x() - _poly_width, pt.y() - _poly_width, _poly_width * 2, _poly_width * 2));

    _leaf_list.emplace_back(pt);

    updateBoundingBox(QPointF(pt.x() - _poly_width, pt.y() - _poly_width));
    updateBoundingBox(QPointF(pt.x() + _poly_width, pt.y() + _poly_width));
  }

  /// operator
  void clear() {
    _point_list.clear();
    _polygon_list.clear();
    _leaf_list.clear();
  }

  /// paiter
  void paintScale(QPainter* painter, qreal lod);

  void paintScaleTop(QPainter* painter, qreal lod);
  void paintScale_1st(QPainter* painter, qreal lod);
  void paintScale_2nd(QPainter* painter, qreal lod);
  void paintScale_3rd(QPainter* painter, qreal lod);
  void drawText(QPainter* painter, QRectF rect, QString str, const qreal lod);

 private:
  QPen _pen;
  QBrush _brush;
  qreal _poly_width = 10;
  GuiClockType _type;

  std::vector<std::pair<QPointF, QPointF>> _point_list;
  //   std::vector<QPolygonF> _polygon_list;
  //   std::vector<QRectF> _leaf_list;
  std::vector<QPointF> _polygon_list;
  std::vector<QPointF> _leaf_list;

  qreal _ll_x = INT32_MAX;
  qreal _ll_y = INT32_MAX;
  qreal _ur_x = 0;
  qreal _ur_y = 0;

  void updateBoundingBox(qreal x, qreal y) {
    _ll_x = std::min(_ll_x, x);
    _ll_y = std::min(_ll_y, y);
    _ur_x = std::max(_ur_x, x);
    _ur_y = std::max(_ur_y, y);
  }

  void updateBoundingBox(QPointF point) {
    _ll_x = std::min(_ll_x, point.x());
    _ll_y = std::min(_ll_y, point.y());
    _ur_x = std::max(_ur_x, point.x());
    _ur_y = std::max(_ur_y, point.y());
  }
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class GuiSpeedupClockTreeItemList {
 public:
  GuiSpeedupClockTreeItemList(GuiGraphicsScene* scene) { _scene = scene; }
  virtual ~GuiSpeedupClockTreeItemList() { clear(); }

  /// getter
  GuiGraphicsScene* get_scene() { return _scene; }
  std::vector<GuiSpeedupClockTreeItem*>& get_item_list() { return _item_list; }

  /// setter
  GuiSpeedupClockTreeItem* add_item(GuiClockType type) {
    GuiSpeedupClockTreeItem* item = nullptr;
    switch (type) {
      case GuiClockType::kRoot: {
        item = _root_item;
      } break;
      case GuiClockType::kLeaf: {
        item = _leaf_item;
      } break;
      case GuiClockType::kNode: {
        item = _node_item;
      } break;
      default: item = _node_item; break;
    }

    if (item == nullptr || !item->has_capacity()) {
      item = new GuiSpeedupClockTreeItem(type);
      _item_list.push_back(item);
      _scene->addItem(item);
    }

    switch (type) {
      case GuiClockType::kRoot: {
        _root_item = item;
      } break;
      case GuiClockType::kLeaf: {
        _leaf_item = item;
      } break;
      case GuiClockType::kNode: {
        _node_item = item;
      } break;
      default: {
        _node_item = item;
      } break;
    }

    return item;
  }

  void clear() {
    for (GuiSpeedupClockTreeItem* item : _item_list) {
      if (item != nullptr) {
        delete item;
        item = nullptr;
      }
    }
    _item_list.clear();
  }

  /// gui
  void update() {
#pragma omp parallel for
    for (auto item : _item_list) {
      item->update();
    }
  }

 private:
  GuiGraphicsScene* _scene;
  GuiSpeedupClockTreeItem* _root_item = nullptr;
  GuiSpeedupClockTreeItem* _node_item = nullptr;
  GuiSpeedupClockTreeItem* _leaf_item = nullptr;
  std::vector<GuiSpeedupClockTreeItem*> _item_list;
};

#endif  // GUI_SPEEDUP_CLK_ITEM
