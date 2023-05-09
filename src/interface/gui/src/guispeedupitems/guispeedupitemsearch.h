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
#ifndef GUI_SPEEDUP_ITEM_SEARCH
#define GUI_SPEEDUP_ITEM_SEARCH
#include <QBrush>
#include <QColor>
#include <QGraphicsItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsRectItem>
#include <QPainter>
#include <QPen>
#include <QString>
#include <QStyleOptionGraphicsItem>

#include "IdbEnum.h"
#include "IdbGeometry.h"
#include "guiattribute.h"

class GuiSpeedupItemSearch : public QGraphicsRectItem {
 public:
  explicit GuiSpeedupItemSearch(QGraphicsRectItem* parent = nullptr) : QGraphicsRectItem(parent) {
    setPen(QColor(255, 255, 255));
    setBrush(QBrush(QColor(255, 255, 255), Qt::BrushStyle::NoBrush));
    _brush = brush();
    _pen   = pen();
  }
  virtual ~GuiSpeedupItemSearch() { }
  virtual QRectF boundingRect() const override;
  virtual QPainterPath shape() const override;
  virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

  /// getter
  QRectF get_bounding_rect() { return _bounding_box; }
  std::vector<QRectF>& get_rect_list() { return _rect_list; }
  std::vector<QRectF>& get_pin_rect_list() { return _pin_rect_list; }
  int32_t get_rect_number() { return _rect_list.size(); }
  int32_t get_pin_rect_number() { return _pin_rect_list.size(); }
  QPointF& get_average_pin_coord() { return _average_pin_coord; }
  std::vector<QPointF>& get_points() { return _pin_list; }
  int32_t get_points_number() { return _pin_list.size(); }

  /// setter
  void set_bounding_rect(QRectF bounding_box) { _bounding_box = bounding_box; }
  void add_rect(qreal ll_x, qreal ll_y, qreal witdh, qreal height) {
    _rect_list.emplace_back(QRectF(ll_x, ll_y, witdh, height));
  }
  void add_rect(QRectF rect) { _rect_list.emplace_back(rect); }

  void add_pin_rect(qreal ll_x, qreal ll_y, qreal witdh, qreal height) {
    _pin_rect_list.emplace_back(QRectF(ll_x, ll_y, witdh, height));
  }
  void add_pin_rect(QRectF rect) { _pin_rect_list.emplace_back(rect); }
  void set_average_pin_coord(QPointF average_pin_coord) { _average_pin_coord = average_pin_coord; }
  void add_point(QPointF point) { _pin_list.emplace_back(point); }
  void add_point(qreal x, qreal y) { _pin_list.emplace_back(QPointF(x, y)); }

  /// operator
  void clear() {
    _rect_list.clear();
    _pin_rect_list.clear();
    _pin_list.clear();
  }

  /// paiter

 private:
  QPen _pen;
  QBrush _brush;

  QRectF _bounding_box;

  std::vector<QRectF> _rect_list;
  std::vector<QRectF> _pin_rect_list;
  QPointF _average_pin_coord;
  std::vector<QPointF> _pin_list;
};

#endif  // GUI_SPEEDUP_ITEM_SEARCH
