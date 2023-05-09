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
#include "guipin.h"

#include <QPainter>
#include <QStyleOptionGraphicsItem>

#include "guigraphicsview.h"

GuiPin::GuiPin(QGraphicsItem* parent) : GuiPolygon(new GuiPinPrivate(), parent), _data(static_cast<GuiPinPrivate*>(_d_ptr)) {
  set_pen(QColor(240, 210, 24));
  set_brush(QColor(240, 210, 24));
  // setFlag(QGraphicsItem::ItemIgnoresTransformations,true);
}

GuiPin::~GuiPin() { }

QRectF GuiPin::boundingRect() const { return isAllowToShow() ? _data->_rect : QRectF(); }

QPointF GuiPin::get_pos() const { return _data->_pos; }
void GuiPin::set_pos(qreal x_pos, qreal y_pos) {
  //_data->_pos = QPointF(x_pos, y_pos);
  //_data->_rect = QRectF(x_pos, y_pos, 8, 8);
  // set_rect(QRectF(x_pos, y_pos, 0.08, 0.08));
}

void GuiPin::set_IOPin(qreal x_indicator, qreal y_indicator, qreal pin_width, IdbConnectDirection direction,
                       RectEdgePosition pin_position) {
  switch (pin_position) {
    case RectEdgePosition::kLeft: setLeftPin(x_indicator, y_indicator, pin_width, direction); break;
    case RectEdgePosition::kRight: setRightPin(x_indicator, y_indicator, pin_width, direction); break;
    case RectEdgePosition::kTop: setTopPin(x_indicator, y_indicator, pin_width, direction); break;
    case RectEdgePosition::kBottom: setBottomPin(x_indicator, y_indicator, pin_width, direction); break;
    default: break;
  }
  _data->_rect = QRectF(x_indicator - pin_width, y_indicator - pin_width, pin_width * 2, pin_width * 2);
  //   update();
}

void GuiPin::setTopPin(qreal x_indicator, qreal y_indicator, qreal pin_width, IdbConnectDirection direction) {
  // header point -> right wing -> left wing (order by clockwise)
  /*
   * description: (e.g. output)
   *    header.point
   *         /\
   *        /  \
   *       /    \
   *      /      \
   *     /____.___\
   *  lwing   ↑    rwing
   *      indicator
   */
  QPointF p1, p2, p3, p4;
  switch (direction) {
    case IdbConnectDirection::kInput:
      p1.setX(x_indicator);
      p1.setY(y_indicator + pin_width);
      p2.setX(x_indicator - pin_width / 2);
      p2.setY(y_indicator);
      p3.setX(x_indicator + pin_width / 2);
      p3.setY(y_indicator);
      _data->_polygon.append(p1);
      _data->_polygon.append(p2);
      _data->_polygon.append(p3);
      break;
    case IdbConnectDirection::kOutput:
      p1.setX(x_indicator);
      p1.setY(y_indicator - pin_width);      // header point of the arrow
      p2.setX(x_indicator + pin_width / 2);  // right wing
      p2.setY(y_indicator);
      p3.setX(x_indicator - pin_width / 2);  // left wing
      p3.setY(y_indicator);
      _data->_polygon.append(p1);
      _data->_polygon.append(p2);
      _data->_polygon.append(p3);
      break;
      /*                 header/indicator
                          / \
                         /   \
             left wing  /     \  ring wing
                        \     /
                         \   /
                          \ /
                      tail point    */
    case IdbConnectDirection::kInOut:
      p1.setX(x_indicator);
      p1.setY(y_indicator + pin_width / 2);
      p2.setX(x_indicator + pin_width / 2);
      p2.setY(y_indicator);
      p3.setX(x_indicator);
      p3.setY(y_indicator - pin_width / 2);
      p4.setX(x_indicator - pin_width / 2);
      p4.setY(y_indicator);
      _data->_polygon.append(p1);
      _data->_polygon.append(p2);
      _data->_polygon.append(p3);
      _data->_polygon.append(p4);
      break;
    default: return;
  }

  _data->_rect;
}
void GuiPin::setBottomPin(qreal x_indicator, qreal y_indicator, qreal pin_width, IdbConnectDirection direction) {
  QPointF p1, p2, p3, p4;
  switch (direction) {
    case IdbConnectDirection::kInput:
      p1.setX(x_indicator);
      p1.setY(y_indicator - pin_width);      // header point of the arrow
      p2.setX(x_indicator + pin_width / 2);  // right wing
      p2.setY(y_indicator);
      p3.setX(x_indicator - pin_width / 2);  // left wing
      p3.setY(y_indicator);
      _data->_polygon.append(p1);
      _data->_polygon.append(p2);
      _data->_polygon.append(p3);
      break;
    case IdbConnectDirection::kOutput:
      p1.setX(x_indicator);
      p1.setY(y_indicator + pin_width);
      p2.setX(x_indicator - pin_width / 2);
      p2.setY(y_indicator);
      p3.setX(x_indicator + pin_width / 2);
      p3.setY(y_indicator);
      _data->_polygon.append(p1);
      _data->_polygon.append(p2);
      _data->_polygon.append(p3);
      break;

    case IdbConnectDirection::kInOut:
      p1.setX(x_indicator);
      p1.setY(y_indicator + pin_width / 2);
      p2.setX(x_indicator + pin_width / 2);
      p2.setY(y_indicator);
      p3.setX(x_indicator);
      p3.setY(y_indicator - pin_width / 2);
      p4.setX(x_indicator - pin_width / 2);
      p4.setY(y_indicator);
      _data->_polygon.append(p1);
      _data->_polygon.append(p2);
      _data->_polygon.append(p3);
      _data->_polygon.append(p4);
      break;
    default: return;
  }
  QPolygonF polygon;
}
void GuiPin::setLeftPin(qreal x_indicator, qreal y_indicator, qreal pin_width, IdbConnectDirection direction) {
  QPointF p1, p2, p3, p4;
  switch (direction) {
    case IdbConnectDirection::kInput:
      p1.setX(x_indicator + pin_width);
      p1.setY(y_indicator);
      p2.setX(x_indicator);
      p2.setY(y_indicator + pin_width / 2);
      p3.setX(x_indicator);
      p3.setY(y_indicator - pin_width / 2);
      _data->_polygon.append(p1);
      _data->_polygon.append(p2);
      _data->_polygon.append(p3);
      break;
    case IdbConnectDirection::kOutput:
      p1.setX(x_indicator - pin_width);
      p1.setY(y_indicator);
      p2.setX(x_indicator);
      p2.setY(y_indicator - pin_width / 2);
      p3.setX(x_indicator);
      p3.setY(y_indicator + pin_width / 2);
      _data->_polygon.append(p1);
      _data->_polygon.append(p2);
      _data->_polygon.append(p3);
      break;

    case IdbConnectDirection::kInOut:
      p1.setX(x_indicator);
      p1.setY(y_indicator + pin_width / 2);
      p2.setX(x_indicator + pin_width / 2);
      p2.setY(y_indicator);
      p3.setX(x_indicator);
      p3.setY(y_indicator - pin_width / 2);
      p4.setX(x_indicator - pin_width / 2);
      p4.setY(y_indicator);
      _data->_polygon.append(p1);
      _data->_polygon.append(p2);
      _data->_polygon.append(p3);
      _data->_polygon.append(p4);
      break;
    default: return;
  }
  QPolygonF polygon;
}
void GuiPin::setRightPin(qreal x_indicator, qreal y_indicator, qreal pin_width, IdbConnectDirection direction) {
  QPointF p1, p2, p3, p4;
  switch (direction) {
    case IdbConnectDirection::kInput:
      p1.setX(x_indicator - pin_width);
      p1.setY(y_indicator);
      p2.setX(x_indicator);
      p2.setY(y_indicator - pin_width / 2);
      p3.setX(x_indicator);
      p3.setY(y_indicator + pin_width / 2);
      _data->_polygon.append(p1);
      _data->_polygon.append(p2);
      _data->_polygon.append(p3);
      break;
    case IdbConnectDirection::kOutput:
      p1.setX(x_indicator + pin_width);
      p1.setY(y_indicator);
      p2.setX(x_indicator);
      p2.setY(y_indicator + pin_width / 2);
      p3.setX(x_indicator);
      p3.setY(y_indicator - pin_width / 2);
      _data->_polygon.append(p1);
      _data->_polygon.append(p2);
      _data->_polygon.append(p3);
      break;

    case IdbConnectDirection::kInOut:
      p1.setX(x_indicator);
      p1.setY(y_indicator + pin_width / 2);
      p2.setX(x_indicator + pin_width / 2);
      p2.setY(y_indicator);
      p3.setX(x_indicator);
      p3.setY(y_indicator - pin_width / 2);
      p4.setX(x_indicator - pin_width / 2);
      p4.setY(y_indicator);
      _data->_polygon.append(p1);
      _data->_polygon.append(p2);
      _data->_polygon.append(p3);
      _data->_polygon.append(p4);
      break;
    default: return;
  }
  QPolygonF polygon;
}

void GuiPin::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  if (isAllowToShow()) {
    _lod = option->levelOfDetailFromTransform(painter->worldTransform());

    //    double scaleFactor = 1.0 / _lod;
    //    painter->scale(scaleFactor, scaleFactor);
    //    update();
    //    if(lod<0.7){
    //      setFlag(QGraphicsItem::ItemIgnoresTransformations);
    //    }else{
    //      setFlag(QGraphicsItem::ItemIgnoresTransformations,false);
    //    }

    GuiPolygon::paint(painter, option, widget);
  }

  setFlag(QGraphicsItem::ItemIsSelectable, isAllowToSelect());
}

inline bool GuiPin::isAllowToShow() const { return true; }
inline bool GuiPin::isAllowToSelect() const { return true; }
