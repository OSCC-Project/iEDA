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
 * @file guiline.cpp
 * @author Wang Jun (wen8365@gmail.com)
 * @brief
 * @version 0.1
 * @date 2021-07-02
 *
 *
 *
 */

#include "guiline.h"

#include <QPainter>
static QPainterPath shapeFromPath(const QPainterPath &path, const QPen &pen);
GuiLine::GuiLine(QGraphicsItem *parent)
    : GuiItem(new GuiLinePrivate(), parent) {
  _data = static_cast<GuiLinePrivate *>(_d_ptr);
}
GuiLine::GuiLine(GuiLinePrivate *data, QGraphicsItem *parent)
    : GuiItem(data, parent) {
  _data = static_cast<GuiLinePrivate *>(_d_ptr);
}
GuiLine::~GuiLine() {}

QLineF GuiLine::get_line() const  {
    return _data->_line;
}
void GuiLine::set_line(const QLineF &line){
    _data->_line = line;
}

QRectF GuiLine::boundingRect() const{
    if (_data->_pen.widthF() == 0.0) {
        const qreal x1 = _data->_line.p1().x();
        const qreal x2 = _data->_line.p2().x();
        const qreal y1 = _data->_line.p1().y();
        const qreal y2 = _data->_line.p2().y();
        qreal lx = qMin(x1, x2);
        qreal rx = qMax(x1, x2);
        qreal ty = qMin(y1, y2);
        qreal by = qMax(y1, y2);
        return QRectF(lx, ty, rx - lx, by - ty);
    }
    return shape().controlPointRect();
}

QPainterPath GuiLine::shape() const{
    QPainterPath path;
    if (_data->_line == QLineF())
        return path;
    path.moveTo(_data->_line.p1());
    path.lineTo(_data->_line.p2());
    return shapeFromPath(path, _data->_pen);
}

void GuiLine::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                    QWidget *widget) {
  painter->setPen(_data->_pen);
  painter->drawLine(_data->_line);
}
GuiLinePrivate::GuiLinePrivate() : _line(QLineF()){}
GuiLinePrivate::~GuiLinePrivate() {}
static QPainterPath shapeFromPath(const QPainterPath &path, const QPen &pen)
{
    const qreal penWidthZero = qreal(0.00000001);
    if (path == QPainterPath() || pen == Qt::NoPen)
        return path;
    QPainterPathStroker ps;
    ps.setCapStyle(pen.capStyle());
    if (pen.widthF() <= 0.0)
        ps.setWidth(penWidthZero);
    else
        ps.setWidth(pen.widthF());
    ps.setJoinStyle(pen.joinStyle());
    ps.setMiterLimit(pen.miterLimit());
    QPainterPath p = ps.createStroke(path);
    p.addPath(path);
    return p;
}
