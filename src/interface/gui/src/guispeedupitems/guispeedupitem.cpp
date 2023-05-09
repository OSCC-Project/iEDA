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
#include "guispeedupitem.h"

void GuiSpeedupItem::adjust_bouding_rect(QRectF rect) {
  //   qreal ll_x = std::min(_bounding_box.left(), rect.left());
  //   qreal ll_y = std::min(_bounding_box.top(), rect.top());
  //   qreal ur_x = std::max(_bounding_box.right(), rect.right());
  //   qreal ur_y = std::max(_bounding_box.bottom(), rect.bottom());

  //   _bounding_box.setRect(ll_x, ll_y, ur_x - ll_x, ur_y - ll_y);
}

QRectF GuiSpeedupItem::boundingRect() const {
  qreal ll_x = INT32_MAX;
  qreal ll_y = INT32_MAX;
  qreal ur_x = 0;
  qreal ur_y = 0;
  //   qreal ll_x = _bounding_box.left() - 1;
  //   qreal ll_y = _bounding_box.top() - 1;
  //   qreal ur_x = _bounding_box.right() + 1;
  //   qreal ur_y = _bounding_box.bottom() + 1;

  for (auto& rect : _rect_list) {
    ll_x = std::min(ll_x, rect.left());
    ll_y = std::min(ll_y, rect.top());
    ur_x = std::max(ur_x, rect.right());
    ur_y = std::max(ur_y, rect.bottom());
  }

  for (auto& point : _point_list) {
    ll_x = std::min(ll_x, point.first.x());
    ll_x = std::min(ll_x, point.second.x());
    ll_y = std::min(ll_y, point.first.y());
    ll_y = std::min(ll_y, point.second.y());
    ur_x = std::max(ur_x, point.first.x());
    ur_x = std::max(ur_x, point.second.x());
    ur_y = std::max(ur_y, point.first.y());
    ur_y = std::max(ur_y, point.second.y());
  }

  return QRectF(ll_x - 1, ll_y - 1, ur_x - ll_x + 2, ur_y - ll_y + 2);
  //   return _bounding_box;
}

QPainterPath GuiSpeedupItem::shape() const {
  QPainterPath path;
  path.addRect(_bounding_box);
  return path;
}

void GuiSpeedupItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  Q_UNUSED(widget);

  if (is_clock_tree_visible()) {
    return;
  }

  if (!is_visible()) {
    return;
  }

  const qreal lod = option->levelOfDetailFromTransform(painter->worldTransform());

  if (lod < 1) {
    // if (_image_top != nullptr) {
    //   //   painter->drawPixmap(_bounding_box.topLeft(), *_image);
    //   //   painter->drawImage(0, 0, *_image_top);
    //   painter->drawImage(static_cast<int>(boundingRect().x()), static_cast<int>(boundingRect().y()), *_image_top);
    // } else {
    //   //   paintScaleTop(painter, lod);
    //   paintImageTop(lod);
    //   //   painter->drawImage(0, 0, *_image_top);
    //   painter->drawImage(static_cast<int>(boundingRect().x()), static_cast<int>(boundingRect().y()), *_image_top);
    // }

    paintScaleTop(painter, lod);

  } else if (lod >= 1 && lod < 10) {
    // if (_image_scale_1 != nullptr) {
    //   //   painter->drawPixmap(_bounding_box.topLeft(), *_image);
    //   //   painter->drawPixmap(0, 0, *_image_scale_1);
    //   painter->drawImage(static_cast<int>(boundingRect().x()), static_cast<int>(boundingRect().y()), *_image_scale_1);
    // } else {
    //   //   paintScaleTop(painter, lod);
    //   paintImageSacel_1st(lod);
    //   //   painter->drawPixmap(0, 0, *_image_scale_1);
    //   painter->drawImage(static_cast<int>(boundingRect().x()), static_cast<int>(boundingRect().y()), *_image_scale_1);
    // }

    paintScale_1st(painter, lod);
  } else if (lod >= 10 && lod < 50) {
    // if (_image_scale_2 != nullptr) {
    //   //   painter->drawPixmap(_bounding_box.topLeft(), *_image);
    //   //   painter->drawPixmap(0, 0, *_image_scale_2);
    //   painter->drawImage(static_cast<int>(boundingRect().x()), static_cast<int>(boundingRect().y()), *_image_scale_2);
    // } else {
    //   //   paintScaleTop(painter, lod);
    //   paintImageScale_2nd(lod);
    //   //   painter->drawPixmap(0, 0, *_image_scale_2);
    //   painter->drawImage(static_cast<int>(boundingRect().x()), static_cast<int>(boundingRect().y()), *_image_scale_2);
    // }

    paintScale_2nd(painter, lod);
  } else {
    // if (_image_scale_3 != nullptr) {
    //   //   painter->drawPixmap(_bounding_box.topLeft(), *_image);
    //   //   painter->drawPixmap(0, 0, *_image_scale_3);
    //   painter->drawImage(static_cast<int>(boundingRect().x()), static_cast<int>(boundingRect().y()), *_image_scale_3);
    // } else {
    //   //   paintScaleTop(painter, lod);
    //   paintImageScale_3rd(lod);
    //   //   painter->drawPixmap(0, 0, *_image_scale_3);
    //   painter->drawImage(static_cast<int>(boundingRect().x()), static_cast<int>(boundingRect().y()), *_image_scale_3);
    // }

    paintScale_3rd(painter, lod);
  }

  QGraphicsRectItem::paint(painter, option, widget);
}

void GuiSpeedupItem::paintScaleTop(QPainter* painter, qreal lod) { }

void GuiSpeedupItem::paintScale_1st(QPainter* painter, qreal lod) { }

void GuiSpeedupItem::paintScale_2nd(QPainter* painter, qreal lod) { }

void GuiSpeedupItem::paintScale_3rd(QPainter* painter, qreal lod) { }

void GuiSpeedupItem::drawText(QPainter* painter, QRectF rect, QString str, const qreal lod) {
  return;
  QFont font("Arial", 1);
  painter->setFont(font);
  painter->save();
  QPen pen = painter->pen();
  pen.setWidthF(pen.widthF() / lod);
  pen.setColor(Qt::white);
  painter->setPen(pen);
  painter->scale(0.1, 0.1);
  painter->drawText(rect.center(), str);
  painter->restore();
}

bool GuiSpeedupItem::is_clock_tree_visible() {
  igui::GuiTreeNode& tree_node = guiConfig->get_clock_tree();
  return tree_node.isChecked("Show");
}