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
#ifndef ARROWDRC_H
#define ARROWDRC_H
#include <QGraphicsItem>
#include "guiitem.h"
#include "guirect.h"

class ArrowDrcPrivate;
class ArrowDrc : public GuiRect
{
 public:
  explicit ArrowDrc(QGraphicsItem *parent = nullptr);

  void set_drc(qreal x_pos,qreal y_pos);

  void paint(QPainter*                       painter,
             const QStyleOptionGraphicsItem* option,
             QWidget*                        widget) override;

  QRectF       boundingRect() const override;
  QPainterPath shape() const override;

 protected:
  ArrowDrc(ArrowDrcPrivate *data,QGraphicsItem *parent = nullptr);
  //  inline bool isAllowToShow() const override;
  //  inline bool isAllowToSelect() const override;

 private:
  ArrowDrcPrivate *_data;
};

class ArrowDrcPrivate : public GuiRectPrivate {
 public:
  explicit ArrowDrcPrivate();
  ~ArrowDrcPrivate();

 private:
  friend class ArrowDrc;
};
#endif // ARROWDRC_H
