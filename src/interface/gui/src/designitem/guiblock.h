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
 * @file guiblock.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief This class inherits from GuiOrientationRect to represent Block in the GUI
 * @version 0.2
 * @date 2021-07-09(V0.1) 2021-08-12(V0.2)
 */

#ifndef GUIBLOCK_H
#define GUIBLOCK_H
#include "guiinstance.h"
class GuiBlockPrivate;
class GuiBlock : public GuiInstance {
 public:
  explicit GuiBlock(QGraphicsItem *parent = nullptr);

  QPen get_halo_pen();
  void set_halo_pen(const QPen &halo_pen);
  void set_halo_pen(QColor color, qreal width = 0, Qt::PenStyle style = Qt::SolidLine);

  QBrush get_halo_brush();
  void set_halo_brush(const QBrush &halo_brush);
  void set_halo_brush(const QColor &color, Qt::BrushStyle style = Qt::SolidPattern);

  QRectF get_halo_rect();
  void set_halo_rect(const QRectF &halo_rect);
  inline void set_halo_rect(qreal x1, qreal y1, qreal x2, qreal y2);

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;

 protected:
  inline bool isAllowToShow() const override;
  inline bool isAllowToSelect() const override;

 private:
  GuiBlockPrivate *_data;
};
inline void GuiBlock::set_halo_rect(qreal x1, qreal y1, qreal x2, qreal y2) {
  set_halo_rect(QRectF(QPointF(x1, y1), QPointF(x2, y2)));
}

class GuiBlockPrivate : public GuiInstancePrivate {
 public:
  explicit GuiBlockPrivate();
  ~GuiBlockPrivate();

 protected:
  QPen _halo_pen;
  QBrush _halo_brush;
  QRectF _halo_rect;
  friend class GuiBlock;
};

#endif  // GUIBLOCK_H
