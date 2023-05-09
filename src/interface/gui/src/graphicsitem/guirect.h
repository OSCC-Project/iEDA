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
 * @file guirect.h
 * @author Wang Jun (wen8365@gmail.com)
 *         Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief Gui rectangle item base class
 * @version 0.2
 * @date 2021-07-02(V0.1) 2021-08-05(V0.2)
 *
 *
 *
 */
#ifndef GUIRECT_H
#define GUIRECT_H
#include "guiitem.h"

using namespace idb;

class GuiRectPrivate;
class GuiRect : public GuiItem {
 public:
  explicit GuiRect(QGraphicsItem* parent = nullptr);
  ~GuiRect();

  QRectF get_rect();
  void set_rect(const QRectF& rect, IdbOrient orientation = IdbOrient::kNone);
  inline void set_rect(qreal x1, qreal y1, qreal x2, qreal y2, IdbOrient orientation = IdbOrient::kNone);

  // IdbOrient get_orientation();
  //  void set_orientation(GuiItem::Orientation orientation);

  QRectF boundingRect() const override;
  QPainterPath shape() const override;

  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

 protected:
  GuiRect(GuiRectPrivate* data, QGraphicsItem* parent);

 private:
  void setPointArea(const QPointF& point);
  GuiRectPrivate* _data;
  void DrawOrientationLine();
};
inline void GuiRect::set_rect(qreal x1, qreal y1, qreal x2, qreal y2, IdbOrient orientation) {
  set_rect(QRectF(QPointF(x1, y1), QPointF(x2, y2)), orientation);
}
class GuiRectPrivate : public GuiItemPrivate {
 public:
  explicit GuiRectPrivate();
  ~GuiRectPrivate();

  /// getter
  QRectF& get_rect() { return _rect; }
  bool is_info_exist() { return _info_list.size() > 0 ? true : false; }
  std::vector<std::string>& get_info_list() { return _info_list; }
  std::string& get_item_info() { return _item_info; }

  /// setter
  void add_info(std::string info) { _info_list.emplace_back(info); }
  void set_item_info(std::string info) { _item_info = info; }

 protected:
  QPointF _pos;
  QRectF _rect;
  IdbOrient _orientation;
  QLineF _line;
  QLineF _arrow[2];
  std::string _item_info;
  std::vector<std::string> _info_list;
  friend class GuiRect;
};

#endif  // GUIRECT_H
