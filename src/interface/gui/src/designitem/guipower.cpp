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
#include "guipower.h"

#include "guiattribute.h"

GuiPower::GuiPower(QGraphicsItem* parent)
    : GuiRect(new GuiPowerPrivate(), parent), _data(static_cast<GuiPowerPrivate*>(_d_ptr)) { }
void GuiPower::set_layer(const std::string& layer) {
  _data->_layer = layer;
  set_pen(attributeInst->getLayerColor(layer));
  set_brush(attributeInst->getLayerColor(layer), Qt::BrushStyle::FDiagPattern);
}

void GuiPower::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  if (isAllowToShow()) {
    //        const qreal lod =
    //        option->levelOfDetailFromTransform(painter->worldTransform());

    //        if(lod < 0.1){
    //            if(_data->_rect.width() > _data->_rect.height())
    //            {
    //                qreal y = (_data->_rect.bottom()+_data->_rect.top())/2;
    //                painter->drawLine(QPointF(_data->_rect.left(), y),
    //                QPointF(_data->_rect.right(), y));
    //            }
    //            else
    //            {
    //                qreal x = (_data->_rect.left()+_data->_rect.right())/2;
    //                painter->drawLine(QPointF(x, _data->_rect.top()),
    //                QPointF(x, _data->_rect.bottom()));
    //            }
    //            return;
    //        }

    GuiRect::paint(painter, option, widget);
    setFlag(QGraphicsItem::ItemIsSelectable, isAllowToSelect());
  } else {
    setFlag(QGraphicsItem::ItemIsSelectable, false);
  }
}

inline bool GuiPower::isAllowToShow() const { return true; }
inline bool GuiPower::isAllowToSelect() const { return true; }
