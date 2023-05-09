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
 * @file GuiInstance.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief This class inherits from QGraphicsItemGroup to represent Net in the GUI
 *        Multiple guilines will be added to this group to represent the actual Routing between pins.
 *         ___________
 *        |       ▪--\|----┐
 *        |____▪______|    |
 *          ___|           |
 *  _______|___            | ← this is wire,the actual Routing.
 * |       ▪   |         __|
 * |\__________| _______|___
 *              |/      ▪   |
 *              |___________|
 * @version 0.1
 * @date 2021-08-17(V0.1)
 *
 *
 *
 */
#ifndef GUIWIRE_H
#define GUIWIRE_H
#include <QGraphicsItemGroup>
#include "guiline.h"
class GuiWire : public QGraphicsItemGroup
{
public:
    GuiWire(QVector<QLineF> wires, QGraphicsItem *parent = nullptr);
private:
    QVector<QLineF> _wires;
};
#endif // GUIWIRE_H
