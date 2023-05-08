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
 * @copyright Copyright (c) 2021
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
