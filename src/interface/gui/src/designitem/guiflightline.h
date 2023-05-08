/**
 * @file GuiInstance.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief This class inherits from QGraphicsItemGroup to represent FlightLine in the GUI
 *        Multiple guilines will be added to this group to represent the connection between pins.
 *         ___________
 *        |       ▪  \|
 *        |____▪___\__|
 *            /     \
 *  _________/_      \ ← this is flightline, just indicates the connection relation,
 * |        ▪  |      \  but does not indicate the actual Routing.
 * |\__________| ______\____
 *              |/      ▪   |
 *              |___________|
 *
 * @version 0.1
 * @date 2021-08-17(V0.1)
 *
 */
#ifndef GUIFLIGHTLINE_H
#define GUIFLIGHTLINE_H
#include <QGraphicsItemGroup>

#include "guiline.h"
class GuiFlightLine : public QGraphicsItemGroup {
 public:
  explicit GuiFlightLine(QVector<QPointF> pins, QGraphicsItem *parent = nullptr);

 private:
  /**
   * @brief The first one in the vector is the pin of this stdcell,
   *        and the others are the pins of other stdcell.We should
   *        connect the pin of this stdcell to the pins of other
   *        stdcell.
   */
  QVector<QPointF> _pins;
};
#endif  // GUIFLIGHTLINE_H
