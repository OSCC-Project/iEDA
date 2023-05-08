/**
 * @file guistandardcell.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief This class inherits from GuiInstance to represent StandardCell in the GUI
 * @version 0.2
 * @date 2021-07-06(V0.1), 2021-08-06(V0.2)
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef GUISTANDARDCELL_H
#define GUISTANDARDCELL_H

#include "guiinstance.h"

class GuiStandardCellPrivate;

class GuiStandardCell : public GuiInstance {
 public:
  explicit GuiStandardCell(QGraphicsItem *parent = nullptr);
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;

 protected:
  inline bool isAllowToShow() const override;
  inline bool isAllowToSelect() const override;

 private:
  GuiStandardCellPrivate *_data;
};

class GuiStandardCellPrivate : public GuiInstancePrivate {
 public:
  explicit GuiStandardCellPrivate() { }
  ~GuiStandardCellPrivate() = default;
};

#endif  // GUISTANDARDCELL_H
