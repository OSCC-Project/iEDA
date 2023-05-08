/**
 * @file guirow.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief This class inherits from GuiInstance to represent StandardCell in the GUI
 * @version 0.1
 * @date 2021-07-06(V0.1), 2021-08-11(V0.2)
 */

#ifndef GUIROW_H
#define GUIROW_H

#include "guirect.h"

class GuiRowPrivate;

class GuiRow : public GuiRect {
 public:
  explicit GuiRow(QGraphicsItem *parent = nullptr);
  GuiRow(GuiRowPrivate *data, QGraphicsItem *parent = nullptr);
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;

 protected:
  inline bool isAllowToShow() const override;
  inline bool isAllowToSelect() const override;

 private:
  GuiRowPrivate *_data;
};

class GuiRowPrivate : public GuiRectPrivate {
 public:
  explicit GuiRowPrivate() { }
  ~GuiRowPrivate() = default;
};

#endif  // GUIROW_H
