/**
 * @file guipower.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief This class inherits from GuiRect to represent power in the GUI
 * @version 0.2
 * @date 2021-07-17(V0.1), 2021-08-18(V0.2)
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef GUIPOWER_H
#define GUIPOWER_H

#include "guirect.h"

class GuiPowerPrivate : public GuiRectPrivate {
 public:
  explicit GuiPowerPrivate() { }
  ~GuiPowerPrivate() = default;

  void set_layer(std::string layer) { _layer = layer; }

 private:
  std::string _layer;

  friend class GuiPower;
};

class GuiPower : public GuiRect {
 public:
  explicit GuiPower(QGraphicsItem* parent = nullptr);

  void set_layer(const std::string& layer);
  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

 protected:
  inline bool isAllowToShow() const override;
  inline bool isAllowToSelect() const override;

 private:
  GuiPowerPrivate* _data;
};

#endif  // GUIPOWER_H
