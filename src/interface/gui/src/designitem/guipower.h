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
 * @file guipower.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief This class inherits from GuiRect to represent power in the GUI
 * @version 0.2
 * @date 2021-07-17(V0.1), 2021-08-18(V0.2)
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
