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
