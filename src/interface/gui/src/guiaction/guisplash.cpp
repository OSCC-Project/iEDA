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
#include "guisplash.h"

GuiSplash::GuiSplash(QWidget *parent, const QPixmap &pixmap, Qt::WindowFlags f)
    : QSplashScreen(parent, pixmap, f) {
  load = new QLabel(tr("Initialize...."), this);
  load->move(0, height() - load->height());
  connect(this, &GuiSplash::loading, this, &GuiSplash::loadingSlot);
}
void GuiSplash::loadingSlot(const QString &text) { setlabel(text); }
void GuiSplash::setlabel(const QString &text) {
  load->setText(QString(tr("Initialize....         %1")).arg(text));
  load->adjustSize();
}
