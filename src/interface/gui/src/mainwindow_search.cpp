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
#include <QPixmap>


#include "mainwindow.h"

void MainWindow::createSearch() {
  // create searchbox
  QToolBar* searchBox = new QToolBar(tr("Search Box"));
  QWidget* spacer     = new QWidget(this);

  spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  spacer->setFixedSize(1000, 20);
  searchBox->addWidget(spacer);

  QPushButton* searchBtn = new QPushButton(this);

  searchBtn->setCursor(Qt::PointingHandCursor);
  searchBtn->setFixedSize(20, 20);
  searchBtn->setToolTip("Search");
  searchBtn->setStyleSheet(
      "QPushButton{border-image:url(./iEDA/src/iGUI/res/icon/search.png); background:transparent;} \
                                     QPushButton:hover{border-image:url(./iEDA/src/iGUI/res/icon/search.png)} \
                                     QPushButton:pressed{border-image:url(./iEDA/src/iGUI/res/icon/search.png)}");
  connect(searchBtn, SIGNAL(clicked(bool)), this, SLOT(search()));

  _edit_search = new GuiSearchEdit(searchBox);

  QMargins margins = _edit_search->textMargins();
  _edit_search->setTextMargins(margins.left(), margins.top(), searchBtn->width(), margins.bottom());

  QHBoxLayout* searchLayout = new QHBoxLayout();

  searchLayout->addStretch();
  searchLayout->addWidget(searchBtn);
  searchLayout->setSpacing(0);
  searchLayout->setContentsMargins(0, 0, 3, 0);

  _edit_search->setTextMargins(3, 0, 3 + 8, 0);
  _edit_search->setFont(QFont("Microsoft YaHei", 10));
  _edit_search->setLayout(searchLayout);
  connect(_edit_search, &GuiSearchEdit::search, this, &MainWindow::search);

  searchBox->addWidget(_edit_search);
  layoutTab->addToolBar(searchBox);
}

void MainWindow::search() {
  QString search_txt = _edit_search->text();
  _scene->search(search_txt);
}
