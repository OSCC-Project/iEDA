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


void MainWindow::createToolbar() {
  // create editTool
  QToolBar* editTool = new QToolBar(tr("Edit"));

  editTool->addActions(editActions);

  layoutTab->addToolBar(editTool);

  // create viewTool
  QToolBar* viewTool = new QToolBar(tr("View"));

  viewTool->addActions(viewActions);

  layoutTab->addToolBar(viewTool);

  // create ToolsBar
  QToolBar* toolsBar = new QToolBar(tr("Tools"));

  toolsBar->addActions(toolsActions);

  layoutTab->addToolBar(toolsBar);

  // create reportBar
  QToolBar* reportBar = new QToolBar(tr("Report"));

  reportBar->addActions(reportActions);

  layoutTab->addToolBar(reportBar);
  layoutTab->addToolBarBreak();

  // create toolbox
  QToolBar* toolbox = new QToolBar(tr("Tool Box"));

  QAction* selectByBox = toolbox->addAction(QIcon("./iEDA/src/iGUI/res/icon/selectByBox.png"),
                                            tr("Select By Box"), this, &MainWindow::selectByBox);

  toolbox->addActions(toolBoxActions->actions());

  toolBoxActions->addAction(selectByBox);

  layoutTab->addToolBar(toolbox);

  // create wireEditTool
  QToolBar* wireEditTool = new QToolBar(tr("wire Edit"));

  wireEditTool->addAction(QIcon("./iEDA/src/iGUI/res/icon/duplicate.png"),
                          tr("Duplicate Selected Wires"), this, &MainWindow::duplicateSelectedWires);
  wireEditTool->addAction(QIcon("./iEDA/src/iGUI/res/icon/split.png"), tr("Split Selected Wires"),
                          this, &MainWindow::splitSelectedWires);
  wireEditTool->addAction(QIcon("./iEDA/src/iGUI/res/icon/merge.png"), tr("Merge Selected Wires"),
                          this, &MainWindow::mergeSelectedWires);
  wireEditTool->addAction(QIcon("./iEDA/src/iGUI/res/icon/trim.png"), tr("Trim Selected Wires"),
                          this, &MainWindow::trimSelectedWires);
  wireEditTool->addAction(QIcon("./iEDA/src/iGUI/res/icon/deleteWire.png"), tr("Delete Wires"),
                          this, &MainWindow::deleteWires);

  layoutTab->addToolBar(Qt::LeftToolBarArea, wireEditTool);

  // creat PanActionTool
  QToolBar* pan = new QToolBar(tr("Pan Action"));

  pan->addAction(QIcon("./iEDA/src/iGUI/res/icon/up.png"), tr("Up"), this, &MainWindow::upAct);
  QAction* downAct  = pan->addAction(QIcon("./iEDA/src/iGUI/res/icon/down.png"), tr("Down"), this,
                                     &MainWindow::downAct);
  QAction* leftAct  = pan->addAction(QIcon("./iEDA/src/iGUI/res/icon/left.png"), tr("Left"), this,
                                     &MainWindow::leftAct);
  QAction* rightAct = pan->addAction(QIcon("./iEDA/src/iGUI/res/icon/right.png"), tr("Right"),
                                     this, &MainWindow::rightAct);

  layoutTab->addToolBar(Qt::LeftToolBarArea, pan);
}
