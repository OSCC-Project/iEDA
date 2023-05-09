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

#include "gui_io.h"
#include "mainwindow.h"

void MainWindow::createGuiFloorplan() {
  createMenuFloorplan();
  createToolbarFloorplan();
}

void MainWindow::createMenuFloorplan() {
  QMenu* floorplanMenu = menuBar()->addMenu(tr("Floorpl&an"));
  floorplanMenu->addAction(tr("Init Floorplan"), this, &MainWindow::InitFloorplan);

  floorplanMenu->addSeparator();

  QMenu* placeInstMenu = floorplanMenu->addMenu(tr("Place Instance"));
  placeInstMenu->addAction(tr("Place IO Pad"), this, &MainWindow::FpPlaceIOPad);
  placeInstMenu->addAction(tr("Place Macro"), this, &MainWindow::FpPlaceMacro);
  placeInstMenu->addAction(tr("Place Core Instance"), this, &MainWindow::FpPlaceInstance);
  placeInstMenu->addAction(tr("Place All Instances"), this, &MainWindow::FpPlaceAllInstance);

  floorplanMenu->addSeparator();

  QMenu* placeIOMenu = floorplanMenu->addMenu(tr("Place IO"));
  placeIOMenu->addAction(tr("Place Port"), this, &MainWindow::FpPlaceIOPort);
  placeIOMenu->addAction(tr("Place All Ports"), this, &MainWindow::FpPlaceAllPorts);
  placeIOMenu->addSeparator();
  placeIOMenu->addAction(tr("Place IO Filler"), this, &MainWindow::FpPlaceIOFiller);

  floorplanMenu->addAction(tr("Tap Cell"), this, &MainWindow::FpTapCell);

  floorplanMenu->addSeparator();

  //   QMenu* automaticFloorplan = floorplanMenu->addMenu(tr("&Automatic Floorplan"));
  //   floorplanMenu->addAction(tr("Resi&ze Floorplan"), this, &MainWindow::resizeFloorplan);
  //   QMenu* relativeFloorplan = floorplanMenu->addMenu(tr("&Relative Floorplan"));

  //   QMenu* row = floorplanMenu->addMenu(tr("Ro&w"));
  //   toolsActions <<
  //   floorplanMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/floorplanToolBox.png"),
  //                                            tr("&Floorplan Toolbox"), this, &MainWindow::floorplanToolbox);
  //   floorplanMenu->addAction(tr("Trace &Macro"), this, &MainWindow::traceMacro);
  //   floorplanMenu->addAction(tr("Macro &Timing Slack Display"), this, &MainWindow::macroTimingSlackDisplay);
  //   QMenu* editFloorplan = floorplanMenu->addMenu(tr("&Edit Floorplan"));
  //   editFloorplan->addAction(QIcon(), tr("&Cut Rectilinear"), this, &MainWindow::cutRectilinear);
  //   editFloorplan->addSeparator();
  //   toolBoxActions->addAction(
  //       editFloorplan->addAction(QIcon("./iEDA/src/iGUI/res/icon/sizeBlock.png"),
  //                                tr("Cre&ate Size Blockage"), this, &MainWindow::createSizeBlockage));
  //   toolBoxActions->addAction(
  //       editFloorplan->addAction(QIcon("./iEDA/src/iGUI/res/icon/placementBlock.png"),
  //                                tr("Crea&te Placement Blockage"), this, &MainWindow::createPlacementBlockage));
  //   toolBoxActions->addAction(
  //       editFloorplan->addAction(QIcon("./iEDA/src/iGUI/res/icon/routeBlock.png"),
  //                                tr("Create &Routing Blockage"), this, &MainWindow::createRoutingBlockage));
  //   editFloorplan->addAction(QIcon("./iEDA/src/iGUI/res/icon/pinBlock.png"),
  //                            tr("Create P&in Blockage"), this, &MainWindow::createPinBlockage);

  //   floorplanMenu->addAction(tr("Sna&p Floorplan"), this, &MainWindow::snapFloorplan, tr("Ctrl+N"));
  //   floorplanMenu->addAction(tr("&Check Floorplan"), this, &MainWindow::checkFloorplan);
  //   floorplanMenu->addAction(tr("C&lear Floorplan"), this, &MainWindow::clearFloorplan);

  //   floorplanMenu->addSeparator();

  //   QMenu* instanceGroup = floorplanMenu->addMenu(tr("Instance Group"));
  //   floorplanMenu->addAction(tr("Ge&nerate Regrouped Netlist"), this, &MainWindow::generateRegroupedNetlist);

  //   floorplanMenu->addSeparator();

  //   floorplanMenu->addAction(tr("&Generate Floorplan"), this, &MainWindow::generateFloorplan);
}

void MainWindow::createToolbarFloorplan() { }

void MainWindow::InitFloorplan() { }

void MainWindow::FpPlaceIOPad() { }
void MainWindow::FpPlaceMacro() { }
void MainWindow::FpPlaceInstance() { }
void MainWindow::FpPlaceAllInstance() { }

void MainWindow::FpPlaceIOPort() { }
void MainWindow::FpPlaceAllPorts() { }
void MainWindow::FpPlaceIOFiller() { }

void MainWindow::FpTapCell() { std::cout << "Tap cell" <<std::endl; }

void MainWindow::resizeFloorplan() { }
void MainWindow::floorplanToolbox() { }
void MainWindow::traceMacro() { }
void MainWindow::macroTimingSlackDisplay() { }
void MainWindow::snapFloorplan() { }
void MainWindow::checkFloorplan() { }
void MainWindow::clearFloorplan() { }
void MainWindow::generateRegroupedNetlist() { }
void MainWindow::generateFloorplan() { }
