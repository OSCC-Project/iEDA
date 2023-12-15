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
#include "mainwindow.h"

#include <QPixmap>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) { initView(); }

MainWindow::~MainWindow() {
  if (_control_tree != nullptr) {
    delete _control_tree;
    _control_tree = nullptr;
  }

  if (_scene != nullptr) {
    delete _scene;
    _scene = nullptr;
  }

  if (_graphicsView != nullptr) {
    delete _graphicsView;
    _graphicsView = nullptr;
  }
}

void MainWindow::initView() {
  //   _splash = new GuiSplash(this, QPixmap("./iEDA/src/iGUI/res/icon/start.png"));
  //   _splash->show();

  setWindowTitle(tr("iEDA"));
  setWindowState(Qt::WindowMaximized);
  createTabWidget();
  ////
  createGuiFile();
  createGuiFloorplan();
  createGuiPdn();
  createGuiPlacement();
  createGuiCTS();
  createGuiRouting();
  createGuiDRC();
  ////
  //   createMenu();
  //   createToolbar();
  createSearch();

  createStatusbar();
  createControlView();
  createScene();

  resize(900, 800);
}

void onLoadError(QString error_msg) { qDebug() << "DB set up failed: " << error_msg; }

void MainWindow::createScene() {
  _scene        = new GuiGraphicsScene();
  _graphicsView = new GuiGraphicsView(this);
  _graphicsView->setScene(_scene);
  layoutTab->setCentralWidget(_graphicsView);
  connect(_graphicsView, &GuiGraphicsView::coordinateChange, this, &MainWindow::setCoordinate);
}

void MainWindow::createTabWidget() {
  tabWidget = new QTabWidget;
  tabWidget->setFont(QFont("Arial", 12, 75));
  layoutTab = new QMainWindow;
  tabWidget->addTab(layoutTab, tr("&Layout"));
  tabWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  setCentralWidget(tabWidget);
}

void MainWindow::createControlView() {
  _control_tree       = new GuiTree(this);
  QDockWidget* widget = new QDockWidget("Control", this);
  widget->setFeatures(QDockWidget::DockWidgetMovable);
  widget->setAllowedAreas(Qt::RightDockWidgetArea);
  widget->setMinimumWidth(400);
  widget->setMaximumWidth(400);
  widget->setWidget(_control_tree);
  layoutTab->addDockWidget(Qt::RightDockWidgetArea, widget);
}

void MainWindow::updateTree() {
  _control_tree->setDbSetup(_scene->get_db_setup());
  _control_tree->updateLayer();
}

void MainWindow::setCoordinate(const QString& position) { coordinate->setText(position); }

void MainWindow::fit() { }
void MainWindow::redraw() { }
void MainWindow::setPreference() { }
void MainWindow::allColor() { }
void MainWindow::setLineColor() { }
void MainWindow::goTo() { }
void MainWindow::findSelect() { }
void MainWindow::deselectAll() { }
void MainWindow::original() { }
void MainWindow::clearoriginal() { }
void MainWindow::editHighlightColor() { }
void MainWindow::dimBackground() { }
void MainWindow::undo() { }
void MainWindow::redo() { }
void MainWindow::copy() { }
void MainWindow::attributeEditor() { }
void MainWindow::dbBrowser() { }
void MainWindow::moveResizeReshape() { }
void MainWindow::editPinGroup() { }
void MainWindow::editNetGroup() { }
void MainWindow::editPinGuide() { }
void MainWindow::pinEditor() { }
void MainWindow::ceateNonDefaultRule() { }
void MainWindow::editWire() { }
void MainWindow::moveWire() { }
void MainWindow::cutWire() { }
void MainWindow::snapWire() { }
void MainWindow::stretchWire() { }
void MainWindow::addViaWire() { }
void MainWindow::addPolygonWire() { }
void MainWindow::editBusGuide() { }
void MainWindow::busColor() { }
void MainWindow::clearBusColor() { }

void MainWindow::specifyPartition() { }
void MainWindow::specifyBlackBox() { }
void MainWindow::clonePlace() { }
void MainWindow::showWireCrossing() { }
void MainWindow::createPhysicalFeedthrough() { }
void MainWindow::feedthroughPorts() { }
void MainWindow::assignPin() { }
void MainWindow::checkPinAssignment() { }
void MainWindow::driveTimingBudget() { }
void MainWindow::commitPartition() { }
void MainWindow::flattenPartition() { }
void MainWindow::assembleDesign() { }
void MainWindow::changePartitionView() { }

void MainWindow::connectGlobalNet() { }
void MainWindow::powerGridLibrary() { }
void MainWindow::power() { }
void MainWindow::powerHistogram() { }
void MainWindow::powerPailResult() { }
void MainWindow::dynamicMovie() { }
void MainWindow::dynamicWaveform() { }
void MainWindow::placeItag() { }
void MainWindow::placeStandardCell() { }
void MainWindow::PlaceSpareCell() { }
void MainWindow::refinePlacement() { }
void MainWindow::ecoPlacement() { }
void MainWindow::checkPlacement() { }
void MainWindow::spareCell() { }
void MainWindow::clearSpareCellDisplay() { }
void MainWindow::scanChain() { }
void MainWindow::densityMap() { }
void MainWindow::pinDensityMap() { }
void MainWindow::displayEdgeComstraints() { }
void MainWindow::displayCellPadding() { }
void MainWindow::displayCellStackGroup() { }
void MainWindow::displayImplantGroup() { }
void MainWindow::optimizeDesign() { }
void MainWindow::interativeECO() { }
void MainWindow::ccOptClockTreeDebugger() { }
void MainWindow::generateRouteGuide() { }
void MainWindow::earlyGlobalRoute() { }
void MainWindow::specialRoute() { }
void MainWindow::mmmcBrower() { }
void MainWindow::generateCapacitanceTable() { }
void MainWindow::extractRC() { }
void MainWindow::reportTiming() { }
void MainWindow::debugTiming() { }
void MainWindow::createBlackBox() { }
void MainWindow::writeSDF() { }
void MainWindow::displayTimingMap() { }
void MainWindow::displayNoiseNet() { }
void MainWindow::verifyDRC() { }
void MainWindow::verifyGeometry() { }
void MainWindow::verifyConnectivity() { }
void MainWindow::verifyProcessAntenna() { }
void MainWindow::verifyACLimit() { }
void MainWindow::verifyEndCap() { }
void MainWindow::verifyMentalDensity() { }
void MainWindow::verifyCutDensity() { }
void MainWindow::verifyPowerVia() { }
void MainWindow::designBrower() { }
void MainWindow::setGlobalVariable() { }
void MainWindow::violationBrowser() { }
void MainWindow::layoutViewer() { }
void MainWindow::cellViewer() { }
void MainWindow::schematicViewer() { }
void MainWindow::logViewer() { }
void MainWindow::flightlineBrowser() { }
void MainWindow::integrationConstraintEditor() { }
void MainWindow::runVSR() { }
void MainWindow::pullBlockConstraint() { }
void MainWindow::setMultipleCpuUsage() { }
void MainWindow::flipChip() { }
void MainWindow::tsv() { }
void MainWindow::VerifyLitho() { }
void MainWindow::checkLithoStatus() { }
void MainWindow::verifyCMP() { }
void MainWindow::checkCMPStatus() { }
void MainWindow::createAct() { }
void MainWindow::viewAct() { }
void MainWindow::writeToGifFile() { }
void MainWindow::screenDump() { }
void MainWindow::displayScreenDump() { }
void MainWindow::createRuler() {
  // _scene->setCurrentShape(Shape::Ruler);
}
void MainWindow::clearAllRulers() {
  // qDebug() << _scene->rulerList.size();
  //   QList<Ruler*> list = _scene->get_ruler_list();
  //   for (int i = 0; i < list.size(); i++) {
  //     _scene->removeItem(list[i]);
  //   }
  update();
}
void MainWindow::clearAllHighlight() { }
void MainWindow::clearSelectedHighlight() { }
void MainWindow::inAct() { }
void MainWindow::outAct() { }
void MainWindow::Selected() { }
void MainWindow::previous() { }
void MainWindow::nextAct() { }
void MainWindow::upAct() {
  //   qDebug() << "up press!";
  //   if (_scene->get_selected_item() != nullptr) {
  //     QGraphicsItem* item  = _scene->get_selected_item();
  //     QPointF _selectedPos = item->scenePos();
  //     item->setPos(_selectedPos.x(), _selectedPos.y() - value);
  //     qDebug() << item->scenePos();
  //   }
}
void MainWindow::downAct() {
  //   qDebug() << "down press!";
  //   if (_scene->get_selected_item() != nullptr) {
  //     QGraphicsItem* item  = _scene->get_selected_item();
  //     QPointF _selectedPos = item->scenePos();
  //     item->setPos(_selectedPos.x(), _selectedPos.y() + value);
  //     qDebug() << item->scenePos();
  //   }
}
void MainWindow::leftAct() {
  //   qDebug() << "left press!";
  //   if (_scene->get_selected_item() != nullptr) {
  //     QGraphicsItem* item  = _scene->get_selected_item();
  //     QPointF _selectedPos = item->scenePos();
  //     item->setPos(_selectedPos.x() - value, _selectedPos.y());
  //     qDebug() << item->scenePos();
  //   }
}
void MainWindow::rightAct() {
  //   qDebug() << "right press!";
  //   if (_scene->get_selected_item() != nullptr) {
  //     QGraphicsItem* item  = _scene->get_selected_item();
  //     QPointF _selectedPos = item->scenePos();
  //     item->setPos(_selectedPos.x() + value, _selectedPos.y());
  //     qDebug() << item->scenePos();
  //   }
}
void MainWindow::cutRectilinear() { }
void MainWindow::createSizeBlockage() { }
void MainWindow::createPlacementBlockage() { }
void MainWindow::createRoutingBlockage() {
  // _scene->setCurrentShape(Shape::Line);
}
void MainWindow::createPinBlockage() { }
void MainWindow::selectFlightline() { }
void MainWindow::editWireTool() { }

void MainWindow::duplicateSelectedWires() { }
void MainWindow::splitSelectedWires() { }
void MainWindow::mergeSelectedWires() { }
void MainWindow::trimSelectedWires() { }
void MainWindow::deleteWires() { }
void MainWindow::selectByBox() { }
