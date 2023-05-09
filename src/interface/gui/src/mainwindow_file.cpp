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


void MainWindow::createGuiFile() {
  createMenuFile();
  createToolbarFile();
}
void MainWindow::createMenuFile() {
  /*--------------------create flieMenu---------------------*/
  QMenu* fileMenu = menuBar()->addMenu(tr("&File"));
  FileActions << fileMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/Import.png"),
                                     tr("&Import Design"), this, &MainWindow::import);
  //   FileActions << fileMenu->addAction(QIcon(""), tr("&Import Design"), this, &MainWindow::import);
  fileMenu->addSeparator();

  //   fileMenu->addAction(tr("R&estore Design"), this, &MainWindow::restore);
  //   fileMenu->addAction(tr("EC&O Design"), this, &MainWindow::eco);
  //   FileActions << fileMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/save.png"),
  //                                      tr("Sa&ve Design"), this, &MainWindow::save, tr("F2"));

  //   fileMenu->addSeparator();

  //   fileMenu->addAction(tr("&Create OA Library"), this, &MainWindow::createOA);

  //   fileMenu->addSeparator();

  //   QMenu* load = fileMenu->addMenu(tr("&Load"));
  //   load->addAction(tr("P&artition"), this, &MainWindow::loadPartition);
  //   load->addAction(tr("&Floorplan"), this, &MainWindow::loadFloorplan);
  //   load->addAction(tr("&I/O File"), this, &MainWindow::loadIOFile);
  //   load->addAction(tr("&Place"), this, &MainWindow::loadPlace);
  //   load->addSeparator();
  //   load->addAction(tr("&DEF"), this, &MainWindow::loadDEF);
  //   load->addAction(tr("PDE&F"), this, &MainWindow::loadPDEF);
  //   load->addAction(tr("S&PFE"), this, &MainWindow::loadSPFE);
  //   load->addAction(tr("&SDF"), this, &MainWindow::loadSDF);
  //   load->addSeparator();
  //   load->addAction(tr("&OA Cellview"), this, &MainWindow::loadOACellview);
  //   QMenu* save = fileMenu->addMenu(tr("&Save"));
  //   save->addAction(tr("P&artition"), this, &MainWindow::savePartition);
  //   save->addAction(tr("&Floorplan"), this, &MainWindow::saveFloorplan);
  //   save->addAction(tr("&I/O File"), this, &MainWindow::saveIOFile);
  //   save->addAction(tr("&Place"), this, &MainWindow::savePlace);
  //   save->addAction(tr("&Netlist"), this, &MainWindow::saveNetlist);
  //   save->addAction(tr("Test&case"), this, &MainWindow::saveTestcase);
  //   save->addSeparator();
  //   save->addAction(tr("&DEF"), this, &MainWindow::saveDEF);
  //   save->addAction(tr("PDE&F"), this, &MainWindow::savePDEF);
  //   save->addAction(tr("Timing &Budget"), this, &MainWindow::saveTimingBudget);
  //   save->addSeparator();
  //   save->addAction(tr("&GDS/OASIS"), this, &MainWindow::saveGDS_OASIS);
  //   save->addAction(tr("&OA Cellview"), this, &MainWindow::saveOACellview);
  //   fileMenu->addSeparator();
  //   fileMenu->addAction(tr("C&heck Design"), this, &MainWindow::check);
  //   QMenu* report = fileMenu->addMenu(tr("&Report"));
  //   reportActions << report->addAction(QIcon("./iEDA/src/iGUI/res/icon/summary.png"),
  //   tr("&Summary"),
  //                                      this, &MainWindow::summary, tr("Shift+Q"));
  //   reportActions << report->addAction(QIcon("./iEDA/src/iGUI/res/icon/selectobj.png"),
  //                                      tr("S&elected Object"), this, &MainWindow::selectedObject);
  //   report->addSeparator();
  //   report->addAction(tr("&Gate Count"), this, &MainWindow::gateCount);
  //   report->addAction(tr("&Netlist Statistics"), this, &MainWindow::netlistStatistics);
  //   fileMenu->addSeparator();

  //   QMenu* recent =
  //       fileMenu->addMenu(QIcon("./iEDA/src/iGUI/res/icon/recent.png"), tr("Recen&t Action"));

  //   fileMenu->addSeparator();

  fileMenu->addAction(tr("E&xit"), this, &QMainWindow::close);
}

void MainWindow::createToolbarFile() {
  // create fileTool
  QToolBar* fileTool = new QToolBar(tr("File"));

  fileTool->addActions(FileActions);

  layoutTab->addToolBar(fileTool);
}

// slot function
void MainWindow::import() {
  if (!_importFile)
    _importFile = new FileImport(this);

  if (QDialog::Rejected == _importFile->showDialog()) {
    return;
  }

  std::map<std::string, std::list<std::string>> fileMap = _importFile->getFilePath();
  if (fileMap.empty())
    return;
  if (!_scene->isEmpty())
    _scene->clear();
  QCoreApplication::processEvents();
  QApplication::setOverrideCursor(Qt::WaitCursor);
  // TODO: Use Loading window in here to wait createChip finish
  _scene->createChip(fileMap);

  QApplication::restoreOverrideCursor();
  //  QCoreApplication::processEvents();
  //  QPixmap pixmap;
  //  QPainter painter(&pixmap);
  //  _scene->render(&painter);
  //  _scene->clear();
  //  QCoreApplication::processEvents();
  //  _scene->addPixmap(pixmap);
}

void MainWindow::restore() {
  //  QImage image(_scene->width(), _scene->height(), QImage::Format_RGB32);
  //  QPainter painter(&image);
  //  _scene->render(&painter);  //关键函数
  QPixmap img = _graphicsView->grab();
  img.save("./test.png");
}
void MainWindow::eco() { }
void MainWindow::save() { }
void MainWindow::createOA() { }
void MainWindow::check() { }
void MainWindow::loadPartition() { }
void MainWindow::loadFloorplan() { }
void MainWindow::loadIOFile() { }
void MainWindow::loadPlace() { }
void MainWindow::loadDEF() { }
void MainWindow::loadPDEF() { }
void MainWindow::loadSPFE() { }
void MainWindow::loadSDF() { }
void MainWindow::loadOACellview() { }
void MainWindow::savePartition() { }
void MainWindow::saveFloorplan() { }
void MainWindow::saveIOFile() { }
void MainWindow::savePlace() { }
void MainWindow::saveNetlist() { }
void MainWindow::saveTestcase() { }
void MainWindow::saveDEF() { }
void MainWindow::savePDEF() { }
void MainWindow::saveTimingBudget() { }
void MainWindow::saveGDS_OASIS() { }
void MainWindow::saveOACellview() { }
void MainWindow::summary() { }
void MainWindow::selectedObject() { }
void MainWindow::gateCount() { }
void MainWindow::netlistStatistics() { }