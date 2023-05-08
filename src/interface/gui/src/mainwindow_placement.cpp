#include <QPixmap>

#include "gui_io.h"
#include "mainwindow.h"

void MainWindow::createGuiPlacement() {
  createMenuPlacement();
  createToolbarPlacement();
}

void MainWindow::createMenuPlacement() {
  QMenu* placeMenu = menuBar()->addMenu(tr("Placement"));
  placeMenu->addAction(tr("Auto Place"), this, &MainWindow::PlaceInstance);
}

void MainWindow::createToolbarPlacement() { }

void MainWindow::PlaceInstance() { guiInst->autoRunPlacer(); }
