#include <QPixmap>

#include "gui_io.h"
#include "mainwindow.h"

void MainWindow::createGuiCTS() {
  createMenuCTS();
  createToolbarCTS();
}

void MainWindow::createMenuCTS() {
  QMenu* ctsMenu = menuBar()->addMenu(tr("CTS"));
  ctsMenu->addAction(tr("Create CTS Nets"), this, &MainWindow::CtsCreateNets);
}

void MainWindow::createToolbarCTS() { }

void MainWindow::CtsCreateNets() { guiInst->autoRunCTS(); }
