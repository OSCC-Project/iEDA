#include <QPixmap>

#include "gui_io.h"
#include "mainwindow.h"

void MainWindow::createGuiRouting() {
  createMenuRouting();
  createToolbarRouting();
}

void MainWindow::createMenuRouting() {
  QMenu* routingMenu = menuBar()->addMenu(tr("Routing"));
  routingMenu->addAction(tr("Routing Nets"), this, &MainWindow::RoutingNets);
}

void MainWindow::createToolbarRouting() { }

void MainWindow::RoutingNets() { guiInst->autoRunRouter(); }
