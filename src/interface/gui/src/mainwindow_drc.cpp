#include <QPixmap>

#include "gui_io.h"
#include "mainwindow.h"

void MainWindow::createGuiDRC() {
  createMenuDRC();
  createToolbarDRC();
}

void MainWindow::createMenuDRC() {
  QMenu* drcMenu = menuBar()->addMenu(tr("DRC"));
  drcMenu->addAction(tr("Check DEF"), this, &MainWindow::DrcCheckDef);
}

void MainWindow::createToolbarDRC() { }

void MainWindow::DrcCheckDef() { guiInst->autoRunDrcCheckDef(); }
