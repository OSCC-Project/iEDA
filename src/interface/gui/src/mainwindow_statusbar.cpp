

#include <QPixmap>

#include "mainwindow.h"

void MainWindow::createStatusbar() {
  statusBar()->showMessage(tr("Ready"));
  coordinate = new QLabel(tr("Current Position"), this);
  statusBar()->addPermanentWidget(coordinate);
}
