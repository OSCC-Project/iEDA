#include <QPixmap>

#include "gui_io.h"
#include "mainwindow.h"

void MainWindow::createGuiPdn() {
  createMenuPdn();
  createToolbarPdn();
}

void MainWindow::createMenuPdn() {
  QMenu* pdnMenu = menuBar()->addMenu(tr("PDN"));
  pdnMenu->addAction(tr("Create Power Nets"), this, &MainWindow::PdnCreateNet);
  pdnMenu->addAction(tr("Connect IO"), this, &MainWindow::PdnConnectIO);

  pdnMenu->addSeparator();

  pdnMenu->addAction(tr("Place Ports"), this, &MainWindow::PdnPlacePorts);

  pdnMenu->addSeparator();

  QMenu* addStripeMenu = pdnMenu->addMenu(tr("Add Stripe"));
  addStripeMenu->addAction(tr("Add Stripe to Standard Cell"), this, &MainWindow::PdnAddStdCellStripe);
  addStripeMenu->addAction(tr("Add Stripe Grid"), this, &MainWindow::PdnAddStripeGrid);

  pdnMenu->addSeparator();

  pdnMenu->addAction(tr("Connect Power Nets"), this, &MainWindow::PdnConnectAllLayers);

}

  void MainWindow::createToolbarPdn(){}

  void MainWindow::PdnCreateNet(){}

  void MainWindow::PdnConnectIO(){}

  void MainWindow::PdnPlacePorts(){}

  void MainWindow::PdnAddStdCellStripe(){}

  void MainWindow::PdnAddStripeGrid(){}

  void MainWindow::PdnConnectAllLayers(){}

