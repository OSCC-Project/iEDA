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


void MainWindow::createMenu() {
  return;
  /*--------------------create viewMenu---------------------*/
  QMenu* viewMenu = menuBar()->addMenu(tr("&View"));

  QMenu* zoom = viewMenu->addMenu(tr("&Zoom"));
  viewActions << zoom->addAction(QIcon("./iEDA/src/iGUI/res/icon/ZoomIn.png"), tr("&In"), this,
                                 &MainWindow::inAct, tr("Z"));
  viewActions << zoom->addAction(QIcon("./iEDA/src/iGUI/res/icon/ZoomOut.png"), tr("&Out"), this,
                                 &MainWindow::outAct, tr("Shift+Z"));
  viewActions.insert(0, zoom->addAction(QIcon("./iEDA/src/iGUI/res/icon/selected.png"),
                                        tr("&Selected"), this, &MainWindow::Selected));
  viewActions << zoom->addAction(QIcon("./iEDA/src/iGUI/res/icon/previous.png"), tr("&Previous"),
                                 this, &MainWindow::previous, tr("W"));
  viewActions << zoom->addAction(QIcon("./iEDA/src/iGUI/res/icon/next.png"), tr("&Next"), this,
                                 &MainWindow::nextAct, tr("Y"));

  QMenu* pan        = viewMenu->addMenu(tr("&Pan"));
  QAction* upAct    = pan->addAction(QIcon("./iEDA/src/iGUI/res/icon/up.png"), tr("&In"), this,
                                     &MainWindow::upAct, tr("Up"));
  QAction* downAct  = pan->addAction(QIcon("./iEDA/src/iGUI/res/icon/down.png"), tr("&Down"), this,
                                     &MainWindow::downAct, tr("Down"));
  QAction* leftAct  = pan->addAction(QIcon("./iEDA/src/iGUI/res/icon/left.png"), tr("&Left"), this,
                                     &MainWindow::leftAct, tr("Left"));
  QAction* rightAct = pan->addAction(QIcon("./iEDA/src/iGUI/res/icon/right.png"), tr("&Right"),
                                     this, &MainWindow::rightAct, tr("Right"));

  viewActions.insert(3, viewMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/fit.png"), tr("&Fit"),
                                            this, &MainWindow::fit, tr("F")));
  viewActions << viewMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/redraw.png"), tr("&Redraw"),
                                     this, &MainWindow::redraw, tr("Ctrl+R"));

  viewMenu->addSeparator();

  FileActions << viewMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/setPreference.png"),
                                     tr("Set &Preference"), this, &MainWindow::setPreference);
  viewMenu->addAction(tr("&All Color"), this, &MainWindow::allColor);
  viewMenu->addAction(tr("Set &FlightLine Congest Color"), this, &MainWindow::setLineColor);
  viewMenu->addAction(tr("Go_To"), this, &MainWindow::goTo);
  FileActions << viewMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/find.png"),
                                     tr("&Find/Select Object"), this, &MainWindow::findSelect, tr("Ctrl+F"));

  viewMenu->addSeparator();

  toolsActions << viewMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/deselect.png"),
                                      tr("&Deselect All"), this, &MainWindow::deselectAll, tr("Ctrl+D"));
  QMenu* highlightSelected = viewMenu->addMenu(
      QIcon("./iEDA/src/iGUI/res/icon/highlightSelected.png"), tr("&Highlight Selected"));
  highlightSelected->addAction(tr("&Original"), this, &MainWindow::original);
  QMenu* clearHighlight = viewMenu->addMenu(QIcon("./iEDA/src/iGUI/res/icon/clearhighlight.png"),
                                            tr("&Clear Highlight"));
  clearHighlight->addAction(tr("Clear &Original"), this, &MainWindow::clearoriginal);
  editActions << clearHighlight->addAction(QIcon("./iEDA/src/iGUI/res/icon/clearall.png"),
                                           tr("Clear All Highlight"), this, &MainWindow::clearAllHighlight);
  editActions << clearHighlight->addAction(QIcon("./iEDA/src/iGUI/res/icon/clearSelected.png"),
                                           tr("Clear Selected Highlight"), this, &MainWindow::clearSelectedHighlight);
  viewMenu->addAction(tr("&Edit Highlight Color"), this, &MainWindow::editHighlightColor);

  viewMenu->addSeparator();

  viewMenu->addAction(tr("Di&m Background"), this, &MainWindow::dimBackground, tr("F12"));

  /*--------------------create editMenu---------------------*/
  toolBoxActions  = new QActionGroup(this);
  QMenu* editMenu = menuBar()->addMenu(tr("&Edit"));
  editActions.insert(0, editMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/undo.png"),
                                            tr("&Undo"), this, &MainWindow::undo, tr("U")));
  editActions.insert(1, editMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/redo.png"),
                                            tr("&Redo"), this, &MainWindow::redo, tr("Shifit+U")));
  toolBoxActions->addAction(editMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/copy.png"),
                                                tr("&Copy"), this, &MainWindow::copy));
  toolsActions << editMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/attribute.png"),
                                      tr("&Attribute Editor"), this, &MainWindow::attributeEditor, tr("Q"));
  editMenu->addAction(tr("DB Br&owser"), this, &MainWindow::dbBrowser, tr("V"));
  toolBoxActions->addAction(
      editMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/moveResizeReshape.png"),
                          tr("&Move/Resize/Reshape"), this, &MainWindow::moveResizeReshape, tr("Shift+R")));

  editMenu->addSeparator();

  editMenu->addAction(tr("Edit P&in Group"), this, &MainWindow::editPinGroup);
  editMenu->addAction(tr("Edit &Net Group"), this, &MainWindow::editNetGroup);
  editMenu->addAction(tr("Edit Pin &Guide"), this, &MainWindow::editPinGuide);
  QMenu* busGuide = editMenu->addMenu(tr("&Bus Guide"));
  busGuide->addAction(QIcon("./iEDA/src/iGUI/res/icon/editBusGuide.png"), tr("&Edit"), this,
                      &MainWindow::editBusGuide);
  busGuide->addAction(tr("&Color"), this, &MainWindow::busColor);
  busGuide->addAction(tr("C&lear Color"), this, &MainWindow::clearBusColor);
  editMenu->addAction(tr("&Pin Editor"), this, &MainWindow::pinEditor);
  editMenu->addSeparator();
  QMenu* wire = editMenu->addMenu(tr("&Wire"));
  wire->addAction(QIcon("./iEDA/src/iGUI/res/icon/editWire.png"), tr("&Edit"), this,
                  &MainWindow::editWire, tr("E"));
  wire->addAction(QIcon("./iEDA/src/iGUI/res/icon/movewire.png"), tr("&Move"), this,
                  &MainWindow::moveWire);
  wire->addAction(tr("&Cut"), this, &MainWindow::cutWire, tr("Shift+X"));
  wire->addAction(tr("S&nap"), this, &MainWindow::snapWire);
  wire->addAction(QIcon("./iEDA/src/iGUI/res/icon/stretch.png"), tr("&Stretch"), this,
                  &MainWindow::stretchWire);
  wire->addSeparator();
  wire->addAction(QIcon("./iEDA/src/iGUI/res/icon/addVia.png"), tr("Add &Via"), this,
                  &MainWindow::addViaWire);
  wire->addAction(QIcon("./iEDA/src/iGUI/res/icon/addPloygon.png"), tr("Add &Polygon"), this,
                  &MainWindow::addPolygonWire);
  editMenu->addAction(tr("C&reate Non Default Rule"), this, &MainWindow::ceateNonDefaultRule);

  /*--------------------create partirionMenu---------------------*/
  QMenu* partitionMenu = menuBar()->addMenu(tr("Partitio&n"));
  partitionMenu->addAction(tr("&Specify Partition"), this, &MainWindow::specifyPartition);
  partitionMenu->addAction(tr("Specify &Black Box"), this, &MainWindow::specifyBlackBox);
  partitionMenu->addAction(tr("Cl&one Place"), this, &MainWindow::clonePlace);

  partitionMenu->addSeparator();

  partitionMenu->addAction(tr("&Create Physical Feedthrough"), this, &MainWindow::createPhysicalFeedthrough);
  partitionMenu->addAction(tr("Show &Wire Crossing"), this, &MainWindow::showWireCrossing);
  partitionMenu->addAction(tr("F&eedthrough Ports"), this, &MainWindow::feedthroughPorts);

  partitionMenu->addSeparator();

  partitionMenu->addAction(tr("Assign &Pin"), this, &MainWindow::assignPin);
  partitionMenu->addAction(tr("&Check Pin Assignment"), this, &MainWindow::checkPinAssignment);
  partitionMenu->addAction(tr("Drive &Timing Budget"), this, &MainWindow::driveTimingBudget);
  partitionMenu->addAction(tr("Commit Partition"), this, &MainWindow::commitPartition);
  partitionMenu->addAction(tr("F&latten Partition"), this, &MainWindow::flattenPartition);
  partitionMenu->addAction(tr("&Assemble Design"), this, &MainWindow::assembleDesign);

  partitionMenu->addSeparator();

  partitionMenu->addAction(tr("Change Partition &View"), this, &MainWindow::changePartitionView);

  /*--------------------create powerMenu---------------------*/
  QMenu* powerMenu = menuBar()->addMenu(tr("Po&wer"));
  powerMenu->addAction(tr("C&onnect Global Net"), this, &MainWindow::connectGlobalNet);
  QMenu* multipSupplyVoltage = powerMenu->addMenu(tr("&Multip Supply Voltage"));
  QMenu* powerPlanning       = powerMenu->addMenu(tr("Power Pla&nning"));
  QMenu* powerAnalysis       = powerMenu->addMenu(tr("&Power Analysis"));
  QMenu* rallAnalysis        = powerMenu->addMenu(tr("&Rail Analysis"));
  QMenu* package             = powerMenu->addMenu(tr("Packag&e"));

  QMenu* reportMenu = powerMenu->addMenu(tr("Repor&t"));
  reportMenu->addAction(tr("PowerGrid &Library"), this, &MainWindow::powerGridLibrary);
  reportMenu->addAction(tr("&Power"), this, &MainWindow::power);
  reportMenu->addAction(tr("Power &Histogram"), this, &MainWindow::powerHistogram);
  reportMenu->addAction(tr("Power Pail R&esult"), this, &MainWindow::powerPailResult);
  reportMenu->addAction(tr("Dynamic &Movie"), this, &MainWindow::dynamicMovie);
  reportMenu->addAction(tr("Dynamic &Waveform"), this, &MainWindow::dynamicWaveform);

  /*--------------------create placeMenu---------------------*/
  QMenu* placeMenu = menuBar()->addMenu(tr("&Place"));
  QMenu* specify   = placeMenu->addMenu(tr("&Specify"));
  placeMenu->addSeparator();
  placeMenu->addAction(tr("Place &Itag"), this, &MainWindow::placeItag);
  placeMenu->addAction(tr("Place Stand&ard Cell"), this, &MainWindow::placeStandardCell);
  placeMenu->addAction(tr("P&lace Spare Cell"), this, &MainWindow::PlaceSpareCell);
  placeMenu->addSeparator();
  placeMenu->addAction(tr("&Refine Placement"), this, &MainWindow::refinePlacement);
  placeMenu->addAction(tr("&ECO Placement"), this, &MainWindow::ecoPlacement);
  placeMenu->addSeparator();
  QMenu* physicalCell = placeMenu->addMenu(tr("P&hysical Cell"));
  QMenu* tieCell      = placeMenu->addMenu(tr("&Tie Hi/Lo Cell"));
  placeMenu->addSeparator();
  QMenu* scanChain = placeMenu->addMenu(tr("Sca&n Chain"));
  placeMenu->addSeparator();
  placeMenu->addAction(tr("Chec&k Placement"), this, &MainWindow::checkPlacement);
  QMenu* display = placeMenu->addMenu(tr("&Display"));
  display->addAction(tr("&Spare Cell"), this, &MainWindow::spareCell);
  display->addAction(tr("&Clear Spare Cell Display"), this, &MainWindow::clearSpareCellDisplay);
  display->addAction(tr("Sc&an Chain"), this, &MainWindow::scanChain);
  display->addAction(tr("&Density Map"), this, &MainWindow::densityMap);
  display->addAction(tr("Pin Density &Map"), this, &MainWindow::pinDensityMap);
  display->addAction(tr("Display &Edge Comstraints"), this, &MainWindow::displayEdgeComstraints);
  display->addAction(tr("Display Cell &Padding"), this, &MainWindow::displayCellPadding);
  display->addAction(tr("Display Cell Stac&k Group"), this, &MainWindow::displayCellStackGroup);
  display->addAction(tr("Display &Implant Group"), this, &MainWindow::displayImplantGroup);
  QMenu* queryDensity = placeMenu->addMenu(tr("&Query Density"));

  /*--------------------create ecoMenu---------------------*/
  QMenu* ecoMenu = menuBar()->addMenu(tr("EC&O"));
  ecoMenu->addAction(tr("O&ptimize Design"), this, &MainWindow::optimizeDesign);
  ecoMenu->addAction(tr("&Interative ECO"), this, &MainWindow::interativeECO);

  /*--------------------create clockMenu---------------------*/
  QMenu* clockMenu = menuBar()->addMenu(tr("&Clock"));
  clockMenu->addAction(tr("CC&Opt Clock Tree Debugger"), this, &MainWindow::ccOptClockTreeDebugger);

  /*--------------------create routeMenu---------------------*/
  QMenu* routeMenu = menuBar()->addMenu(tr("&Route"));
  routeMenu->addAction(tr("&Generate Route Guide"), this, &MainWindow::generateRouteGuide);
  routeMenu->addAction(tr("&Early Global Route"), this, &MainWindow::earlyGlobalRoute);
  routeMenu->addSeparator();
  routeMenu->addAction(tr("&Special Route"), this, &MainWindow::specialRoute);
  QMenu* nanoRoute = routeMenu->addMenu(tr("&NanoRoute"));
  routeMenu->addSeparator();
  QMenu* mentalFill = routeMenu->addMenu(tr("&Mental Fill"));
  QMenu* viaFill    = routeMenu->addMenu(tr("&Via Fill"));

  /*--------------------create timingMenu---------------------*/
  QMenu* timingMenu = menuBar()->addMenu(tr("&Timing"));
  timingMenu->addAction(tr("MMM&C Brower"), this, &MainWindow::mmmcBrower);
  timingMenu->addSeparator();
  timingMenu->addAction(tr("&Generate Capacitance Table"), this, &MainWindow::generateCapacitanceTable);
  timingMenu->addAction(tr("&Extract RC"), this, &MainWindow::extractRC);
  timingMenu->addAction(tr("&Report Timing"), this, &MainWindow::reportTiming);
  timingMenu->addAction(tr("&Debug Timing"), this, &MainWindow::debugTiming);
  timingMenu->addAction(tr("&Create Black Box"), this, &MainWindow::createBlackBox);
  timingMenu->addAction(tr("&Write SDF"), this, &MainWindow::writeSDF);
  timingMenu->addAction(tr("Display Timing &Map"), this, &MainWindow::displayTimingMap);
  timingMenu->addAction(tr("Display &Noise Net"), this, &MainWindow::displayNoiseNet);

  /*--------------------create verifyMenu---------------------*/
  QMenu* verifyMenu = menuBar()->addMenu(tr("Verif&y"));
  verifyMenu->addAction(tr("Verify &DRC"), this, &MainWindow::verifyDRC);
  verifyMenu->addAction(tr("Verify &Geometry"), this, &MainWindow::verifyGeometry);
  verifyMenu->addAction(tr("Verify &Connectivity"), this, &MainWindow::verifyConnectivity);
  verifyMenu->addAction(tr("Verify &Process Antenna"), this, &MainWindow::verifyProcessAntenna);
  verifyMenu->addAction(tr("Verify &AC Limit"), this, &MainWindow::verifyACLimit);
  verifyMenu->addAction(tr("Verify &End Cap"), this, &MainWindow::verifyEndCap);
  verifyMenu->addSeparator();
  verifyMenu->addAction(tr("Verify &Mental Density"), this, &MainWindow::verifyMentalDensity);
  verifyMenu->addAction(tr("Verify C&ut Density"), this, &MainWindow::verifyCutDensity);
  verifyMenu->addAction(tr("Verify P&ower Via"), this, &MainWindow::verifyPowerVia);

  /*--------------------create pegasusMenu---------------------*/
  QMenu* pegasusMenu = menuBar()->addMenu(tr("Pegasus"));

  /*--------------------create toolsMenu---------------------*/
  QMenu* toolsMenu = menuBar()->addMenu(tr("Too&ls"));
  toolsActions << toolsMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/designBrowser.png"),
                                       tr("&Design Broswer"), this, &MainWindow::designBrower);
  QMenu* setMode = toolsMenu->addMenu(tr("setMode"));
  toolsMenu->addAction(tr("Set Global Variable"), this, &MainWindow::setGlobalVariable);
  toolsActions << toolsMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/violationBrowser.png"),
                                       tr("Violation Browser"), this, &MainWindow::violationBrowser);
  toolsMenu->addAction(tr("Layout Viewer"), this, &MainWindow::layoutViewer);
  toolsMenu->addAction(tr("Cell Viewer"), this, &MainWindow::cellViewer);
  toolsMenu->addAction(tr("Schematic Viewer"), this, &MainWindow::schematicViewer);
  toolsMenu->addAction(tr("Log Viewer"), this, &MainWindow::logViewer);
  toolsMenu->addAction(tr("Flightline Browser"), this, &MainWindow::flightlineBrowser);

  QMenu* mixedSignal = toolsMenu->addMenu(tr("Mixed Signal"));
  mixedSignal->addAction(tr("Integration Constraint Editor"), this, &MainWindow::integrationConstraintEditor);
  mixedSignal->addAction(tr("Run VSR"), this, &MainWindow::runVSR);
  mixedSignal->addAction(tr("Pull Block Constraint"), this, &MainWindow::pullBlockConstraint);

  toolsMenu->addAction(tr("Set Multiple CPU Usage"), this, &MainWindow::setMultipleCpuUsage);
  toolsMenu->addSeparator();
  toolsMenu->addAction(tr("Flip Chip"), this, &MainWindow::flipChip);
  toolsMenu->addAction(tr("TSV"), this, &MainWindow::tsv);
  QMenu* conformal = toolsMenu->addMenu(tr("Conformal"));

  QMenu* dfm   = toolsMenu->addMenu(tr("DFM"));
  QMenu* litho = dfm->addMenu(tr("&Litho"));
  litho->addAction(tr("&Verify Litho"), this, &MainWindow::VerifyLitho);
  litho->addAction(tr("&Check Litho Status"), this, &MainWindow::checkLithoStatus);
  QMenu* cmp = dfm->addMenu(tr("&CMP"));
  cmp->addAction(tr("&Verify CMP"), this, &MainWindow::verifyCMP);
  cmp->addAction(tr("&Check CMP Status"), this, &MainWindow::checkCMPStatus);

  toolsMenu->addSeparator();

  QMenu* snapshot = toolsMenu->addMenu(tr("Snapshot"));
  snapshot->addAction(tr("&Create"), this, &MainWindow::createAct);
  snapshot->addAction(tr("&View"), this, &MainWindow::viewAct);

  QMenu* screenCapture = toolsMenu->addMenu(tr("Screen Capture"));
  screenCapture->addAction(tr("&Write To GIF File"), this, &MainWindow::writeToGifFile);
  screenCapture->addAction(tr("&Screen Dump"), this, &MainWindow::screenDump);
  screenCapture->addAction(tr("D&isplay Screen Dump"), this, &MainWindow::displayScreenDump);

  toolBoxActions->addAction(toolsMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/ruler.png"),
                                                 tr("Create Ruler"), this, &MainWindow::createRuler, tr("K")));
  toolsActions << toolsMenu->addAction(QIcon("./iEDA/src/iGUI/res/icon/clearRuler.png"),
                                       tr("Creal All Rulers"), this, &MainWindow::clearAllRulers, tr("Shift+k"));

  /*--------------------create routeMenw---------------------*/
  QMenu* windowsMenu = menuBar()->addMenu(tr("W&indows"));

  /*--------------------create routeMenw---------------------*/
  QMenu* flowsMenu = menuBar()->addMenu(tr("Flow&s"));

  /*--------------------create routeMenw---------------------*/
  QMenu* helpMenu = menuBar()->addMenu(tr("&Help"));
}
