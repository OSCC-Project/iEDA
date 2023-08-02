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
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QApplication>
#include <QHBoxLayout>
#include <QMainWindow>
#include <QMenuBar>
#include <QSplitter>
#include <QStatusBar>
#include <QToolBar>
#include <QWidget>

#include "fileimport.h"
#include "guigraphicsscene.h"
#include "guigraphicsview.h"
#include "guiloading.h"
#include "guisearchedit.h"
#include "guisplash.h"
#include "guitree.h"

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  MainWindow(QWidget* parent = nullptr);
  ~MainWindow();

  /// getter
  GuiGraphicsScene* get_scene() { return _scene; }
  GuiGraphicsView* get_View() { return _graphicsView; }

  /// operator
  void initView();
  void updateTree();

 private:
  /// file operation
  void createGuiFile();
  void createMenuFile();
  void createToolbarFile();

  /// floorplan operation
  void createGuiFloorplan();
  void createMenuFloorplan();
  void createToolbarFloorplan();

  /// pdn
  void createGuiPdn();
  void createMenuPdn();
  void createToolbarPdn();

  /// Placement
  void createGuiPlacement();
  void createMenuPlacement();
  void createToolbarPlacement();

  /// CTS
  void createGuiCTS();
  void createMenuCTS();
  void createToolbarCTS();

  /// Routing
  void createGuiRouting();
  void createMenuRouting();
  void createToolbarRouting();

  /// DRC
  void createGuiDRC();
  void createMenuDRC();
  void createToolbarDRC();

  /// search box
  void createSearch();

  /// menu
  void createMenu();

  ///
  void createTabWidget();
  void createToolbar();
  void createStatusbar();

  // scence
  void createScene();

  // control view
  void createControlView();

  // other
  void (*onLoadError)(QString error_msg) = 0;
  void setCoordinate(const QString& position);

  GuiTree* _control_tree   = nullptr;
  GuiGraphicsScene* _scene = nullptr;

  QTabWidget* tabWidget  = nullptr;
  QMainWindow* layoutTab = nullptr;

  QLabel* coordinate;

  GuiLoading* l      = nullptr;
  GuiSplash* _splash = nullptr;

  FileImport* _importFile = nullptr;

  QList<QAction*> FileActions;
  QList<QAction*> editActions;
  QList<QAction*> viewActions;
  QList<QAction*> toolsActions;
  QList<QAction*> reportActions;
  QActionGroup* toolBoxActions = nullptr;
  QList<QAction*> wireEditActions;
  GuiGraphicsView* _graphicsView = nullptr;
  double value                   = 3.28;
  GuiSearchEdit* _edit_search    = nullptr;

 public slots:
  void import();

  //////////////
  void restore();
  void eco();
  void save();
  void createOA();
  void check();
  void loadPartition();
  void loadFloorplan();
  void loadIOFile();
  void loadPlace();
  void loadDEF();
  void loadPDEF();
  void loadSPFE();
  void loadSDF();
  void loadOACellview();
  void savePartition();
  void saveFloorplan();
  void saveIOFile();
  void savePlace();
  void saveNetlist();
  void saveTestcase();
  void saveDEF();
  void savePDEF();
  void saveTimingBudget();
  void saveGDS_OASIS();
  void saveOACellview();
  void summary();
  void selectedObject();
  void gateCount();
  void netlistStatistics();
  void fit();
  void redraw();
  void setPreference();
  void allColor();
  void setLineColor();
  void goTo();
  void findSelect();
  void deselectAll();
  void original();
  void clearoriginal();
  void editHighlightColor();
  void dimBackground();
  void undo();
  void redo();
  void copy();
  void attributeEditor();
  void dbBrowser();
  void moveResizeReshape();
  void editPinGroup();
  void editNetGroup();
  void editPinGuide();
  void pinEditor();
  void ceateNonDefaultRule();
  void editWire();
  void moveWire();
  void cutWire();
  void snapWire();
  void stretchWire();
  void addViaWire();
  void addPolygonWire();
  void editBusGuide();
  void busColor();
  void clearBusColor();

  void specifyPartition();
  void specifyBlackBox();
  void clonePlace();
  void showWireCrossing();
  void createPhysicalFeedthrough();
  void feedthroughPorts();
  void assignPin();
  void checkPinAssignment();
  void driveTimingBudget();
  void commitPartition();
  void flattenPartition();
  void assembleDesign();
  void changePartitionView();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// floorplan
  void InitFloorplan();
  void FpPlaceIOPad();
  void FpPlaceMacro();
  void FpPlaceInstance();
  void FpPlaceAllInstance();

  void FpPlaceIOPort();
  void FpPlaceAllPorts();
  void FpPlaceIOFiller();

  void FpTapCell();

  void PdnCreateNet();
  void PdnConnectIO();
  void PdnPlacePorts();
  void PdnAddStdCellStripe();
  void PdnAddStripeGrid();
  void PdnConnectAllLayers();

  void PlaceInstance();

  void CtsCreateNets();

  void RoutingNets();

  void DrcCheckDef();

  void resizeFloorplan();
  void floorplanToolbox();
  void traceMacro();
  void macroTimingSlackDisplay();
  void snapFloorplan();
  void checkFloorplan();
  void clearFloorplan();
  void generateRegroupedNetlist();
  void generateFloorplan();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// pdn
  void connectGlobalNet();
  void powerGridLibrary();
  void power();
  void powerHistogram();
  void powerPailResult();
  void dynamicMovie();
  void dynamicWaveform();
  void placeItag();
  void placeStandardCell();
  void PlaceSpareCell();
  void refinePlacement();
  void ecoPlacement();
  void checkPlacement();
  void spareCell();
  void clearSpareCellDisplay();
  void scanChain();
  void densityMap();
  void pinDensityMap();
  void displayEdgeComstraints();
  void displayCellPadding();
  void displayCellStackGroup();
  void displayImplantGroup();
  void optimizeDesign();
  void interativeECO();
  void ccOptClockTreeDebugger();
  void generateRouteGuide();
  void earlyGlobalRoute();
  void specialRoute();
  void mmmcBrower();
  void generateCapacitanceTable();
  void extractRC();
  void reportTiming();
  void debugTiming();
  void createBlackBox();
  void writeSDF();
  void displayTimingMap();
  void displayNoiseNet();
  void verifyDRC();
  void verifyGeometry();
  void verifyConnectivity();
  void verifyProcessAntenna();
  void verifyACLimit();
  void verifyEndCap();
  void verifyMentalDensity();
  void verifyCutDensity();
  void verifyPowerVia();
  void designBrower();
  void setGlobalVariable();
  void violationBrowser();
  void layoutViewer();
  void cellViewer();
  void schematicViewer();
  void logViewer();
  void flightlineBrowser();
  void integrationConstraintEditor();
  void runVSR();
  void pullBlockConstraint();
  void setMultipleCpuUsage();
  void flipChip();
  void tsv();
  void VerifyLitho();
  void checkLithoStatus();
  void verifyCMP();
  void checkCMPStatus();
  void createAct();
  void viewAct();
  void writeToGifFile();
  void screenDump();
  void displayScreenDump();
  void createRuler();
  void clearAllRulers();
  void clearAllHighlight();
  void clearSelectedHighlight();
  void inAct();
  void outAct();
  void Selected();
  void previous();
  void nextAct();
  void upAct();
  void downAct();
  void leftAct();
  void rightAct();
  void cutRectilinear();
  void createSizeBlockage();
  void createPlacementBlockage();
  void createRoutingBlockage();
  void createPinBlockage();
  void selectFlightline();
  void editWireTool();
  void duplicateSelectedWires();
  void splitSelectedWires();
  void mergeSelectedWires();
  void trimSelectedWires();
  void deleteWires();
  void selectByBox();
  void search();
};

#endif  // MAINWINDOW_H
