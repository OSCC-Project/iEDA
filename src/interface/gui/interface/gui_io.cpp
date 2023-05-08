/**
 * iEDA
 * Copyright (C) 2021  PCL
 *
 * This program is free software;
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "gui_io.h"

namespace igui {
  GuiIO* GuiIO::_instance = nullptr;

  void GuiIO::init() {
    int argc    = 1;
    char** argv = nullptr;
    _app        = new QApplication(argc, argv);
  }

  void GuiIO::startGUI(IdbBuilder* builder, std::string type) {
    if (_gui_win == nullptr) {
      //   int argc    = 1;
      //   char** argv = nullptr;
      //   static QApplication win(argc, argv);
      _gui_win = new MainWindow();
      _gui_win->showMaximized();
      _gui_win->get_scene()->createChip(builder, type);
      _gui_win->updateTree();
      //   win.exec();
    } else {
      _gui_win->showMaximized();
      _gui_win->get_scene()->createChip(builder, type);
      _gui_win->updateTree();
    }
  }

  void GuiIO::showWindow() { _gui_win->showMaximized(); }

  void GuiIO::hideWindow() { _gui_win->hide(); }

  int GuiIO::exec() {
    if (_app != nullptr) {
      return _app->exec();
    }

    return 0;
  }

  void GuiIO::timerStart(int ms) {
    _gui_win->get_scene()->timerCreate();
    _gui_win->get_scene()->timerStart(ms);
  }

  void GuiIO::timerStop() { _gui_win->get_scene()->timerStop(); }

}  // namespace igui