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