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
#pragma once

#include <tcl.h>

#include <QCoreApplication>
#include <QSocketNotifier>
#include <QTimer>
#include <iostream>
#include <map>

namespace tcl {

class TclQtObject : public QObject
{
  Q_OBJECT
 public:
  TclQtObject(QObject* parent, Tcl_FileProc* proc, ClientData clientData, int mask)
      : QObject(parent), _proc(proc), _clientData(clientData), _mask(mask)
  {
  }

  void tclCallback(int type, int fd)
  {
    if (!(_mask & type)) {
      return;
    }

    (*_proc)(_clientData, type);
  }

 private:
  Tcl_FileProc* _proc;
  ClientData _clientData;
  int _mask;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class GuiTclNotifier : public QObject
{
  Q_OBJECT
 public:
  static GuiTclNotifier* getInstance()
  {
    if (!_notifier) {
      _notifier = new GuiTclNotifier();
    }
    return _notifier;
  }
  static void setup();
  static void createFileHandler(int fd, int mask, Tcl_FileProc* proc, ClientData clientData);
  static void deleteFileHandler(int fd);
  static void setTimer(Tcl_Time const* tcl_timer);
  static int waitForEvent(Tcl_Time const* tcl_timer);
  static void* initNotifier() { return 0; }
  static void finalizeNotifier(ClientData clientData) {}
  static void alertNotifier(ClientData clientData) {}
  static void serviceModeHook(int mode) {}

 public slots:
  void onRead(int fd) { tclCallback<TCL_READABLE>(fd); }
  void onWrite(int fd) { tclCallback<TCL_WRITABLE>(fd); }
  void onError(int fd) { tclCallback<TCL_EXCEPTION>(fd); }
  void onTimer() { Tcl_ServiceAll(); }

 private:
  std::map<int, TclQtObject*> _handlers;
  QTimer* _timer;
  static GuiTclNotifier* _notifier;

  GuiTclNotifier();
  ~GuiTclNotifier() = default;

  template <int TclActivityType>
  static void tclCallback(int fd)
  {
    auto it = getInstance()->_handlers.find(fd);
    if (it == getInstance()->_handlers.end()) {
      std::cout << "No registered file handler for fd = " << fd << std::endl;
      return;
    }

    it->second->tclCallback(TclActivityType, fd);
  }
};

}  // namespace tcl
