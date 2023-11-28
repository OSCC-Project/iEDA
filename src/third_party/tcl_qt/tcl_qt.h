// Implementation of classes to integrate the Qt and Tcl event loops, part of EDASkel, a sample EDA app
// Copyright (C) 2010 Jeffrey Elliot Trull <edaskel@att.net>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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
