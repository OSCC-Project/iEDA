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
#include "tcl_qt.h"

namespace tcl {

GuiTclNotifier* GuiTclNotifier::_notifier = nullptr;

GuiTclNotifier::GuiTclNotifier()
{
  _timer = new QTimer(this);
  _timer->setSingleShot(true);
  QObject::connect(_timer, &QTimer::timeout, this, &GuiTclNotifier::onTimer);
}

void GuiTclNotifier::setup()
{
  Tcl_NotifierProcs notifier;
  notifier.createFileHandlerProc = createFileHandler;
  notifier.deleteFileHandlerProc = deleteFileHandler;
  notifier.setTimerProc = setTimer;
  notifier.waitForEventProc = waitForEvent;
  notifier.initNotifierProc = initNotifier;
  notifier.finalizeNotifierProc = finalizeNotifier;
  notifier.alertNotifierProc = alertNotifier;
  notifier.serviceModeHookProc = serviceModeHook;
  Tcl_SetNotifier(&notifier);
}

void GuiTclNotifier::createFileHandler(int fd, int mask, Tcl_FileProc* proc, ClientData clientData)
{
  getInstance()->deleteFileHandler(fd);

  auto new_handler = new TclQtObject(getInstance(), proc, clientData, mask);
  if (mask & TCL_READABLE) {
    QSocketNotifier* socket_notifier = new QSocketNotifier(fd, QSocketNotifier::Read, new_handler);
    QObject::connect(socket_notifier, &QSocketNotifier::activated, getInstance(), &GuiTclNotifier::onRead);
  }
  if (mask & TCL_WRITABLE) {
    QSocketNotifier* socket_notifier = new QSocketNotifier(fd, QSocketNotifier::Write, new_handler);
    QObject::connect(socket_notifier, &QSocketNotifier::activated, getInstance(), &GuiTclNotifier::onWrite);
  }
  if (mask & TCL_EXCEPTION) {
    QSocketNotifier* socket_notifier = new QSocketNotifier(fd, QSocketNotifier::Exception, new_handler);
    QObject::connect(socket_notifier, &QSocketNotifier::activated, getInstance(), &GuiTclNotifier::onError);
  }
  getInstance()->_handlers.insert(std::make_pair(fd, new_handler));
}

void GuiTclNotifier::deleteFileHandler(int fd)
{
  auto it_exist = getInstance()->_handlers.find(fd);
  if (it_exist == getInstance()->_handlers.end()) {
    return;
  }

  it_exist->second->deleteLater();
  getInstance()->_handlers.erase(it_exist);
}

void GuiTclNotifier::setTimer(Tcl_Time const* tcl_timer)
{
  if (getInstance()->_timer->isActive()) {
    getInstance()->_timer->stop();
  }
  if (tcl_timer) {
    getInstance()->_timer->start(tcl_timer->sec * 1000 + tcl_timer->usec / 1000);
  }
}

int GuiTclNotifier::waitForEvent(Tcl_Time const* tcl_timer)
{
  int time_out = 0;
  if (tcl_timer) {
    time_out = tcl_timer->sec * 1000 + tcl_timer->usec / 1000;
    if (time_out == 0) {
      if (!QCoreApplication::hasPendingEvents()) {
        return 0;
      }
    } else {
      setTimer(tcl_timer);
    }
  }

  QCoreApplication::processEvents(QEventLoop::WaitForMoreEvents);
  return 1;
}

}  // namespace tcl
