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
