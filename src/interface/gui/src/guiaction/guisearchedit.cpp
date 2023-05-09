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
#include "guisearchedit.h"

#include <QDebug>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QListView>
#include <QPushButton>
#include <QStringListModel>
GuiSearchEdit::GuiSearchEdit(QWidget *parent) : QLineEdit(parent) {
  setEchoMode(QLineEdit::EchoMode::Normal);

  _list_view = new QListView(this);
  _list_view->setStyleSheet(
      "QListView::item:selected {border: 1px solid #6a6ea9;}"
      "QListView::item:selected:!active {background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #ABAFE5, stop: 1 "
      "#8588B2);}"
      "QListView::item:selected:active {background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #6a6ea9, stop: 1 "
      "#888dd9);}"
      "QListView::item:hover{background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #6a6ea9, stop: 1 #888dd9);}");
  _model = new QStringListModel(this);
  _list_view->setWindowFlags(Qt::ToolTip);
  connect(this, SIGNAL(textChanged(const QString &)), this, SLOT(setCompleter(const QString &)));
  connect(_list_view, SIGNAL(clicked(const QModelIndex &)), this, SLOT(completeText(const QModelIndex &)));
}
GuiSearchEdit::~GuiSearchEdit() { }

void GuiSearchEdit::focusOutEvent(QFocusEvent *e) {
  //  listView->hide();
}

void GuiSearchEdit::replyMoveSignal() { _list_view->hide(); }

void GuiSearchEdit::keyPressEvent(QKeyEvent *e) {
  //   if (!_list_view->isHidden())
  //   {
  //     int key = e->key();
  //     int count = _list_view->model()->rowCount();
  //     QModelIndex currentIndex = _list_view->currentIndex();
  //     if (Qt::Key_Down == key)
  //     {
  //       int row = currentIndex.row() + 1;
  //       if (row >= count)
  //       {
  //         row = 0;
  //       }
  //       QModelIndex index = _list_view->model()->index(row, 0);
  //       _list_view->setCurrentIndex(index);
  //     } else if (Qt::Key_Up == key)
  //     {
  //       int row = currentIndex.row() - 1;
  //       if (row < 0)
  //       {
  //         row = count - 1;
  //       }
  //       QModelIndex index = _list_view->model()->index(row, 0);
  //       _list_view->setCurrentIndex(index);
  //     } else if (Qt::Key_Escape == key)
  //     {
  //       _list_view->hide();
  //     } else if (Qt::Key_Enter == key || Qt::Key_Return == key)
  //     {
  //       if (currentIndex.isValid())
  //       {
  //         QString text = _list_view->currentIndex().data().toString();
  //         setText(text);
  //       }
  //       _list_view->hide();
  //     } else
  //     {
  //       _list_view->hide();
  //       QLineEdit::keyPressEvent(e);
  //     }
  //   } else
  //   {
  //     QLineEdit::keyPressEvent(e);
  //   }

  if (Qt::Key_Return == e->key() || Qt::Key_Enter == e->key()) {
    search();
  }

  QLineEdit::keyPressEvent(e);
}

void GuiSearchEdit::setCompleter(const QString &text) {
  if (text.isEmpty()) {
    _list_view->hide();
    return;
  }
  if ((text.length() > 1) && (!_list_view->isHidden())) {
    return;
  }
  QStringList sl;

  //   qDebug() << _names.size();
  foreach (QString names, _names) {
    // model1
    //    if (names.contains(text))
    //    {
    //      sl << names;
    //    }
    // model2
    if (names.indexOf(text, 0, Qt::CaseInsensitive) == 0)
      sl << names;
  }
  _model->setStringList(sl);
  _list_view->setModel(_model);
  if (_model->rowCount() == 0) {
    return;
  }

  _list_view->setMinimumWidth(width());
  _list_view->setMaximumWidth(width());
  QPoint p(0, height());
  int x = mapToGlobal(p).x();
  int y = mapToGlobal(p).y() + 1;
  _list_view->move(x, y);
  _list_view->show();
}

void GuiSearchEdit::completeText(const QModelIndex &index) {
  QString text = index.data().toString();
  setText(text);
  _list_view->hide();
}
