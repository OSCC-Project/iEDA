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
#ifndef COMPLETELINEEDIT_H
#define COMPLETELINEEDIT_H
#include <QDebug>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QListView>
#include <QStringList>
#include <QStringListModel>

class GuiSearchEdit : public QLineEdit {
  Q_OBJECT
 public:
  GuiSearchEdit(QWidget *parent = 0);
  ~GuiSearchEdit();
  QStringList get_names() { return _names; }
  void set_name_list(QStringList list) { _names = list; }

 public slots:
  void setCompleter(const QString &text);
  void completeText(const QModelIndex &index);

 signals:
  void search();

 protected:
  virtual void keyPressEvent(QKeyEvent *e);
  virtual void focusOutEvent(QFocusEvent *e);

 private slots:
  void replyMoveSignal();

 private:
  QListView *_list_view;
  QStringList _names;
  QStringListModel *_model;
};
#endif  // COMPLETELINEEDIT_H
