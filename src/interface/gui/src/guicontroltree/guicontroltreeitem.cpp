/**
 * @file guicontroltreeitem.cpp
 * @author Wang Jun (wen8365@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-07-02
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "guicontroltreeitem.h"

#include <QDebug>

GuiTreeItemColumn GuiControlTreeItem::_item_column = GuiTreeItemColumn();

void GuiControlTreeItem::lockColumns(QList<int> columns) {
  QListIterator<int> iterator(columns);
  while (iterator.hasNext()) {
    int column = iterator.next();
    _lock_columns.insert(column, true);
  }
}

void GuiControlTreeItem::unlockColumns(QList<int> columns) {
  QListIterator<int> iterator(columns);
  while (iterator.hasNext()) {
    int column = iterator.next();
    if (_lock_columns.value(column)) _lock_columns.remove(column);
  }
}

void GuiControlTreeItem::addSlave(int master, int slave) {
  QMap<int, QList<int>>::iterator slaves_itr = _m_s.find(master);
  if (slaves_itr == _m_s.end()) {
    QList<int> slaves;
    slaves.append(slave);
    _m_s.insert(master, slaves);
  } else {
    QList<int> slaves = *slaves_itr;
    slaves.append(slave);
  }
}

void GuiControlTreeItem::initColumns(Qt::CheckState state) {
  QListIterator<ColumnInfo> c_itr(_item_column.get_columns());
  while (c_itr.hasNext()) {
    ColumnInfo ci = c_itr.next();
    switch (ci.type) {
      case ColumnInfo::LABEL:
        setText(ci.col, text(_item_column.labelCol()));
        setTextAlignment(ci.col, Qt::AlignVCenter);
        break;
      case ColumnInfo::CHECK:
        setCheckState(ci.col, state);
        setFlags(flags() | Qt::ItemIsAutoTristate);
        break;
      default:
        break;
    }
    if (ci.master > -1) addSlave(ci.master, ci.col);

    if (ci.master > -1 && checkState(ci.master) == Qt::Unchecked)
      lockColumns(getSlaves(ci.master));
  }
}
