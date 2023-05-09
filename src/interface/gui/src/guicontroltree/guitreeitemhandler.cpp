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
/**
 * @file guitreeitemhandler.cpp
 * @author Wang Jun (wen8365@gmail.com)
 * @brief
 * @version 0.1
 * @date 2021-07-02
 *
 *
 *
 */

#include "guitreeitemhandler.h"

#include <QDebug>

// bool GuiTreeItemHandler::error(const QXmlParseException &exception) {
//   qDebug() << "parse error at line " << exception.lineNumber() << ", column " << exception.columnNumber() << ":"
//            << exception.message();
//   return false;
// }

// bool GuiTreeItemHandler::fatalError(const QXmlParseException &exception) {
//   qDebug() << "Fatal error at line " << exception.lineNumber() << ", column " << exception.columnNumber() << ":"
//            << exception.message();
//   return false;
// }

// bool GuiTreeItemHandler::warning(const QXmlParseException &exception) {
//   qDebug() << "parse warning at line " << exception.lineNumber() << ", column " << exception.columnNumber() << ":"
//            << exception.message();
//   return false;
// }

bool GuiTreeItemHandler::startElement(const QString &namespace_URI, const QString &local_name, const QString &q_name,
                                      const QXmlAttributes &atts) {
  GuiControlTreeItem *item = new GuiControlTreeItem(local_name);
  item->initColumns();

  if (!_item_stack.isEmpty()) {
    GuiControlTreeItem *last_item = _item_stack.top();
    QString last_name             = last_item->get_label();
    int level                     = QString::compare(last_name, local_name);
    if (level < 0)  // means this node is the child of the node at top in nodeStack
    {
      last_item->addChild(item);
    } else {
      // level == 0, will not happen in normal,
      // because if the privious node has the same level,
      // it will remove from _item_stack at endElement.
      // level > 0, means this node is the father of the privious node,
      // impossible.
      qDebug() << "invalid level relationship: " << level << ", " << last_name << " is not lower than " << local_name
               << "\n";
      return false;
    }
  } else {
    _root = item;
  }
  _item_stack.push(item);

  return true;
}

bool GuiTreeItemHandler::endElement(const QString &namespace_URI, const QString &local_name, const QString &q_name) {
  _item_stack.pop();
  return true;
}

bool GuiTreeItemHandler::characters(const QString &ch) {
  if (ch.trimmed().isEmpty())  // when end element, this function will be
                               // invoked with empty ch
    return true;
  // when start element, set current item label with ch
  GuiControlTreeItem *item = _item_stack.top();
  const QString &label     = ch.trimmed();
  item->set_label(label);
  return true;
}
