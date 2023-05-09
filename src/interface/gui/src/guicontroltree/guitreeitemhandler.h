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
 * @file guitreeitemhandler.h
 * @author Wang Jun (wen8365@gmail.com)
 * @brief Handler serve in parsing control tree's xml configuration files
 * @version 0.1
 * @date 2021-07-02
 *
 *
 *
 */

#ifndef GUIHANDLER_H
#define GUIHANDLER_H

#include <QDebug>
#include <QStack>
#include <QXmlDefaultHandler>

#include "guicontroltreeitem.h"

class GuiTreeItemHandler : public QXmlDefaultHandler {
 public:
  GuiTreeItemHandler() : QXmlDefaultHandler(){};
  virtual ~GuiTreeItemHandler() {
    delete _root;
    _root = nullptr;
  };
  bool startElement(const QString& namespace_URI, const QString& local_name, const QString& q_name,
                    const QXmlAttributes& atts) override;
  bool endElement(const QString& namespace_URI, const QString& local_name, const QString& q_name) override;
  bool characters(const QString& ch) override;

  //   bool warning(const QXmlParseException& exception) override;
  //   bool error(const QXmlParseException& exception) override;
  //   bool fatalError(const QXmlParseException& exception) override;

  QTreeWidgetItem* get_root() { return _root; };

 private:
  QStack<GuiControlTreeItem*> _item_stack;
  QTreeWidgetItem* _root;
};

#endif  // GUIHANDLER_H
