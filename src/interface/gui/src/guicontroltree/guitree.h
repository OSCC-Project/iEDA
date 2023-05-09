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

#ifndef GUITREE_H
#define GUITREE_H

#include <QMouseEvent>
#include <QTreeWidget>

#include "../guiDB/dbsetup.h"
#include "guiConfig.h"
#include "guiConfigTree.h"
#include "guiattribute.h"

class GuiTree : public QTreeWidget {
  Q_OBJECT
 public:
  GuiTree(QWidget* parent = nullptr);
  virtual ~GuiTree() = default;

  void updateLayer();
  void setDbSetup(DbSetup* db_setup) { _db_setup = db_setup; }

 public slots:
  void onItemClicked(QTreeWidgetItem* item, int column);

 protected:
  //   void mousePressEvent(QMouseEvent* event) override;

 private:
  DbSetup* _db_setup = nullptr;

  void initHeader();

  void initByTreeNode(igui::GuiTreeNode& tree_node);

  void updateTreeNode(igui::GuiTreeNode& tree_node);
};

#endif  // GUITREE_H
