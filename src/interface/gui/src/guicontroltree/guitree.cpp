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
#include "guitree.h"

#include <QDebug>
#include <QPainter>
#include <QStyleFactory>

#include "guijsonparser.h"
#include "guixmlparser.h"

GuiTree::GuiTree(QWidget *parent) : QTreeWidget(parent) {
  setStyleSheet(
      "QTreeView::item:hover{background-color:rgb(192,255,192)}"
      "QTreeView::item:selected{background-color:rgb(255,127,127)}"
      "QTreeView::item{height:20}"
      "QTreeView::branch:has-siblings:!adjoins-item {border-image: "
      "url(:/icon/vline.png) 0;}"
      "QTreeView::branch:has-siblings:adjoins-item {border-image: "
      "url(:/icon/branch-more.png) 0;}"
      "QTreeView::branch:!has-children:!has-siblings:adjoins-item "
      "{border-image: url(:/icon/branch-end.png) 0;}"
      "QTreeView::branch:has-children:!has-siblings:closed,"
      "QTreeView::branch:closed:has-children:has-siblings {border-image: none; "
      "image: url(:/icon/branch-closed.png);}"
      "QTreeView::branch:open:has-children:!has-siblings,"
      "QTreeView::branch:open:has-children:has-siblings {border-image: none; "
      "image: url(:/icon/branch-open.png);}");

  initHeader();
  initByTreeNode(guiConfig->get_shape_tree());
  initByTreeNode(guiConfig->get_instance_tree());
  initByTreeNode(guiConfig->get_net_tree());
  //   initByTreeNode(guiConfig->get_clock_tree());
  initByTreeNode(guiConfig->get_specialnet_tree());
  initByTreeNode(guiConfig->get_trackgrid_tree());
  initByTreeNode(guiConfig->get_drc_tree());
  initByTreeNode(guiConfig->get_layer_tree());

  // bind slots
  connect(this, SIGNAL(itemClicked(QTreeWidgetItem *, int)), this, SLOT(onItemClicked(QTreeWidgetItem *, int)));
}

void GuiTree::initHeader() {
  setColumnCount((int)igui::GuiTreeColEnum::kMax);

  headerItem()->setText((int)igui::GuiTreeColEnum::kLayerColor, "");
  headerItem()->setTextAlignment((int)igui::GuiTreeColEnum::kLayerColor, Qt::AlignVCenter);

  headerItem()->setText((int)igui::GuiTreeColEnum::kName, "");
  headerItem()->setTextAlignment((int)igui::GuiTreeColEnum::kName, Qt::AlignVCenter);

  headerItem()->setText((int)igui::GuiTreeColEnum::kVisible, "V");
  headerItem()->setTextAlignment((int)igui::GuiTreeColEnum::kVisible, Qt::AlignVCenter);
}

/**
 * @Brief : init tree by config
 * @param  tree_node
 */
void GuiTree::initByTreeNode(igui::GuiTreeNode &tree_node) {
  QTreeWidgetItem *topItem = new QTreeWidgetItem();
  topItem->setText((int)igui::GuiTreeColEnum::kLayerColor, "");
  topItem->setTextAlignment((int)igui::GuiTreeColEnum::kLayerColor, Qt::AlignVCenter | Qt::AlignLeft);
  topItem->setText((int)igui::GuiTreeColEnum::kName, QString::fromStdString(tree_node.get_node_name()));
  topItem->setTextAlignment((int)igui::GuiTreeColEnum::kName, Qt::AlignVCenter | Qt::AlignLeft);
  topItem->setCheckState((int)igui::GuiTreeColEnum::kVisible, Qt::CheckState::Unchecked);
  topItem->setFlags(topItem->flags() | Qt::ItemIsAutoTristate);
  addTopLevelItem(topItem);

  auto &option_list = tree_node.get_option_list();
  for (int i = 0; i < option_list.size(); ++i) {
    QTreeWidgetItem *childItem = new QTreeWidgetItem();
    childItem->setText((int)igui::GuiTreeColEnum::kLayerColor, "");
    childItem->setTextAlignment((int)igui::GuiTreeColEnum::kLayerColor, Qt::AlignVCenter | Qt::AlignLeft);
    if (tree_node.isLayerList()) {
      childItem->setBackground((int)igui::GuiTreeColEnum::kLayerColor,
                               QBrush(attributeInst->getLayerColor(option_list[i]._name), Qt::BrushStyle::SolidPattern));
    }
    childItem->setText((int)igui::GuiTreeColEnum::kName, QString::fromStdString(option_list[i]._name));
    childItem->setTextAlignment((int)igui::GuiTreeColEnum::kName, Qt::AlignVCenter | Qt::AlignRight);
    childItem->setCheckState((int)igui::GuiTreeColEnum::kVisible, option_list[i]._visible == igui::GuiOptionValue::kOn
                                                                      ? Qt::CheckState::Checked
                                                                      : Qt::CheckState::Unchecked);
    childItem->setFlags(childItem->flags() | Qt::ItemIsAutoTristate);

    topItem->addChild(childItem);
  }

  expandAll();
}
/**
 * @Brief : update tree node by config
 * @param  tree_node
 */
void GuiTree::updateTreeNode(igui::GuiTreeNode &tree_node) {
  QString node_name                  = QString::fromStdString(tree_node.get_node_name());
  QList<QTreeWidgetItem *> item_list = findItems(node_name, Qt::MatchExactly, (int)igui::GuiTreeColEnum::kName);
  std::cout << "item_list size = " << item_list.size() << std::endl;
  for (auto item : item_list) {
    /// set state
    item->setCheckState((int)igui::GuiTreeColEnum::kVisible, Qt::CheckState::Unchecked);

    /// clear child
    int child_num = item->childCount();
    while (child_num > 0) {
      auto child = item->child(child_num);
      item->removeChild(child);

      child_num--;
    }

    /// reset child
    auto &option_list = tree_node.get_option_list();
    std::cout << " option_list.size() = " << option_list.size() << std::endl;
    for (int i = 0; i < option_list.size(); ++i) {
      QTreeWidgetItem *childItem = new QTreeWidgetItem();
      childItem->setText((int)igui::GuiTreeColEnum::kLayerColor, "");
      childItem->setTextAlignment((int)igui::GuiTreeColEnum::kLayerColor, Qt::AlignVCenter | Qt::AlignLeft);
      if (tree_node.isLayerList()) {
        childItem->setBackground((int)igui::GuiTreeColEnum::kLayerColor,
                                 QBrush(attributeInst->getLayerColor(option_list[i]._name), Qt::BrushStyle::SolidPattern));
      }
      childItem->setText((int)igui::GuiTreeColEnum::kName, QString::fromStdString(option_list[i]._name));
      childItem->setTextAlignment((int)igui::GuiTreeColEnum::kName, Qt::AlignVCenter | Qt::AlignRight);
      childItem->setCheckState((int)igui::GuiTreeColEnum::kVisible, option_list[i]._visible == igui::GuiOptionValue::kOn
                                                                        ? Qt::CheckState::Checked
                                                                        : Qt::CheckState::Unchecked);
      childItem->setFlags(childItem->flags() | Qt::ItemIsAutoTristate);

      item->addChild(childItem);
    }

    expandAll();
  }
}

void GuiTree::updateLayer() { updateTreeNode(guiConfig->get_layer_tree()); }

void GuiTree::onItemClicked(QTreeWidgetItem *item, int column) {
  if (column != (int)igui::GuiTreeColEnum::kVisible)
    return;

  Qt::CheckState state = item->checkState(column);
  /// save data
  std::string parent_name = "";
  std::string node_name   = "";

  /// case 1 : click on the parent node
  if (item->childCount() > 0) {
    node_name = item->text((int)igui::GuiTreeColEnum::kName).toStdString();
  } else {
    /// case 2 : click on the child node
    QTreeWidgetItem *parent_item = item->parent();
    if (parent_item != nullptr) {
      parent_name = parent_item->text((int)igui::GuiTreeColEnum::kName).toStdString();
      node_name   = item->text((int)igui::GuiTreeColEnum::kName).toStdString();
    }
  }

  guiConfig->saveData(state == Qt::CheckState::Checked ? true : false, node_name, parent_name);

  if (_db_setup != nullptr) {
    _db_setup->update(node_name, parent_name);
  }
}
