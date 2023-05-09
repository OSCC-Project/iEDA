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
#include "guiConfig.h"
#include "guispeedupclocktree.h"
#include "idbfastsetup.h"
#include "omp.h"

void IdbSpeedUpSetup::showClockTree(std::vector<iplf::CtsTreeNodeMap*>& _node_list) {
  set_type(DbSetupType::kClockTree);

  if (_node_list.size() == 0) {
    return;
  }

  IdbCore* db_core  = _layout->get_core();
  QRectF boudingbox = _transform.db_to_guidb_rect(db_core->get_bounding_box());

  /// calculate step first
  qreal unit_step = get_leaf_step(_node_list);

  qreal x_offset = 0;
  for (auto tree_map : _node_list) {
    auto root_node = tree_map->get_root();

    if (root_node == nullptr) {
      std::cout << "[Error] : cts tree data has no root" << std::endl;
      continue;
    }

    qreal node_width = unit_step * root_node->get_leaf_num() * 2;

    x_offset += node_width;

    GuiSpeedupClockTreeItem* root_item = buildNodeItem(
        root_node, QPointF(boudingbox.x() + x_offset - node_width / 2, boudingbox.y() + boudingbox.height() / 3), unit_step);
  }

  //   DbSetup::fitView(_fit_ux - _fit_lx, _fit_uy - _fit_ly);

  //   _gui_design->get_clock_list()->update();
}

GuiSpeedupClockTreeItem* IdbSpeedUpSetup::buildNodeItem(iplf::CtsTreeNode* clk_node, QPointF pt, qreal unit_step) {
  if (clk_node == nullptr) {
    return nullptr;
  }

  /// set item
  GuiSpeedupClockTreeItem* item = nullptr;
  if (clk_node->is_root()) {
    item = _gui_design->get_clock_list()->add_item(GuiClockType::kRoot);
    item->add_polygon(pt);
  } else if (clk_node->is_leaf()) {
    item = _gui_design->get_clock_list()->add_item(GuiClockType::kLeaf);
    item->add_leaf(pt);
  } else {
    item = _gui_design->get_clock_list()->add_item(GuiClockType::kNode);
    item->add_polygon(pt);
  }

  /// if leaf, return
  if (clk_node->is_leaf()) {
    return item;
  }

  /// build child
  buildChildNode(item, clk_node, pt, unit_step);

  return item;
}

qreal IdbSpeedUpSetup::get_leaf_step(std::vector<iplf::CtsTreeNodeMap*>& _node_list) {
  if (_node_list.size() == 0) {
    return 0;
  }

  int64_t leaf_num = 0;
  for (auto tree_map : _node_list) {
    auto root_node = tree_map->get_root();

    if (root_node == nullptr) {
      std::cout << "[Error] : cts tree data has no root" << std::endl;
      continue;
    }

    leaf_num += root_node->get_leaf_num();
  }

  if (leaf_num == 0) {
    return 0;
  }

  IdbCore* db_core  = _layout->get_core();
  QRectF boudingbox = _transform.db_to_guidb_rect(db_core->get_bounding_box());
  /// calculate step first
  return boudingbox.width() / leaf_num;
}

qreal IdbSpeedUpSetup::get_min_delay(iplf::CtsTreeNode* clk_node) {
  auto child_node_list = clk_node->get_child_list();

  /// calculate connected coordinate y
  qreal min_delay = 0;
  if (child_node_list.size() > 1) {
    for (auto child_node : child_node_list) {
      if (min_delay == 0) {
        min_delay = child_node->get_delay();
        continue;
      }

      min_delay = std::min(min_delay, child_node->get_delay());
    }
  }

  return min_delay;
}

void IdbSpeedUpSetup::buildChildNode(GuiSpeedupClockTreeItem* item, iplf::CtsTreeNode* clk_node, QPointF pt,
                                     qreal unit_step) {
  auto& child_list  = clk_node->get_child_list();
  int64_t child_num = child_list.size();
  /// build all child
  if (child_num == 0) {
    /// no child
    return;
  } else if (child_num == 1) {
    /// child number == 1
    /// build line
    QPointF child_pt = QPointF(pt.x(), pt.y() + child_list[0]->get_delay() * 10);
    item             = addPoint(item, pt, child_pt);

    /// build node
    GuiSpeedupClockTreeItem* child_item = buildNodeItem(child_list[0], child_pt, unit_step);
  } else {
    /// calculate connected coordinate y
    qreal min_delay = get_min_delay(clk_node);

    /// child number > 2
    QPointF connected_pt = QPointF(pt.x(), pt.y() + min_delay * 6);
    if (min_delay > 0) {
      item = addPoint(item, pt, connected_pt);
    }

    /// build child leaf, and let leaf in the center view
    auto child_leaf_list   = clk_node->get_child_leaf_list();
    int64_t child_leaf_num = get_nodelist_child_num(child_leaf_list);

    QPointF child_connected_pt =
        child_leaf_num > 1 ? QPointF(pt.x() - unit_step * clk_node->get_leaf_num() / 2 + unit_step * child_leaf_num / 2,
                                     pt.y() + min_delay * 8)
                           : connected_pt;
    if (child_leaf_num > 1) {
      item = addPoint(item, connected_pt, child_connected_pt);
    }

    qreal leaf_start_x  = pt.x() - unit_step * clk_node->get_leaf_num() / 2;  /// from left side
    int64_t leaf_offset = -1;
    for (auto leaf : child_leaf_list) {
      int64_t leaf_num = leaf->get_leaf_num() == 0 ? 1 : leaf->get_leaf_num();
      leaf_offset += leaf_num;

      qreal x = leaf_start_x + unit_step * leaf_offset;
      qreal y = pt.y() + leaf->get_delay() * 10;

      /// add line to item
      item = addPoint(item, child_connected_pt, QPointF(x, y));

      /// add line to item
      GuiSpeedupClockTreeItem* child_item = buildNodeItem(leaf, QPointF(x, y), unit_step);
    }

    /// build child node
    auto child_node_list   = clk_node->get_child_node_list();
    int64_t node_offset    = 0;
    int64_t child_node_num = child_node_list.size();
    /// split child nodes into 2 parts beside the leaf list
    qreal node_start_x = leaf_start_x + unit_step * child_leaf_num;

    for (int i = 0; i < child_node_num; i++) {
      int64_t leaf_num = child_node_list[i]->get_leaf_num() == 0 ? 1 : child_node_list[i]->get_leaf_num();

      node_offset += leaf_num;
      qreal x = node_start_x + unit_step * node_offset - unit_step * (leaf_num <= 1 ? 0 : leaf_num / 2);
      qreal y = pt.y() + child_node_list[i]->get_delay() * 10;

      /// add line to item
      item = addPoint(item, connected_pt, QPointF(x, y));

      /// add line to item
      GuiSpeedupClockTreeItem* child_item = buildNodeItem(child_node_list[i], QPointF(x, y), unit_step);
    }
  }
}

GuiSpeedupClockTreeItem* IdbSpeedUpSetup::addPoint(GuiSpeedupClockTreeItem* item, QPointF pt, QPointF child_pt) {
  if (!item->has_capacity()) {
    item = _gui_design->get_clock_list()->add_item(item->get_type());
  }

  item->add_point(pt, child_pt);

  return item;
}

int64_t IdbSpeedUpSetup::get_nodelist_child_num(std::vector<iplf::CtsTreeNode*>& node_list) {
  int64_t num = 0;
  for (auto node : node_list) {
    int leaf_num = node->get_leaf_num();
    num += (leaf_num == 0 ? 1 : leaf_num);
  }

  return num;
}
