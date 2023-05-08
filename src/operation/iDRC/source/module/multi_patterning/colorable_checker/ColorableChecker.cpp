#include "ColorableChecker.h"
namespace idrc {

std::vector<DrcConflictNode*>& ColorableChecker::colorable_check(const std::vector<DrcConflictGraph*>& sub_graph_list)
{
  int cnt = 0;
  for (auto sub_graph : sub_graph_list) {
    ++cnt;
    std::cout << "check sub_graph :" << cnt << " of " << sub_graph_list.size() << " "
              << "node num" << sub_graph->get_node_num() << std::endl;

    _visited.clear();

    _origin_subgraph_node_list.clear();
    // std::vector<DrcConflictNode*> subgraph_node_list = sub_graph->get_conflict_node_list();
    // _origin_subgraph_node_list.insert(subgraph_node_list.begin(), subgraph_node_list.end());

    _temp_uncolorable_node_list.clear();
    // _node_to_color.clear();
    _colorable_node_num = 0;
    _fewest_uncolorable_node_num = INT_MAX;
    _graph_node_num = sub_graph->get_node_num();

    _uncolorable_node_num = 0;
    _record_uncolorable_num = false;
    // colorable_check(sub_graph);
    colorable_check_new(sub_graph);
  }
  return _uncolorable_node_list;
}

void ColorableChecker::addRecordsOfUncolorableNode()
{
  _uncolorable_node_list.insert(_uncolorable_node_list.end(), _temp_uncolorable_node_list.begin(), _temp_uncolorable_node_list.end());
}

// void ColorableChecker::colorable_check(DrcConflictGraph* sub_graph)
// {
//   std::vector<DrcConflictNode*> node_list = sub_graph->get_conflict_node_list();
//   std::sort(node_list.begin(), node_list.end(), DrcConflictNodeCmp());
//   DrcConflictNode* grph_node = node_list.front();
//   bool isAllNodeColorable = dfs(grph_node);
//   //必须满足两者!!一次深搜可能不能遍历所有点，找到可染色方案_temp_uncolorable_node_list可能没清零
//   if (!isAllNodeColorable && _temp_uncolorable_node_list.size() != 0) {
//     addRecordsOfUncolorableNode();
//     std::cout << " find uncolorable node" << std::endl;
//   }
// }

bool ColorableChecker::judgeIsColorable(DrcConflictNode* node, int color)
{
  for (auto conflict_node : node->get_conflict_node_list()) {
    // if (_origin_subgraph_node_list.find(conflict_node) == _origin_subgraph_node_list.end()) {
    //   continue;
    // }
    if (conflict_node->get_color() == color) {
      return false;
    }
  }
  return true;
}

void ColorableChecker::storeUncolorableNode()
{
  int uncolorable_node_cnt = 0;
  std::vector<DrcConflictNode*> temp_node_list;
  for (auto node : _visited) {
    if (node->get_color() == 0) {
      ++uncolorable_node_cnt;
      temp_node_list.push_back(node);
    }
  }
  if (_fewest_uncolorable_node_num > uncolorable_node_cnt) {
    _fewest_uncolorable_node_num = uncolorable_node_cnt;
    _temp_uncolorable_node_list.clear();
    _temp_uncolorable_node_list = temp_node_list;
  }
}

bool ColorableChecker::dfs(DrcConflictNode* node)
{
  bool isAllNodeColorable = false;
  // _node_to_color[node] = 0;
  _visited.insert(node);
  for (int color = 1; color <= _optional_color_num; ++color) {
    // _node_to_color[node] = 0;
    if (judgeIsColorable(node, color)) {
      node->set_color(color);
      ++_colorable_node_num;

      if (_colorable_node_num == _graph_node_num) {
        return true;
      }
    }

    // No color is available
    if (node->get_color() == 0) {
      ++_uncolorable_node_num;
      if (_uncolorable_node_num >= _fewest_uncolorable_node_num && _record_uncolorable_num == true) {
        // skip
        --_uncolorable_node_num;
        continue;
      }
    }

    int visited_node_num = _visited.size();
    if (visited_node_num == _graph_node_num) {
      // Identifies that logging has started
      _record_uncolorable_num = true;
      storeUncolorableNode();
    }

    std::vector<DrcConflictNode*> conflict_node_list = node->get_conflict_node_list();
    std::sort(conflict_node_list.begin(), conflict_node_list.end(), DrcConflictNodeCmp());
    for (auto conflict_node : conflict_node_list) {
      // if (_origin_subgraph_node_list.find(conflict_node) == _origin_subgraph_node_list.end()) {
      //   continue;
      // }
      if (_visited.find(conflict_node) == _visited.end()) {
        isAllNodeColorable = dfs(conflict_node);
        if (isAllNodeColorable) {
          return isAllNodeColorable;
        }
      }
    }

    // _node_to_color[node] = color;
    if (node->get_color() != 0) {
      node->erase_color();
      --_colorable_node_num;
    } else if (node->get_color() == 0) {
      --_uncolorable_node_num;
    }
  }

  _visited.erase(node);
  return isAllNodeColorable;
}

///////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void ColorableChecker::colorable_check_new(DrcConflictGraph* sub_graph)
{
  std::vector<DrcConflictNode*> node_list = sub_graph->get_conflict_node_list();
  std::sort(node_list.begin(), node_list.end(), DrcConflictNodeCmp());
  _origin_subgraph_node_list = node_list;
  bool isAllNodeColorable = DFS(0);
  //必须满足两者!!一次深搜可能不能遍历所有点，找到可染色方案_temp_uncolorable_node_list可能没清零
  if (!isAllNodeColorable && _temp_uncolorable_node_list.size() != 0) {
    addRecordsOfUncolorableNode();
    std::cout << " find uncolorable node" << std::endl;
  }
}
bool ColorableChecker::DFS(int i)
{
  bool isAllNodeColorable = false;
  bool isColored = false;
  DrcConflictNode* node = _origin_subgraph_node_list[i];
  _visited.insert(node);
  for (int color = 1; color <= _optional_color_num; ++color) {
    if (judgeIsColorable(node, color)) {
      isColored = true;
      node->set_color(color);
      ++_colorable_node_num;

      if (_colorable_node_num == _graph_node_num) {
        return true;
      } else if ((i + 1) <= (_graph_node_num - 1)) {
        i = i + 1;
        isAllNodeColorable = DFS(i);
        if (isAllNodeColorable) {
          return isAllNodeColorable;
        }
      }
      node->erase_color();
      --_colorable_node_num;
    }
  }

  // No color is available
  if (!isColored) {
    node->set_color(0);
    ++_uncolorable_node_num;
    if (!(_uncolorable_node_num >= _fewest_uncolorable_node_num && _record_uncolorable_num == true)) {
      // skip
      if ((i + 1) <= (_graph_node_num - 1)) {
        i = i + 1;
        DFS(i);
      }
    }
    --_uncolorable_node_num;
  }

  int visited_node_num = _visited.size();
  if (visited_node_num == _graph_node_num) {
    // Identifies that logging has started
    _record_uncolorable_num = true;
    storeUncolorableNode();
  }

  _visited.erase(node);
  return isAllNodeColorable;
}

}  // namespace idrc