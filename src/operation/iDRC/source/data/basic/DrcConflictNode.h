#ifndef IDRC_SRC_DB_CONFLICTNODE_H_
#define IDRC_SRC_DB_CONFLICTNODE_H_

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <vector>

#include "DrcRect.h"

namespace idrc {
class DrcConflictNode
{
 public:
  DrcConflictNode() {}
  explicit DrcConflictNode(DrcRect* drc_rect) { _spot_rect = drc_rect; }
  explicit DrcConflictNode(DrcPolygon* drc_polygon) { _spot_polygon = drc_polygon; }
  explicit DrcConflictNode(int node_id) { _node_id = node_id; }
  ~DrcConflictNode() {}
  // add conflict node
  void addConflictNode(DrcConflictNode* conflict_node) { _conflict_node_list.push_back(conflict_node); }
  // setter
  void set_parent_node(DrcConflictNode* parent) { _parent_node = parent; }
  void erase_parent_node() { _parent_node = nullptr; }
  void set_node_id(int id) { _node_id = id; }
  void set_color(int color) { _color = color; }
  void erase_color() { _color = 0; }
  void set_conflict_node_list(std::vector<DrcConflictNode*>& conflict_node_list)
  {
    _conflict_node_list.clear();
    _conflict_node_list = conflict_node_list;
  }
  // getter
  std::vector<DrcConflictNode*>& get_conflict_node_list() { return _conflict_node_list; }
  int get_node_id() const { return _node_id; }
  DrcRect* get_rect() { return _spot_rect; }
  DrcPolygon* get_polygon() { return _spot_polygon; }
  DrcConflictNode* get_parent_node() { return _parent_node; }
  int get_color() const { return _color; }
  int get_conflict_node_num() { return _conflict_node_list.size(); }

 private:
  int _node_id = -1;
  int _color = 0;
  DrcConflictNode* _parent_node = nullptr;
  DrcRect* _spot_rect = nullptr;
  DrcPolygon* _spot_polygon = nullptr;
  std::vector<DrcConflictNode*> _conflict_node_list;
};

struct DrcConflictNodeCmp
{
  bool operator()(DrcConflictNode* a, DrcConflictNode* b) { return (a->get_conflict_node_num()) > (b->get_conflict_node_num()); }
};
}  // namespace idrc

#endif