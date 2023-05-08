#ifndef IDRC_SRC_MODULE_CONNECTED_COMPONENT_H_
#define IDRC_SRC_MODULE_CONNECTED_COMPONENT_H_
#include "DrcConflictGraph.h"

namespace idrc {
class ConnectedComponentFinder
{
 public:
  ConnectedComponentFinder() {}
  ~ConnectedComponentFinder();

  std::vector<DrcConflictGraph*>& getAllConnectedComponentInGraph(DrcConflictGraph* conflict_graph);

 private:
  int _index = 0;
  std::map<DrcConflictNode*, int> _dfn;
  std::map<DrcConflictNode*, int> _low;
  std::set<DrcConflictNode*> _node_in_stack;
  std::vector<DrcConflictNode*> _temp_stack;
  std::vector<DrcConflictGraph*> _connected_component_list;

  void init();
  bool isInStack(DrcConflictNode* conflict_node);
  void Tarjan(DrcConflictNode* node);
  void storeConnectedComponent(DrcConflictNode* node);
  void subGraphPruning(std::vector<DrcConflictNode*>& sub_graph);
};
}  // namespace idrc

#endif
