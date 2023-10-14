#pragma once

#include "base/tree.h"

namespace salt {

class SaltInterface
{
 protected:
  vector<shared_ptr<TreeNode>> sl_nodes;  // nodes of the shallow-light tree
  vector<DTYPE> shortest_dists;
  vector<DTYPE> cur_dists;
  shared_ptr<TreeNode> sl_src;  // source node of the shallow-light tree

  void Init(Tree& min_tree, shared_ptr<Pin> src_pin);  // tree of minimum weight
  void Finalize(const Net& net, Tree& tree);
  virtual bool Relax(const shared_ptr<TreeNode>& u, const shared_ptr<TreeNode>& v) = 0;  // from u to v
  virtual void DFS(const shared_ptr<TreeNode>& mst_node, const shared_ptr<TreeNode>& sl_node, double eps) = 0;
};

class SaltBuilder : public SaltInterface
{
 public:
  void Run(const Net& net, Tree& tree, double eps, int refineLevel = 3);

 protected:
  bool Relax(const shared_ptr<TreeNode>& u, const shared_ptr<TreeNode>& v);  // from u to v
  void DFS(const shared_ptr<TreeNode>& mst_node, const shared_ptr<TreeNode>& sl_node, double eps);
};

}  // namespace salt