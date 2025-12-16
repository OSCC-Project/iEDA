#pragma once

#include <cmath>
#include <map>
#include <set>

#include "tree.h"

namespace salt {

class RSABase
{
 public:
  void ReplaceRootChildren(Tree& tree);
  virtual void Run(const Net& net, Tree& tree) = 0;

 protected:
  DTYPE MaxOvlp(DTYPE z1, DTYPE z2);
};

// Inner Node: an inner node alway has two child (outer) nodes
// tn->children[0]: smaller angle, clockwise, left
// tn->children[1]: larger angle, counter clockwise, right
class InnerNode
{
 public:
  unsigned id;
  shared_ptr<TreeNode> tn;
  DTYPE dist;
  double angle;  // [-pi, pi]
  InnerNode(const shared_ptr<TreeNode>& treeNode) : tn(treeNode), dist(abs(tn->loc.x) + abs(tn->loc.y)), angle(atan2(tn->loc.y, tn->loc.x))
  {
    static unsigned gid = 0;
    id = gid++;
  }  // use id instead of pointer to make it deterministic
};
class CompInnerNode
{
 public:
  bool operator()(const InnerNode* a, const InnerNode* b) const
  {
    return a->dist > b->dist ||  // prefer fathest one
           (a->dist == b->dist
            && (a->angle > b->angle ||                      // prefer single direction
                (a->angle == b->angle && a->id > b->id)));  // distinguish two nodes with same loc (not touched normally)
  }
};

// Outer Node: an outer node has one or two parent (inner) nodes
class OuterNode
{
 public:
  shared_ptr<TreeNode> cur;               // itself
  InnerNode *left_parent, *right_parent;  // left parent, right parent
  OuterNode(const shared_ptr<TreeNode>& c, InnerNode* l = nullptr, InnerNode* r = nullptr) : cur(c), left_parent(l), right_parent(r) {}
};

// RSA Builder
class RsaBuilder : public RSABase
{
 public:
  void Run(const Net& net, Tree& tree);

 private:
  // inner nodes
  set<InnerNode*, CompInnerNode> inner_nodes;

  // outer nodes
  map<double, OuterNode> outer_nodes;
  // the unique key is always guaranteed, better for query/find by key
  inline double OuterNodeKey(const shared_ptr<TreeNode>& tn) { return atan2(tn->loc.y, tn->loc.x); }
  // larger angle, counter clockwise, right
  map<double, OuterNode>::iterator NextOuterNode(const map<double, OuterNode>::iterator& it);
  // smaller angle, clockwise, left
  map<double, OuterNode>::iterator PrevOuterNode(const map<double, OuterNode>::iterator& it);

  // remove an outer node
  bool TryMaxOvlpSteinerNode(OuterNode& left, OuterNode& right);  // maximize the overlapping
  void RemoveAnOuterNode(const shared_ptr<TreeNode>& node, bool del_left = true, bool del_right = true);

  // add an outer node
  bool TryDominatingOneSide(OuterNode& p, OuterNode& c);
  void TryDominating(const shared_ptr<TreeNode>& node);
  void AddAnOuterNode(const shared_ptr<TreeNode>& node);

  // for debug
  void PrintInnerNodes();
  void PrintOuterNodes();
};

}  // namespace salt
