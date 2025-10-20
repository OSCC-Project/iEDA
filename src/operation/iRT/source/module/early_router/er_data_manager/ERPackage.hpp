#pragma once

#include "ERPillar.hpp"
#include "TNode.hpp"

namespace irt {

class ERPackage
{
 public:
  ERPackage() = default;
  explicit ERPackage(TNode<ERPillar>* parent_pillar_node, TNode<ERPillar>* child_pillar_node)
  {
    _parent_pillar_node = parent_pillar_node;
    _child_pillar_node = child_pillar_node;
  }
  ~ERPackage() = default;
  // getter
  TNode<ERPillar>* get_parent_pillar_node() { return _parent_pillar_node; }
  TNode<ERPillar>* get_child_pillar_node() { return _child_pillar_node; }
  // const getter
  const TNode<ERPillar>* get_parent_pillar_node() const { return _parent_pillar_node; }
  const TNode<ERPillar>* get_child_pillar_node() const { return _child_pillar_node; }
  // setter
  void set_parent_pillar_node(TNode<ERPillar>* parent_pillar_node) { _parent_pillar_node = parent_pillar_node; }
  void set_child_pillar_node(TNode<ERPillar>* child_pillar_node) { _child_pillar_node = child_pillar_node; }
  // function
  ERPillar& getParentPillar() { return _parent_pillar_node->value(); }
  ERPillar& getChildPillar() { return _child_pillar_node->value(); }

 private:
  TNode<ERPillar>* _parent_pillar_node = nullptr;
  TNode<ERPillar>* _child_pillar_node = nullptr;
};

}  // namespace irt
