#pragma once

#include "LAPillar.hpp"
#include "TNode.hpp"

namespace irt {

class LAPackage
{
 public:
  LAPackage() = default;
  explicit LAPackage(TNode<LAPillar>* parent_pillar_node, TNode<LAPillar>* child_pillar_node)
  {
    _parent_pillar_node = parent_pillar_node;
    _child_pillar_node = child_pillar_node;
  }
  ~LAPackage() = default;
  // getter
  TNode<LAPillar>* get_parent_pillar_node() { return _parent_pillar_node; }
  TNode<LAPillar>* get_child_pillar_node() { return _child_pillar_node; }
  // const getter
  const TNode<LAPillar>* get_parent_pillar_node() const { return _parent_pillar_node; }
  const TNode<LAPillar>* get_child_pillar_node() const { return _child_pillar_node; }
  // setter
  void set_parent_pillar_node(TNode<LAPillar>* parent_pillar_node) { _parent_pillar_node = parent_pillar_node; }
  void set_child_pillar_node(TNode<LAPillar>* child_pillar_node) { _child_pillar_node = child_pillar_node; }
  // function
  LAPillar& getParentPillar() { return _parent_pillar_node->value(); }
  LAPillar& getChildPillar() { return _child_pillar_node->value(); }

 private:
  TNode<LAPillar>* _parent_pillar_node = nullptr;
  TNode<LAPillar>* _child_pillar_node = nullptr;
};

}  // namespace irt
