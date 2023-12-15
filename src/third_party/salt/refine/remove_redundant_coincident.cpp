#include <cassert>
#include <stack>

#include "refine.h"
namespace salt {

void Refine::removeRedundantCoincident(Tree& tree)
{
  std::stack<shared_ptr<TreeNode>> stack;
  auto source = tree.source;
  stack.push(source);
  while (!stack.empty()) {
    auto node = stack.top();
    stack.pop();
    auto children = node->children;
    bool exist_opt = false;
    for (auto child : children) {
      if (node->loc == child->loc) {
        exist_opt = true;
        break;
      }
    }
    if (!exist_opt) {
      for (auto child : children) {
        stack.push(child);
      }
      continue;
    }
    // move target child's children to node, and remove target child
    if (node->pin) {
      for (auto child : children) {
        if (node->loc != child->loc) {
          continue;
        }
        assert(!child->pin || !node->pin);
        TreeNode::resetParent(child);
        auto sub_children = child->children;
        for (auto sub_child : sub_children) {
          TreeNode::resetParent(sub_child);
          TreeNode::setParent(sub_child, node);
        }
      }
      stack.push(node);
      continue;
    }
    // let target child be new node
    for (auto child : children) {
      if (node->loc != child->loc) {
        continue;
      }
      auto parent = node->parent;
      TreeNode::resetParent(child);
      TreeNode::resetParent(node);
      TreeNode::setParent(child, parent);

      for (auto sub_child : children) {
        if (sub_child == child) {
          continue;
        }
        TreeNode::resetParent(sub_child);
        TreeNode::setParent(sub_child, child);
      }
      stack.push(child);
      node->children.clear();
      break;
    }
  }
}

}  // namespace salt