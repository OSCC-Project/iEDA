#include "rsa.h"

#include <unordered_map>
#include <unordered_set>

namespace salt {

constexpr double kPI = 3.14159265358979323846; /* pi */

void RSABase::ReplaceRootChildren(Tree& tree)
{
  const Net* old_net = tree.net;

  // create tmp_net and fake_pins
  Net tmp_net = *old_net;
  tmp_net.pins.clear();
  unordered_map<shared_ptr<Pin>, shared_ptr<TreeNode>> pin_to_old_node;
  tmp_net.pins.push_back(tree.source->pin);
  unordered_set<shared_ptr<Pin>> fake_pins;
  pin_to_old_node[tree.source->pin] = tree.source;
  for (auto c : tree.source->children) {  // only contains the direct children of tree.source
    shared_ptr<Pin> pin = c->pin;
    if (!pin) {
      pin = make_shared<Pin>(c->loc);
      fake_pins.insert(pin);
    }
    tmp_net.pins.push_back(pin);
    pin_to_old_node[pin] = c;
  }
  tree.source->children.clear();  // free them...

  // get rsa and graft the old subtrees to it
  Run(tmp_net, tree);
  tree.postOrder([&](const shared_ptr<TreeNode>& node) {
    if (node->pin) {
      auto old_node = pin_to_old_node[node->pin];
      if (fake_pins.find(node->pin) != fake_pins.end())
        node->pin = nullptr;
      if (node->parent)
        for (auto c : old_node->children)
          TreeNode::setParent(c, node);
    }
  });
  tree.net = old_net;
  tree.RemoveTopoRedundantSteiner();
}

DTYPE RSABase::MaxOvlp(DTYPE z1, DTYPE z2)
{
  if (z1 >= 0 && z2 >= 0)
    return min(z1, z2);
  else if (z1 <= 0 && z2 <= 0)
    return max(z1, z2);
  else
    return 0;
}

void RsaBuilder::Run(const Net& net, Tree& tree)
{
  // Shift all pins to make source (0,0)
  auto ori_src_loc = net.source()->loc;
  for (auto& p : net.pins)
    p->loc -= ori_src_loc;

  // Init inner_nodes with all sinks
  for (auto p : net.pins)
    if (p->IsSink())
      inner_nodes.insert(new InnerNode(make_shared<TreeNode>(p)));

  // Process a inner node in each iteration
  while (!inner_nodes.empty()) {
    if ((*inner_nodes.begin())->dist == 0)
      break;  // TODO: clear
    shared_ptr<TreeNode> node = (*inner_nodes.begin())->tn;
    auto for_delete = *inner_nodes.begin();
    inner_nodes.erase(inner_nodes.begin());
    if (!node->pin) {  // steiner node
      assert(node->children.size() == 2);
      if (node->children[0])
        RemoveAnOuterNode(node->children[0], true, false);
      if (node->children[1])
        RemoveAnOuterNode(node->children[1], false, true);
      node->children[0]->parent = node;
      node->children[1]->parent = node;
    } else {  // pin node
      TryDominating(node);
    }
    delete for_delete;
    AddAnOuterNode(node);
  }

  // connet the remaining outer_nodes to the source
  tree.source = make_shared<TreeNode>(net.source());
  for (const auto& on : outer_nodes)
    TreeNode::setParent(on.second.cur, tree.source);
  tree.net = &net;

  // shift all pins back
  for (auto& p : net.pins)
    p->loc += ori_src_loc;
  tree.preOrder([&](const shared_ptr<TreeNode>& node) { node->loc += ori_src_loc; });

  // clear
  for (auto in : inner_nodes)
    delete in;
  inner_nodes.clear();
  outer_nodes.clear();
}

map<double, OuterNode>::iterator RsaBuilder::NextOuterNode(const map<double, OuterNode>::iterator& it)
{
  auto res = next(it);
  if (res != outer_nodes.end())
    return res;
  else
    return outer_nodes.begin();
}

map<double, OuterNode>::iterator RsaBuilder::PrevOuterNode(const map<double, OuterNode>::iterator& it)
{
  if (it != outer_nodes.begin())
    return prev(it);
  else
    return prev(outer_nodes.end());
}

bool RsaBuilder::TryMaxOvlpSteinerNode(OuterNode& left, OuterNode& right)
{
  double rl_ang = atan2(right.cur->loc.y, right.cur->loc.x) - atan2(left.cur->loc.y, left.cur->loc.x);
  // if (rl_ang>-kPI && rl_ang<0) return false; // there is smaller arc
  DTYPE new_x, new_y;
  if ((rl_ang > -kPI && rl_ang < 0) || rl_ang > kPI) {  // large arc
    new_x = 0;
    new_y = 0;
  } else {
    new_x = MaxOvlp(left.cur->loc.x, right.cur->loc.x);
    new_y = MaxOvlp(left.cur->loc.y, right.cur->loc.y);
  }
  // if (new_x==0 && new_y==0) return false; // non-neighboring quadrant
  auto tn = make_shared<TreeNode>(new_x, new_y);
  tn->children = {left.cur, right.cur};
  auto in = new InnerNode(tn);
  left.right_parent = in;
  right.left_parent = in;
  inner_nodes.insert(in);
  // cout << "add a tmp steiner point" << endl;
  // tn->Print(0,true);
  return true;
}

void RsaBuilder::RemoveAnOuterNode(const shared_ptr<TreeNode>& node, bool del_left, bool del_right)
{
  auto outer_cur = outer_nodes.find(OuterNodeKey(node));
  assert(outer_cur != outer_nodes.end());
  InnerNode *inner_left = outer_cur->second.left_parent, *inner_right = outer_cur->second.right_parent;
  auto outer_left = outer_nodes.end(), outer_right = outer_nodes.end();
  if (inner_left != nullptr) {
    outer_left = PrevOuterNode(outer_cur);
    assert(outer_left->second.cur == inner_left->tn->children[0]);
    assert(outer_cur->second.cur == inner_left->tn->children[1]);
    inner_nodes.erase(inner_left);
    if (del_left)
      delete inner_left;  // inner parent become invalid now
    outer_left->second.right_parent = nullptr;
  }
  if (inner_right != nullptr) {
    outer_right = NextOuterNode(outer_cur);
    assert(outer_cur->second.cur == inner_right->tn->children[0]);
    assert(outer_right->second.cur == inner_right->tn->children[1]);
    inner_nodes.erase(inner_right);
    if (del_right)
      delete inner_right;  // inner parent become invalid now
    outer_right->second.left_parent = nullptr;
  }
  // delete outer_cur->second.first; //  outer child should be kept
  outer_nodes.erase(outer_cur);
  if (del_left && del_right && outer_right != outer_nodes.end() && outer_left != outer_nodes.end() && outer_left != outer_right) {
    TryMaxOvlpSteinerNode(outer_left->second, outer_right->second);
  }
}

// p dominates c
inline bool Dominate(const Point& p, const Point& c)
{
  return ((p.x >= 0 && c.x >= 0 && p.x <= c.x) || (p.x <= 0 && c.x <= 0 && p.x >= c.x))      // x
         && ((p.y >= 0 && c.y >= 0 && p.y <= c.y) || (p.y <= 0 && c.y <= 0 && p.y >= c.y));  // y
}

bool RsaBuilder::TryDominatingOneSide(OuterNode& p, OuterNode& c)
{
  if (!Dominate(p.cur->loc, c.cur->loc))
    return false;
  TreeNode::setParent(c.cur, p.cur);
  RemoveAnOuterNode(c.cur);
  return true;
}

void RsaBuilder::TryDominating(const shared_ptr<TreeNode>& node)
{
  OuterNode outer_cur(node);
  if (outer_nodes.empty())
    return;
  else if (outer_nodes.size() == 1) {
    TryDominatingOneSide(outer_cur, outer_nodes.begin()->second);
    return;
  }
  // get outer_right & outer_left
  auto outer_right = outer_nodes.upper_bound(OuterNodeKey(node));
  if (outer_right == outer_nodes.end())
    outer_right = outer_nodes.begin();
  auto outer_left = PrevOuterNode(outer_right);
  assert(outer_left != outer_nodes.end() && outer_right != outer_nodes.end());
  assert(outer_left->second.right_parent == outer_right->second.left_parent);
  // try dominating twice
  TryDominatingOneSide(outer_cur, outer_left->second);
  TryDominatingOneSide(outer_cur, outer_right->second);
}

// suppose no case of [node = min(node, an outer node)]
void RsaBuilder::AddAnOuterNode(const shared_ptr<TreeNode>& node)
{
  OuterNode outer_cur(node);
  if (!outer_nodes.empty()) {
    // get outer_right & outer_left
    auto outer_right = outer_nodes.upper_bound(OuterNodeKey(node));
    if (outer_right == outer_nodes.end())
      outer_right = outer_nodes.begin();
    auto outer_left = PrevOuterNode(outer_right);
    assert(outer_left != outer_nodes.end() && outer_right != outer_nodes.end());
    assert(outer_left->second.right_parent == outer_right->second.left_parent);
    // delete parent(outer_right, outer_left)
    if (outer_left->second.right_parent) {
      inner_nodes.erase(outer_left->second.right_parent);
      delete outer_left->second.right_parent;  // inner parent become invalid now
    }
    // add two parents
    TryMaxOvlpSteinerNode(outer_left->second, outer_cur);
    TryMaxOvlpSteinerNode(outer_cur, outer_right->second);
  }
  outer_nodes.insert({OuterNodeKey(node), outer_cur});
}

void RsaBuilder::PrintInnerNodes()
{
  cout << "Inner nodes (# = " << inner_nodes.size() << ")" << endl;
  for (auto in : inner_nodes) {
    cout << in->tn;
  }
}

void RsaBuilder::PrintOuterNodes()
{
  cout << "Outer nodes (# = " << outer_nodes.size() << ")" << endl;
  for (auto on : outer_nodes) {
    cout << on.second.cur;
  }
}

}  // namespace salt