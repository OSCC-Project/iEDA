#include "tree.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace salt {

void TreeNode::PrintSingle(ostream& os) const
{
  os << "Node " << id << ": " << loc << (pin ? ", pin" : "") << ", " << children.size() << " children";
}

void TreeNode::PrintRecursiveHelp(ostream& os, vector<bool>& prefix) const
{
  for (auto pre : prefix)
    os << (pre ? "  |" : "   ");
  if (!prefix.empty())
    os << "-> ";
  PrintSingle(os);
  os << endl;
  if (children.size() > 0) {
    prefix.push_back(true);
    for (size_t i = 0; i < children.size() - 1; ++i) {
      if (children[i])
        children[i]->PrintRecursiveHelp(os, prefix);
      else
        os << "<null>" << endl;
    }
    prefix.back() = false;
    children.back()->PrintRecursiveHelp(os, prefix);
    prefix.pop_back();
  }
}

void TreeNode::PrintRecursive(ostream& os) const
{
  vector<bool> prefix;  // prefix indicates whether an ancestor is a last child or not
  PrintRecursiveHelp(os, prefix);
}

void TreeNode::setParent(const shared_ptr<TreeNode>& childNode, const shared_ptr<TreeNode>& parentNode)
{
  childNode->parent = parentNode;
  parentNode->children.push_back(childNode);
}

void TreeNode::resetParent(const shared_ptr<TreeNode>& node)
{
  assert(node->parent);

  auto& n = node->parent->children;
  auto it = find(n.begin(), n.end(), node);
  assert(it != n.end());
  *it = n.back();
  n.pop_back();

  node->parent.reset();
}

void TreeNode::reroot(const shared_ptr<TreeNode>& node)
{
  if (node->parent) {
    reroot(node->parent);
    auto old_parent = node->parent;
    TreeNode::resetParent(node);
    TreeNode::setParent(old_parent, node);
  }
}

bool TreeNode::isAncestor(const shared_ptr<TreeNode>& ancestor, const shared_ptr<TreeNode>& descendant)
{
  auto tmp = descendant;
  do {
    if (tmp == ancestor) {
      return true;
    }
    tmp = tmp->parent;
  } while (tmp);
  return false;
}

void TreeNode::preOrder(const shared_ptr<TreeNode>& node, const function<void(const shared_ptr<TreeNode>&)>& visit)
{
  visit(node);
  for (auto c : node->children)
    preOrder(c, visit);
}

void TreeNode::postOrder(const shared_ptr<TreeNode>& node, const function<void(const shared_ptr<TreeNode>&)>& visit)
{
  for (auto c : node->children)
    postOrder(c, visit);
  visit(node);
}

void TreeNode::postOrderCopy(const shared_ptr<TreeNode>& node, const function<void(const shared_ptr<TreeNode>&)>& visit)
{
  auto tmp = node->children;
  for (auto c : tmp)
    postOrderCopy(c, visit);
  visit(node);
}

void Tree::Reset(bool freeTreeNodes)
{
  if (freeTreeNodes) {
    postOrder([](const shared_ptr<TreeNode>& node) { node->children.clear(); });
  }
  source.reset();
  net = nullptr;
}

void Tree::Read(istream& is)
{
  Net* p_net = new Net;

  // header
  string buf, option;
  int num_pin = 0;
  while (is >> buf && buf != "Tree")
    ;
  assert(buf == "Tree");
  getline(is, buf);
  istringstream iss(buf);
  iss >> p_net->id >> p_net->name >> num_pin >> option;
  assert(num_pin > 0);
  p_net->with_cap = (option == "-cap");

  // pins
  int i, parent_idx;
  DTYPE x, y;
  double c = 0.0;
  p_net->pins.resize(num_pin);
  vector<int> parent_idxs;
  vector<shared_ptr<TreeNode>> tree_nodes;
  for (auto& pin : p_net->pins) {
    is >> i >> x >> y >> parent_idx;
    assert(i == tree_nodes.size());
    if (p_net->with_cap)
      is >> c;
    pin = make_shared<Pin>(x, y, i, c);
    tree_nodes.push_back(make_shared<TreeNode>(x, y, pin, i));
    parent_idxs.push_back(parent_idx);
  }
  assert(tree_nodes.size() == num_pin);

  // non-pin nodes
  getline(is, buf);  // consume eol
  streampos pos;
  while (true) {
    pos = is.tellg();
    getline(is, buf);
    istringstream iss2(buf);
    iss2 >> i >> x >> y >> parent_idx;
    if (iss2.fail())
      break;
    assert(i == tree_nodes.size());
    tree_nodes.push_back(make_shared<TreeNode>(x, y, nullptr, i));
    parent_idxs.push_back(parent_idx);
  }
  is.seekg(pos);  // go back

  // parents
  for (unsigned i = 0; i < tree_nodes.size(); ++i) {
    parent_idx = parent_idxs[i];
    if (parent_idx >= 0) {
      assert(parent_idx < tree_nodes.size());
      salt::TreeNode::setParent(tree_nodes[i], tree_nodes[parent_idx]);
    } else {
      assert(parent_idx == -1);
      source = tree_nodes[i];
    }
  }

  net = p_net;
  // TODO: check dangling nodes
}

void Tree::Read(const string& file_name)
{
  ifstream is(file_name);
  if (is.fail()) {
    cout << "ERROR: Cannot open file " << file_name << endl;
    exit(1);
  }
  Read(is);
}

void Tree::Write(ostream& os)
{
  // header
  os << "Tree " << net->GetHeader() << endl;

  // nodes
  // Note: source pin may not be covered in some intermediate state
  int num_nodes = UpdateId();
  auto nodes = ObtainNodes();
  vector<shared_ptr<TreeNode>> sorted_nodes(num_nodes, nullptr);  // bucket sort
  for (auto node : nodes) {
    sorted_nodes[node->id] = node;
  }
  for (auto node : sorted_nodes) {
    if (!node)
      continue;
    int parentId = node->parent ? node->parent->id : -1;
    os << node->id << " " << node->loc.x << " " << node->loc.y << " " << parentId;
    if (net->with_cap && node->pin)
      os << " " << node->pin->cap;
    os << endl;
  }
}

void Tree::Write(const string& prefix, bool with_net_info)
{
  ofstream ofs(prefix + (with_net_info ? ("_" + net->name) : "") + ".tree");
  Write(ofs);
  ofs.close();
}

void Tree::Print(ostream& os) const
{
  os << "Tree ";
  if (net)
    os << net->id << ": #pins=" << net->pins.size() << endl;
  else
    os << "<no_net_associated>" << endl;
  if (source)
    source->PrintRecursive(os);
  else
    os << "<null>" << endl;
}

int Tree::UpdateId()
{
  int num_node = net->pins.size();
  preOrder([&](const shared_ptr<TreeNode>& node) {
    if (node->pin) {
      assert(node->pin->id < net->pins.size());
      node->id = node->pin->id;
    } else {
      node->id = num_node++;
    }
  });
  return num_node;
}

vector<shared_ptr<TreeNode>> Tree::ObtainNodes() const
{
  vector<shared_ptr<TreeNode>> nodes;
  preOrder([&](const shared_ptr<TreeNode>& node) { nodes.push_back(node); });
  return nodes;
}

void Tree::SetParentFromChildren()
{
  preOrder([](const shared_ptr<TreeNode>& node) {
    for (auto& c : node->children) {
      c->parent = node;
    }
  });
}

void Tree::SetParentFromUndirectedAdjList()
{
  preOrder([](const shared_ptr<TreeNode>& node) {
    for (auto it = node->children.begin(); it != node->children.end();) {
      auto child = *it;
      if (!child->parent) {
        child->parent = node;
        ++it;
      } else {
        it = node->children.erase(it);
      }
      auto& sub_children = child->children;
      auto sub_it = find(sub_children.begin(), sub_children.end(), node);
      assert(sub_it != sub_children.end());
      *sub_it = sub_children.back();
      sub_children.pop_back();
    }
  });
}

void Tree::reroot()
{
  TreeNode::reroot(source);
}

void Tree::QuickCheck()
{
  int num_pin = net->pins.size(), num_checked = 0;
  vector<bool> pin_exist(num_pin, false);
  preOrder([&](const shared_ptr<TreeNode>& node) {
    if (!node) {
      cerr << "Error: empty node" << endl;
    }
    if (node->pin) {
      auto id = node->pin->id;
      if (!(id >= 0 && id < num_pin && pin_exist[id] == false)) {
        cerr << "Error: Steiner node with incorrect id" << endl;
      }
      pin_exist[id] = true;
      ++num_checked;
    }
    for (auto& c : node->children) {
      if (!c->parent || c->parent != node) {
        cerr << "Error: inconsistent parent-child relationship" << endl;
      }
    }
  });
  if (num_checked != num_pin) {
    cerr << "Error: pin not covered" << endl;
  }
}

void Tree::RemovePhyRedundantSteiner()
{
  postOrderCopy([](const shared_ptr<TreeNode>& node) {
    if (!node->parent || node->loc != node->parent->loc)
      return;
    if (node->pin) {
      if (node->parent->pin && node->parent->pin != node->pin)
        return;
      node->parent->pin = node->pin;
    }
    for (auto c : node->children)
      TreeNode::setParent(c, node->parent);
    TreeNode::resetParent(node);
  });
}

void Tree::RemoveTopoRedundantSteiner()
{
  postOrderCopy([](const shared_ptr<TreeNode>& node) {
    // degree may change after post-order traversal of its children
    if (node->pin)
      return;
    if (node->children.empty()) {
      TreeNode::resetParent(node);
    } else if (node->children.size() == 1) {
      auto old_parent = node->parent, oldChild = node->children[0];
      TreeNode::resetParent(node);
      TreeNode::resetParent(oldChild);
      TreeNode::setParent(oldChild, old_parent);
    }
  });
}

void Tree::RemoveEmptyChildren()
{
  preOrder([](const shared_ptr<TreeNode>& node) {
    int size = 0;
    for (int i = 0; i < node->children.size(); ++i) {
      if (node->children[i])
        node->children[size++] = node->children[i];
    }
    node->children.resize(size);
  });
}

}  // namespace salt