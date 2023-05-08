#pragma once

#include <utility>
#include <vector>

namespace icts {
using std::make_pair;
using std::pair;
using std::vector;

// Binary tree node base part
struct BTNodeBase {
  typedef BTNodeBase *base_ptr;

  base_ptr _parent;
  base_ptr _left;
  base_ptr _right;

  BTNodeBase() : _parent(nullptr), _left(nullptr), _right(nullptr) {}

  static base_ptr minimun(base_ptr x) {
    while (x->_left) {
      x = x->_left;
    }
    return x;
  }

  static base_ptr maximum(base_ptr x) {
    while (x->_right) {
      x = x->_right;
    }
    return x;
  }
};

// Binary tree base iterator
struct BTBaseIterator {
  typedef BTNodeBase *base_ptr;

  void increment() {
    if (_node->_right != nullptr) {
      _node = _node->_right;
      while (_node->_left != nullptr) {
        _node = _node->_left;
      }
    } else {
      base_ptr y = _node->_parent;
      while (_node == y->_right) {
        _node = y;
        y = y->_parent;
      }
      if (_node->_right != y) {  // the right subtree is null of root node.
        _node = y;
      }
    }
  }

  base_ptr _node;
};

// Binary tree node
template <typename Value>
struct BTNode : public BTNodeBase {
  typedef Value value_type;
  typedef BTNode<value_type> *link_type;

  BTNode() = default;
  BTNode(const value_type &value) : _value(value) {}
  ~BTNode() = default;

  link_type left() { return (link_type)_left; }
  link_type right() { return (link_type)_right; }
  link_type parent() { return (link_type)_parent; }
  value_type value() { return _value; }

  value_type _value;
};

// Binary tree iterator
template <typename Value>
struct BTIterator : public BTBaseIterator {
  typedef Value value_type;
  typedef value_type *pointer;
  typedef value_type &reference;
  typedef BTIterator<Value> iterator;
  typedef BTNode<Value> *link_type;

  BTIterator() {}
  BTIterator(link_type x) { _node = x; }
  BTIterator(const iterator &iter) { _node = iter._node; }

  reference operator*() const { return ((link_type)_node)->_value; }
  pointer operator->() const { return &(operator*()); }
  iterator &operator++();
  iterator operator++(int);

  bool operator==(const iterator &x) const { return _node == x._node; }
  bool operator!=(const iterator &x) const { return _node != x._node; }
};

// Binary tree
template <typename Value>
class BinaryTree {
 protected:
  typedef Value value_type;
  typedef value_type *pointer;
  typedef value_type &reference;

  typedef BTNodeBase *base_ptr;
  typedef BTNode<value_type> binary_tree_node;
  typedef BTNode<value_type> *link_type;

 public:
  typedef BTIterator<value_type> iterator;

  BinaryTree() : _node_count(0) { init(); }
  BinaryTree(value_type root_value);
  BinaryTree(const BinaryTree &btree);
  ~BinaryTree() {
    clear();
    putNode(_header);
  }

  iterator begin() const { return leftmost(); }
  iterator end() const { return _header; }
  bool empty() const { return _node_count == 0; }
  size_t size() const { return _node_count; }
  size_t max_size() const { return size_t(-1); }
  void clear();

  link_type &root() const { return (link_type &)_header->_parent; }
  value_type root_value() const { return value((link_type)_header->_parent); }

  bool replace(BinaryTree &rhs);

  bool is_root(const iterator &iter) const { return iter->_parent == _header; }
  bool is_leaf(const iterator &iter) const;
  vector<pair<value_type, value_type>> get_edges() const;

  static BinaryTree *merge(value_type root_value, BinaryTree *left_subtree,
                           BinaryTree *right_subtree);
  static BinaryTree merge(value_type root_value, BinaryTree &left_subtree,
                          BinaryTree &right_subtree);

  BinaryTree &operator=(const BinaryTree &x);
  bool operator==(const BinaryTree &x) const;

 protected:
  link_type get_node() { return new binary_tree_node(); }
  void putNode(link_type p) { delete p; }
  link_type createNode(const value_type &x) { return new binary_tree_node(x); }
  void destroyNode(link_type p);
  link_type clone_node(link_type x);

  link_type &leftmost() const { return (link_type &)_header->_left; }
  link_type &rightmost() const { return (link_type &)_header->_right; }
  static link_type &left(link_type x) { return (link_type &)x->_left; }
  static link_type &right(link_type x) { return (link_type &)x->_right; }
  static link_type &parent(link_type x) { return (link_type &)x->_parent; }
  static reference value(link_type x) { return x->_value; }
  static link_type minimum(link_type x) {
    return (link_type)BTNodeBase::minimun(x);
  }
  static link_type maximum(link_type x) {
    return (link_type)BTNodeBase::maximum(x);
  }
  static int nodeNum(link_type node);

 private:
  void init();
  void init_header();
  void erase(link_type x);
  link_type copy(link_type x, link_type parent);
  link_type copy(link_type x);
  bool equal(link_type x, link_type y) const;
  void build_edges(link_type node,
                   vector<pair<value_type, value_type>> &edges) const;

 private:
  link_type _header;   // the parent pointer to root node
  size_t _node_count;  // node number of binary tree
};

// Binary tree iterator
template <typename Value>
BTIterator<Value> &BTIterator<Value>::operator++() {
  increment();
  return *this;
}

template <typename Value>
BTIterator<Value> BTIterator<Value>::operator++(int) {
  iterator tmp = *this;
  increment();
  return tmp;
}

// Binary tree part
template <typename Value>
BinaryTree<Value>::BinaryTree(value_type root_value) {
  init();
  root() = createNode(root_value);
  leftmost() = root();
  rightmost() = _header;
  parent(root()) = _header;
  _node_count = 1;
}

template <typename Value>
BinaryTree<Value>::BinaryTree(const BinaryTree &btree) {
  _header = get_node();
  root() = copy(btree.root(), _header);
  leftmost() = minimum(root());
  rightmost() = maximum(root());
  parent(root()) = _header;
  _node_count = btree._node_count;
}

template <typename Value>
inline void BinaryTree<Value>::init() {
  _header = get_node();
  init_header();
}

template <typename Value>
void BinaryTree<Value>::init_header() {
  root() = nullptr;
  leftmost() = _header;
  rightmost() = _header;
  _node_count = 0;
}

template <typename Value>
void BinaryTree<Value>::clear() {
  if (_node_count != 0) {
    erase(root());
    root() = nullptr;
    leftmost() = _header;
    rightmost() = _header;
    _node_count = 0;
  }
}

template <typename Value>
bool BinaryTree<Value>::replace(BinaryTree &rhs) {
  if (rhs.root_value() != root_value()) {
    return false;
  }
  auto *cur_root = root();
  auto *rhs_root = rhs.root();

  cur_root->_left = rhs_root()->_left;
  cur_root->_right = rhs_root()->_right;
  leftmost() = rhs.leftmost();
  rightmost() = rhs.rightmost();

  rhs_root->_left = nullptr;
  rhs_root->_right = nullptr;
  rhs.leftmost() = rhs_root;
  rhs.rightmost() = rhs_root;

  return true;
}

template <typename Value>
int BinaryTree<Value>::nodeNum(link_type node) {
  return nodeNum(left(node)) + nodeNum(right(node)) + 1;
}

template <typename Value>
inline bool BinaryTree<Value>::is_leaf(const iterator &iter) const {
  base_ptr node = iter._node;
  return node->_left == nullptr && node->_right == nullptr;
}

template <typename Value>
BinaryTree<Value> *BinaryTree<Value>::merge(value_type root_value,
                                            BinaryTree *left_subtree,
                                            BinaryTree *right_subtree) {
  // build a new tree
  BinaryTree *tree = new BinaryTree(root_value);
  link_type root = tree->root();
  tree->_node_count =
      tree->_node_count + left_subtree->size() + right_subtree->size();

  // connect two subtree to root
  left(root) = left_subtree->root();
  if (left_subtree->root() != nullptr) {
    parent(left_subtree->root()) = root;
    tree->leftmost() = left_subtree->leftmost();
  }
  right(root) = right_subtree->root();
  if (right_subtree->root() != nullptr) {
    parent(right_subtree->root()) = root;
    tree->rightmost() = right_subtree->rightmost();
  }

  // free subtree header node
  left_subtree->init_header();
  right_subtree->init_header();
  delete left_subtree;
  delete right_subtree;

  return tree;
}

template <typename Value>
BinaryTree<Value> BinaryTree<Value>::merge(value_type root_value,
                                           BinaryTree &left_subtree,
                                           BinaryTree &right_subtree) {
  // build a new tree
  BinaryTree tree(root_value);
  link_type root = tree.root();
  tree._node_count =
      tree._node_count + left_subtree.size() + right_subtree.size();

  // connect two subtree to root
  left(root) = left_subtree.root();
  if (left_subtree.root() != nullptr) {
    parent(left_subtree.root()) = root;
    tree.leftmost() = left_subtree.leftmost();
  }
  right(root) = right_subtree.root();
  if (right_subtree.root() != nullptr) {
    parent(right_subtree.root()) = root;
    tree.rightmost() = right_subtree.rightmost();
  }

  // free subtree header node
  left_subtree.init_header();
  right_subtree.init_header();

  return tree;
}

template <typename Value>
vector<pair<Value, Value>> BinaryTree<Value>::get_edges() const {
  vector<pair<value_type, value_type>> edges;
  build_edges(root(), edges);
  return edges;
}

template <typename Value>
void BinaryTree<Value>::build_edges(
    link_type node, vector<pair<value_type, value_type>> &edges) const {
  if (node != nullptr) {
    // add edge if subtree is not null
    if (left(node) != nullptr) {
      edges.push_back(make_pair(node->_value, left(node)->_value));
    }
    if (right(node) != nullptr) {
      edges.push_back(make_pair(node->_value, right(node)->_value));
    }

    // recursive to solve the original question
    build_edges(left(node), edges);
    build_edges(right(node), edges);
  }
}

template <typename Value>
void BinaryTree<Value>::erase(link_type x) {
  if (x != nullptr) {
    erase(left(x));
    erase(right(x));
    destroyNode(x);
  }
}

template <typename Value>
BinaryTree<Value> &BinaryTree<Value>::operator=(const BinaryTree &btree) {
  if (this != &btree) {
    clear();
    if (btree.root() != nullptr) {
      root() = copy(btree.root(), _header);
      leftmost() = minimum(root());
      rightmost() = maximum(root());
      parent(root()) = _header;
      _node_count = btree._node_count;
    }
  }
  return *this;
}

template <typename Value>
bool BinaryTree<Value>::operator==(const BinaryTree &x) const {
  if (root() == x.root()) {
    return true;
  }
  return equal(root(), x.root());
}

template <typename Value>
bool BinaryTree<Value>::equal(link_type x, link_type y) const {
  if (!x && !y) {
    return true;
  }
  if ((!x && y) || (x && !y)) {
    return false;
  }
  return value(x) == value(y) && equal(left(x), left(y)) &&
         equal(right(x), right(y));
}

// suitable for left and right pointer
template <typename Value>
BTNode<Value> *BinaryTree<Value>::copy(link_type x) {
  if (!x) {
    return nullptr;
  }
  link_type node = clone_node(x);
  left(node) = copy(left(x));
  right(node) = copy(right(x));

  return node;
}

// suitable for left, right and parent ponter
template <typename Value>
BTNode<Value> *BinaryTree<Value>::copy(link_type x, link_type parent) {
  link_type child_root = clone_node(x);
  child_root->_parent = parent;

  if (x->_right) {
    child_root->_right = copy(right(x), child_root);
  }
  parent = child_root;
  x = left(x);

  while (x != nullptr) {
    link_type y = clone_node(x);
    parent->_left = y;
    y->_parent = parent;
    if (x->_right) {
      y->_right = copy(right(x), y);
    }
    parent = y;
    x = left(x);
  }

  return child_root;
}

template <typename Value>
void BinaryTree<Value>::destroyNode(link_type p) {
  // p->~decltype (&p->_value)();
  putNode(p);
}

template <typename Value>
BTNode<Value> *BinaryTree<Value>::clone_node(link_type x) {
  link_type tmp = createNode(x->_value);
  tmp->_left = nullptr;
  tmp->_right = nullptr;
  tmp->_parent = nullptr;
  return tmp;
}

}  // namespace icts
