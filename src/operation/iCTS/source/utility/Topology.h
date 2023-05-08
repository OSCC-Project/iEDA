#pragma once
#include <utility>
#include <vector>

namespace icts {

template <typename T>
class VertexIterator;
template <typename T>
class PreorderVertexIterator;
template <typename T>
class PostorderVertexIterator;
template <typename T>
class EdgeIterator;

typedef int vertex_type;

template <typename T>
struct TopoNode {
  typedef T data_type;

  vertex_type left() const { return _left; }
  vertex_type right() const { return _right; }
  vertex_type parent() const { return _parent; }
  data_type& data() { return _data; }
  void left(vertex_type vertex) { _left = vertex; }
  void right(vertex_type vertex) { _right = vertex; }
  void parent(vertex_type vertex) { _parent = vertex; }
  int sub_wire_length() const { return _sub_wire_length; }
  data_type _data;
  vertex_type _parent;
  vertex_type _left;
  vertex_type _right;
  int _sub_wire_length;
};

template <typename T>
class Topology {
 public:
  typedef T value_type;
  typedef VertexIterator<T> vertex_iterator;
  typedef PreorderVertexIterator<T> preorder_vertex_iterator;
  typedef PostorderVertexIterator<T> postorder_vertex_iterator;
  typedef EdgeIterator<T> edge_iterator;

  friend class VertexIterator<T>;
  friend class PreorderVertexIterator<T>;
  friend class PostorderVertexIterator<T>;

  Topology() = default;
  Topology(const std::vector<TopoNode<T>>& nodes, vertex_type root)
      : _nodes(nodes), _root(root) {}
  Topology(const Topology&) = default;
  ~Topology() = default;

  value_type& value(vertex_type vertex) { return _nodes[vertex]._data; }
  TopoNode<T>& node(vertex_type vertex) { return _nodes[vertex]; }
  vertex_type root() const { return _root; }
  std::vector<TopoNode<T>>& nodes() { return _nodes; }
  size_t size() const { return _nodes.size(); }
  void clear() {
    _root = -1;
    _nodes.clear();
  }

  std::pair<preorder_vertex_iterator, preorder_vertex_iterator>
  preorder_vertexs();

  std::pair<postorder_vertex_iterator, postorder_vertex_iterator>
  postorder_vertexs();

  std::pair<edge_iterator, edge_iterator> edges();

  template <typename P, typename Assign>
  void copy(Topology<P>& topo, Assign assign) {
    _root = topo.root();
    for (auto topo_node : topo.nodes()) {
      T val;
      assign(val, topo_node.data());
      TopoNode<T> node{val, topo_node.parent(), topo_node.left(),
                       topo_node.right()};
      _nodes.push_back(node);
    }
  }

 private:
  vertex_type leftmost(vertex_type vertex) const {
    vertex_type left = -1;
    while ((left = _nodes[vertex]._left) != -1) {
      vertex = left;
    }
    return vertex;
  }

 private:
  std::vector<TopoNode<T>> _nodes;
  vertex_type _root;
};

template <typename T>
class VertexIterator {
 public:
  typedef T value_type;
  typedef value_type& reference;
  typedef value_type* pointer;
  typedef VertexIterator Self;

  VertexIterator(std::vector<TopoNode<T>>& nodes)
      : _nodes(nodes), _vertex(-1) {}
  VertexIterator(std::vector<TopoNode<T>>& nodes, vertex_type vertex)
      : _nodes(nodes), _vertex(vertex) {}
  void cut() {
    _nodes[_vertex].left(-1);
    _nodes[_vertex].right(-1);
  }
  vertex_type get_vertex() const { return _vertex; }
  void set_vertex(vertex_type vertex) { _vertex = vertex; }
  value_type& value() { return _nodes[_vertex].data(); }
  value_type& value(vertex_type vertex) { return _nodes[vertex].data(); }
  TopoNode<T>& node(vertex_type vertex) { return _nodes[vertex]; }
  vertex_type left_vertex(vertex_type vertex) const {
    return _nodes[vertex].left();
  }
  vertex_type right_vertex(vertex_type vertex) const {
    return _nodes[vertex].right();
  }
  vertex_type parent_vertex(vertex_type vertex) const {
    return _nodes[vertex].parent();
  }
  vertex_type leftmost(vertex_type vertex) const {
    vertex_type lv = -1;
    while (!this->is_null(lv = left_vertex(vertex))) {
      vertex = lv;
    }
    return vertex;
  }

  bool is_null(vertex_type vertex) const { return vertex == -1; }
  bool is_leaf() const {
    return is_null(left_vertex(_vertex)) && is_null(right_vertex(_vertex));
  }
  bool is_root() const { return is_null(parent_vertex(_vertex)); }

 private:
  std::vector<TopoNode<T>>& _nodes;
  vertex_type _vertex;
};

template <typename T>
class PreorderVertexIterator : public VertexIterator<T> {
 public:
  typedef typename VertexIterator<T>::reference reference;
  typedef typename VertexIterator<T>::pointer pointer;
  typedef PreorderVertexIterator Self;

  PreorderVertexIterator(std::vector<TopoNode<T>>& nodes)
      : VertexIterator<T>(nodes) {}
  PreorderVertexIterator(std::vector<TopoNode<T>>& nodes, vertex_type vertex)
      : VertexIterator<T>(nodes, vertex) {}

  Self left() const {
    Self itr = *this;
    auto vertex = this->left_vertex(this->get_vertex());
    itr.set_vertex(vertex);
    return itr;
  }
  Self right() const {
    Self itr = *this;
    auto vertex = this->right_vertex(this->get_vertex());
    itr.set_vertex(vertex);
    return itr;
  }
  Self parent() const {
    Self itr = *this;
    auto vertex = this->parent_vertex(this->get_vertex());
    itr.set_vertex(vertex);
    return itr;
  }

  reference operator*() { return this->value(this->get_vertex()); }
  pointer operator->() { return &(operator*()); }
  Self& operator++() {
    vertex_type cur_vertex = this->get_vertex();
    if (this->is_null(cur_vertex)) {
      return *this;
    }

    auto lv = this->left_vertex(cur_vertex);
    if (!this->is_null(lv)) {
      this->set_vertex(lv);
      return *this;
    }
    auto rv = this->right_vertex(cur_vertex);
    if (!this->is_null(rv)) {
      this->set_vertex(rv);
      return *this;
    }

    vertex_type pv = this->parent_vertex(cur_vertex);
    while (!this->is_null(pv) && this->right_vertex(pv) == cur_vertex) {
      cur_vertex = pv;
      pv = this->parent_vertex(cur_vertex);
    }
    cur_vertex = this->is_null(pv) ? -1 : this->right_vertex(pv);
    this->set_vertex(cur_vertex);
    return *this;
  }
  Self operator++(int) {
    Self tmp = *this;
    operator++();
    return tmp;
  }
  bool operator==(const Self& x) const {
    return this->get_vertex() == x.get_vertex();
  }
  bool operator!=(const Self& x) const { return !operator==(x); }
};

template <typename T>
class PostorderVertexIterator : public VertexIterator<T> {
 public:
  typedef typename VertexIterator<T>::reference reference;
  typedef typename VertexIterator<T>::pointer pointer;
  typedef PostorderVertexIterator Self;

  PostorderVertexIterator(std::vector<TopoNode<T>>& nodes)
      : VertexIterator<T>(nodes) {}
  PostorderVertexIterator(std::vector<TopoNode<T>>& nodes, vertex_type vertex)
      : VertexIterator<T>(nodes, vertex) {}

  Self left() const {
    Self itr = *this;
    auto vertex = this->left_vertex(this->get_vertex());
    itr.set_vertex(vertex);
    return itr;
  }
  Self right() const {
    Self itr = *this;
    auto vertex = this->right_vertex(this->get_vertex());
    itr.set_vertex(vertex);
    return itr;
  }
  Self parent() const {
    Self itr = *this;
    auto vertex = this->parent_vertex(this->get_vertex());
    itr.set_vertex(vertex);
    return itr;
  }
  reference operator*() { return this->value(this->get_vertex()); }
  pointer operator->() { return &(operator*()); }
  Self& operator++() {
    vertex_type cur_vertex = this->get_vertex();
    if (this->is_null(cur_vertex)) {
      return *this;
    }
    auto parent_vertex = this->parent_vertex(cur_vertex);
    if (this->is_null(parent_vertex)) {
      this->set_vertex(-1);
      return *this;
    }
    auto right_vertex = this->right_vertex(parent_vertex);
    if (this->is_null(right_vertex) || right_vertex == cur_vertex) {
      this->set_vertex(parent_vertex);
      return *this;
    }

    cur_vertex = first_vertex(right_vertex);
    this->set_vertex(cur_vertex);
    return *this;
  }
  Self operator++(int) {
    Self tmp = *this;
    operator++();
    return tmp;
  }
  vertex_type first_vertex(vertex_type vertex) {
    vertex_type ans_vertex = vertex;
    while (!this->is_null(vertex)) {
      if (!this->is_null(this->left_vertex(vertex))) {
        vertex = this->left_vertex(vertex);
      } else if (!this->is_null(this->right_vertex(vertex))) {
        vertex = this->right_vertex(vertex);
      } else {
        ans_vertex = vertex;
        break;
      }
    }
    return ans_vertex;
  }

  bool operator==(const Self& x) const {
    return this->get_vertex() == x.get_vertex();
  }
  bool operator!=(const Self& x) const { return !operator==(x); }
};

template <typename T>
class EdgeIterator : public PostorderVertexIterator<T> {
 public:
  typedef EdgeIterator Self;

  typedef std::pair<T, T> value_type;
  typedef value_type* pointer;
  typedef value_type& reference;

  EdgeIterator(std::vector<TopoNode<T>>& nodes)
      : PostorderVertexIterator<T>(nodes) {}
  EdgeIterator(std::vector<TopoNode<T>>& nodes, vertex_type vertex)
      : PostorderVertexIterator<T>(nodes, vertex) {}

  value_type operator*() {
    auto cur_vertex = this->get_vertex();
    auto parent_vertex = this->parent_vertex(cur_vertex);
    return std::make_pair(this->value(parent_vertex), this->value(cur_vertex));
  }
  pointer operator->() const = delete;
  Self& operator++() {
    PostorderVertexIterator<T>::operator++();
    return *this;
  }
  Self operator++(int) {
    Self tmp = *this;
    operator++();
    return tmp;
  }

  bool operator==(const Self& x) const {
    return this->get_vertex() == x.get_vertex();
  }
  bool operator!=(const Self& x) const { return !operator==(x); }
};

template <typename T>
std::pair<PreorderVertexIterator<T>, PreorderVertexIterator<T>>
Topology<T>::preorder_vertexs() {
  PreorderVertexIterator<T> start(_nodes, _root);
  PreorderVertexIterator<T> end(_nodes);
  return std::make_pair(start, end);
}

template <typename T>
std::pair<PostorderVertexIterator<T>, PostorderVertexIterator<T>>
Topology<T>::postorder_vertexs() {
  postorder_vertex_iterator begin(_nodes), end(_nodes);
  begin.set_vertex(begin.first_vertex(_root));
  return std::make_pair(begin, end);
}

template <typename T>
std::pair<EdgeIterator<T>, EdgeIterator<T>> Topology<T>::edges() {
  edge_iterator begin(_nodes), end(_nodes);
  begin.set_vertex(begin.first_vertex(_root));
  end.set_vertex(_root);
  return std::make_pair(begin, end);
}

}  // namespace icts