// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#ifndef IMP_HYPERGRAPH_H
#define IMP_HYPERGRAPH_H
#include <cassert>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
/**
 * TODO: make enable to using std::ranges through fix class GraphIterator and class GraphRange.
 * hmatrix = (6, 5, 11);
 *
 *     he0  he1  he2  he3  he4
 *
 * v0  e0------------------e9
 *     |                   |
 * v1  e1--------e4        |
 *               |         |
 * v2       e2---e5---e6---e10
 *          |         |
 * v3       e3        |
 *                    |
 * v4                 e7
 *                    |
 * v5                 e8
 *
 * vertices:
 * v0 = {(he0, e0), (he4, e9)};
 * v1 = {(he0, e1), (he2, e4)};
 * v2 = {(he1, e2), (he2, e5), (he3, e6), (he4, e10)};
 * v3 = {(he1, e3)};
 * v4 = {(he3, e7)};
 * v5 = {(he3, e8)};
 *
 * hedges:
 * he0 = {(v0, e0), (v1, e1)};
 * he1 = {(v2, e2), (v3, e3)};
 * he2 = {(v1, e4), (v2, e5)};
 * he3 = {(v2, e6), (v4, e7), (v5, e8)};
 * he4 = {(v0, e9), (v2, e10)};
 *
 * edges:
 * e0 = {he0, v0};
 * e1 = {he0, v1};
 * ...
 * ...
 * e10 = {he4, v2};
 */
namespace imp {
struct None
{
};

template <template <typename> class ptr_type, typename T>
class GraphIterator
{
 public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;

  GraphIterator(const std::vector<ptr_type<T>>& data, size_t index) : _data(data), _index(index) {}

  const reference operator*() const
  {
    if constexpr (std::is_same_v<ptr_type<T>, std::weak_ptr<T>>) {
      return *(_data.at(_index).lock());
    } else {
      return *(_data.at(_index));
    }
  }

  reference operator*()
  {
    if constexpr (std::is_same_v<ptr_type<T>, std::weak_ptr<T>>) {
      return *(_data.at(_index).lock());
    } else {
      return *(_data.at(_index));
    }
  }
  GraphIterator& operator=(const GraphIterator& other)
  {
    if (this != &other && &_data != &other._data) {
      const_cast<std::vector<ptr_type<T>>&>(_data) = other._data;
    }
    _index = other._index;
    return *this;
  }
  GraphIterator& operator+=(size_t pos)
  {
    _index += pos;
    return *this;
  }

  GraphIterator operator+(size_t pos) const
  {
    size_t index = _index + pos;
    return GraphIterator<ptr_type, T>(_data, index);
  }

  GraphIterator& operator-=(size_t pos)
  {
    _index -= pos;
    return *this;
  }

  GraphIterator operator-(size_t pos) const
  {
    size_t index = _index - pos;
    return GraphIterator<ptr_type, T>(_data, index);
  }

  GraphIterator& operator++()
  {
    ++_index;
    return *this;
  }

  GraphIterator operator++(int)
  {
    GraphIterator<ptr_type, T> temp(*this);
    ++(*this);
    return temp;
  }

  GraphIterator& operator--()
  {
    --_index;
    return *this;
  }

  GraphIterator operator--(int)
  {
    GraphIterator<ptr_type, T> temp(*this);
    --(*this);
    return temp;
  }

  bool operator==(const GraphIterator& other) const { return &_data == &other._data && _index == other._index; }
  bool operator!=(const GraphIterator& other) const { return &_data != &other._data || _index != other._index; }

  bool operator<(const GraphIterator& other) const { return _index < other._index; }
  bool operator>(const GraphIterator& other) const { return _index > other._index; }
  bool operator<=(const GraphIterator& other) const { return _index <= other._index; }
  bool operator>=(const GraphIterator& other) const { return _index >= other._index; }

  reference operator[](size_t pos) { return *(_data.at(_index + pos)); }
  const reference operator[](size_t pos) const { return *(_data.at(_index + pos)); }

  difference_type operator-(const GraphIterator& other) const { return _index - other._index; }

 private:
  const std::vector<ptr_type<T>>& _data;
  size_t _index;
};

template <template <typename> class ptr_type, typename T>
auto make_iterator(const std::vector<ptr_type<T>>& data, size_t index)
{
  return GraphIterator<ptr_type, T>(data, index);
}

template <typename Iterator>
class GraphRange
{
 public:
  GraphRange() = delete;
  GraphRange(Iterator begin, Iterator end) : _begin(begin), _end(end) {}
  Iterator begin() { return _begin; }
  Iterator end() { return _end; }

 private:
  Iterator _begin;
  Iterator _end;
};

template <typename Config>
concept HyperGraphConfig = requires {
  typename Config::VertexProperty;
  typename Config::HedgeProperty;
  typename Config::EdgeProperty;
  typename Config::GraphProperty;
};

struct DefaultHyperGraph
{
  typedef None VertexProperty;
  typedef None HedgeProperty;
  typedef None EdgeProperty;
  typedef None GraphProperty;
};

template <HyperGraphConfig Config = DefaultHyperGraph>
class HyperGraph
{
 public:
  struct Vertex;
  struct Hedge;
  struct Edge;
  using VertexProperty = Config::VertexProperty;
  using HedgeProperty = Config::HedgeProperty;
  using EdgeProperty = Config::EdgeProperty;
  using GraphProperty = Config::GraphProperty;
  // using GraphIterator<std::shared_ptr, Vertex> = GraphIterator<Vertex>;
  // using GraphIterator<std::shared_ptr, Hedge> = GraphIterator<Hedge>;
  // using GraphIterator<std::shared_ptr, Edge> = GraphIterator<Edge>;

 public:
  HyperGraph(const GraphProperty& graph_property = GraphProperty());
  HyperGraph(size_t pos, const GraphProperty& graph_property = GraphProperty());
  HyperGraph(const std::vector<VertexProperty>& vertex_properties, const GraphProperty& graph_property = GraphProperty());
  ~HyperGraph();

  bool empty() const { return _vertices.empty(); }
  GraphProperty& property() { return _property; }
  const GraphProperty& property() const { return _property; }

  const Vertex& vertex_at(size_t pos) const;
  Vertex& vertex_at(size_t pos);

  const Edge& edge_at(size_t pos) const;
  Edge& edge_at(size_t pos);

  const Hedge& hyper_edge_at(size_t pos) const;
  Hedge& hyper_edge_at(size_t pos);

  size_t add_vertex(const VertexProperty& vertex_property = VertexProperty());
  std::vector<size_t> add_vertices(const std::vector<VertexProperty>& vertex_properties);
  std::vector<size_t> add_vertices(size_t n);

  size_t add_edge(size_t vertex_id, size_t hedge_id, const EdgeProperty& edge_property = EdgeProperty());

  size_t add_hyper_edge(const std::vector<size_t>& vertex_ids, const std::vector<EdgeProperty>& edge_properties,
                        const HedgeProperty& hedge_property = HedgeProperty());
  size_t add_hyper_edge(const std::vector<size_t>& vertex_ids);

  size_t add_hyper_edges(const std::vector<size_t>& eptr, const std::vector<size_t>& eind,
                         const std::vector<HedgeProperty>& hedge_properties, const std::vector<EdgeProperty>& edge_properties);
  size_t vSize() const { return _vertices.size(); }
  size_t heSize() const { return _hyper_edges.size(); }
  size_t eSize() const { return _edges.size(); }

  const GraphIterator<std::shared_ptr, Vertex> vbegin() const { return GraphIterator<std::shared_ptr, Vertex>(_vertices, 0); }
  const GraphIterator<std::shared_ptr, Vertex> vend() const { return GraphIterator<std::shared_ptr, Vertex>(_vertices, _vertices.size()); }

  GraphIterator<std::shared_ptr, Vertex> vbegin() { return GraphIterator<std::shared_ptr, Vertex>(_vertices, 0); }
  GraphIterator<std::shared_ptr, Vertex> vend() { return GraphIterator<std::shared_ptr, Vertex>(_vertices, _vertices.size()); }

  const GraphIterator<std::shared_ptr, Hedge> hebegin() const { return GraphIterator<std::shared_ptr, Hedge>(_hyper_edges, 0); }
  const GraphIterator<std::shared_ptr, Hedge> hend() const
  {
    return GraphIterator<std::shared_ptr, Hedge>(_hyper_edges, _hyper_edges.size());
  }

  GraphIterator<std::shared_ptr, Hedge> hebegin() { return GraphIterator<std::shared_ptr, Hedge>(_hyper_edges, 0); }
  GraphIterator<std::shared_ptr, Hedge> hend() { return GraphIterator<std::shared_ptr, Hedge>(_hyper_edges, _hyper_edges.size()); }

  GraphIterator<std::shared_ptr, Edge> ebegin() { return GraphIterator<std::shared_ptr, Edge>(_edges, 0); }
  GraphIterator<std::shared_ptr, Edge> eend() { return GraphIterator<std::shared_ptr, Edge>(_edges, _edges.size()); }

  GraphRange<GraphIterator<std::shared_ptr, Vertex>> vRange() { return {vbegin(), vend()}; }
  GraphRange<GraphIterator<std::shared_ptr, Hedge>> heRange() { return {hebegin(), hend()}; }

  // Noted that the pos will chage after removing;
  bool remove_vertex(size_t pos);
  bool remove_edge(size_t pos);
  bool remove_hyper_edge(size_t pos);

  GraphIterator<std::shared_ptr, Vertex> remove_vertex(const GraphIterator<std::shared_ptr, Vertex>& pos);
  GraphIterator<std::shared_ptr, Edge> remove_edge(const GraphIterator<std::shared_ptr, Edge>& pos);
  GraphIterator<std::shared_ptr, Hedge> remove_hyper_edge(const GraphIterator<std::shared_ptr, Hedge>& pos);

  void reserve_vertices(size_t n) { _vertices.reserve(n); }
  void reserve_hedges(size_t n) { _hyper_edges.reserve(n); }
  void reserve_edges(size_t n) { _edges.reserve(n); }

 public:
  class Vertex
  {
   public:
    Vertex(size_t pos, const std::vector<std::weak_ptr<Hedge>>& hyper_edges, const std::vector<std::weak_ptr<Edge>>& edges,
           const VertexProperty& property)
        : _pos(pos), _hyper_edges(hyper_edges), _edges(edges), _property(property)
    {
    }
    const GraphIterator<std::weak_ptr, Hedge> hebegin() const { return GraphIterator<std::weak_ptr, Hedge>(_hyper_edges, 0); }
    GraphIterator<std::weak_ptr, Hedge> hebegin() { return GraphIterator<std::weak_ptr, Hedge>(_hyper_edges, 0); }

    const GraphIterator<std::weak_ptr, Hedge> hend() const
    {
      return GraphIterator<std::weak_ptr, Hedge>(_hyper_edges, _hyper_edges.size());
    }
    GraphIterator<std::weak_ptr, Hedge> hend() { return GraphIterator<std::weak_ptr, Hedge>(_hyper_edges, _hyper_edges.size()); }

    size_t degree() const { return _hyper_edges.size(); }
    VertexProperty& property() { return _property; }
    const VertexProperty& property() const { return _property; }

    size_t pos() const { return _pos; }

   private:
    friend class HyperGraph;
    size_t _pos;
    std::vector<std::weak_ptr<Hedge>> _hyper_edges;
    std::vector<std::weak_ptr<Edge>> _edges;
    VertexProperty _property{};
  };

  class Hedge
  {
   public:
    Hedge(size_t pos, std::vector<std::weak_ptr<Vertex>> vertices, std::vector<std::weak_ptr<Edge>> edges, HedgeProperty property)
        : _pos(pos), _vertices(vertices), _edges(edges), _property(property)
    {
    }
    const GraphIterator<std::weak_ptr, Vertex> vbegin() const { return GraphIterator<std::weak_ptr, Vertex>(_vertices, 0); }
    GraphIterator<std::weak_ptr, Vertex> vbegin() { return GraphIterator<std::weak_ptr, Vertex>(_vertices, 0); }

    const GraphIterator<std::weak_ptr, Vertex> vend() const { return GraphIterator<std::weak_ptr, Vertex>(_vertices, _vertices.size()); }
    GraphIterator<std::weak_ptr, Vertex> vend() { return GraphIterator<std::shared_ptr, Vertex>(_vertices, _vertices.size()); }

    const Vertex& vertex_at(size_t pos) const { return *(_vertices.at(pos).lock()); }
    Vertex& vertex_at(size_t pos) { return *(_vertices.at(pos).lock()); }

    const Edge& edge_at(size_t pos) const { return *(_edges.at(pos).lock()); }
    Edge& edge_at(size_t pos) { return *(_edges.at(pos).lock()); }

    size_t degree() const { return _vertices.size(); }
    HedgeProperty& property() { return _property; }
    const HedgeProperty& property() const { return _property; }

    size_t pos() const { return _pos; }

   private:
    friend class HyperGraph;
    size_t _pos;
    std::vector<std::weak_ptr<Vertex>> _vertices;
    std::vector<std::weak_ptr<Edge>> _edges;
    HedgeProperty _property{};
  };

  class Edge
  {
   public:
    Edge(size_t pos, std::weak_ptr<Vertex> vertex, std::weak_ptr<Hedge> hedge, const EdgeProperty& property)
        : _pos(pos), _vertex(vertex), _hedge(hedge), _property(property)
    {
    }
    Vertex& vertex() { return *_vertex.lock(); }
    const Vertex& vertex() const { return *_vertex.lock(); }
    EdgeProperty& property() { return _property; }
    const EdgeProperty& property() const { return _property; }
    size_t pos() const { return _pos; }

   private:
    friend class HyperGraph;
    size_t _pos;
    std::weak_ptr<Vertex> _vertex;
    std::weak_ptr<Hedge> _hedge;
    EdgeProperty _property{};
  };

 private:
  GraphProperty _property{};
  std::vector<std::shared_ptr<Vertex>> _vertices;
  std::vector<std::shared_ptr<Hedge>> _hyper_edges;
  std::vector<std::shared_ptr<Edge>> _edges;
  // std::std::shared_ptr<IdAllocator> _id_allocator;
};

template <HyperGraphConfig Config>
inline HyperGraph<Config>::HyperGraph(const GraphProperty& graph_property) : _property(graph_property)
{
}

template <HyperGraphConfig Config>
inline HyperGraph<Config>::HyperGraph(size_t pos, const GraphProperty& graph_property) : HyperGraph(graph_property)
{
  add_vertices(pos);
}

template <HyperGraphConfig Config>
inline HyperGraph<Config>::HyperGraph(const std::vector<VertexProperty>& vertex_properties, const GraphProperty& graph_property)
    : HyperGraph(graph_property)
{
  add_vertices(vertex_properties);
}
template <HyperGraphConfig Config>
inline HyperGraph<Config>::~HyperGraph()
{
}
// template <HyperGraphConfig Config>
// inline const Vertex<typename Config::VertexProperty>& HyperGraph<Config>::vertex_at(size_t vertex_id) const
// {
//   return *(_vertices.at(vertex_id));
// }
template <HyperGraphConfig Config>
inline const HyperGraph<Config>::Vertex& HyperGraph<Config>::vertex_at(size_t pos) const
{
  return *(_vertices.at(pos));
}
template <HyperGraphConfig Config>
inline HyperGraph<Config>::Vertex& HyperGraph<Config>::vertex_at(size_t pos)
{
  return *(_vertices.at(pos));
}

template <HyperGraphConfig Config>
inline const HyperGraph<Config>::Edge& HyperGraph<Config>::edge_at(size_t pos) const
{
  return *(_edges.at(pos));
}

template <HyperGraphConfig Config>
inline HyperGraph<Config>::Edge& HyperGraph<Config>::edge_at(size_t pos)
{
  return *(_edges.at(pos));
}

template <HyperGraphConfig Config>
inline const HyperGraph<Config>::Hedge& HyperGraph<Config>::hyper_edge_at(size_t pos) const
{
  return *(_hyper_edges.at(pos));
}

template <HyperGraphConfig Config>
inline HyperGraph<Config>::Hedge& HyperGraph<Config>::hyper_edge_at(size_t pos)
{
  return *(_hyper_edges.at(pos));
}

template <HyperGraphConfig Config>
inline size_t HyperGraph<Config>::add_vertex(const VertexProperty& vertex_property)
{
  size_t pos = _vertices.size();
  // auto v = std::make_shared<Vertex>(pos, std::vector<std::shared_ptr<Hedge>>(), std::vector<std::shared_ptr<Edge>>(), vertex_property);
  _vertices.emplace_back(new Vertex(pos, {}, {}, vertex_property));
  return pos;
}

template <HyperGraphConfig Config>
inline std::vector<size_t> HyperGraph<Config>::add_vertices(const std::vector<VertexProperty>& vertex_properties)
{
  std::vector<size_t> ids;
  _vertices.reserve(_vertices.size() + vertex_properties.size());
  for (auto&& i : vertex_properties) {
    ids.push_back(add_vertex(i));
  }
  return ids;
}

template <HyperGraphConfig Config>
inline std::vector<size_t> HyperGraph<Config>::add_vertices(size_t n)
{
  return add_vertices(std::vector<VertexProperty>(n));
}

template <HyperGraphConfig Config>
inline size_t HyperGraph<Config>::add_edge(size_t vertex_id, size_t hedge_id, const EdgeProperty& edge_property)
{
  assert((_vertices.size() >= vertex_id) && (_hyper_edges.size() >= hedge_id));
  size_t pos = _edges.size();
  std::shared_ptr<Vertex> vertex = _vertices[vertex_id];
  std::shared_ptr<Hedge> hedge = _hyper_edges[hedge_id];
  _edges.emplace_back(new Edge(pos, vertex, hedge, edge_property));
  vertex->_hyper_edges.push_back(hedge);
  vertex->_edges.push_back(_edges.back());
  hedge->_vertices.push_back(vertex);
  hedge->_edges.push_back(_edges.back());
  return pos;
}

template <HyperGraphConfig Config>
inline size_t HyperGraph<Config>::add_hyper_edge(const std::vector<size_t>& vertex_ids, const std::vector<EdgeProperty>& edge_properties,
                                                 const HedgeProperty& hedge_property)
{
  size_t pos = _hyper_edges.size();
  _hyper_edges.emplace_back(new Hedge(pos, {}, {}, hedge_property));
  for (size_t i = 0; i < vertex_ids.size(); i++) {
    assert(_vertices.size() >= vertex_ids[i]);
    add_edge(vertex_ids[i], pos, edge_properties[i]);
  }
  return pos;
}

template <HyperGraphConfig Config>
inline size_t HyperGraph<Config>::add_hyper_edge(const std::vector<size_t>& vertex_ids)
{
  return add_hyper_edge(vertex_ids, std::vector<EdgeProperty>(vertex_ids.size()));
}

template <HyperGraphConfig Config>
Config::VertexProperty& VertexProperty(const HyperGraph<Config>& graph, size_t pos)
{
  return graph.vertex_at(pos)._property;
}

template <HyperGraphConfig Config>
Config::HedgeProperty HedgeProperty(HyperGraph<Config> graph, size_t pos)
{
}

template <HyperGraphConfig Config>
Config::EdgeProperty EdgeProperty(HyperGraph<Config> graph, size_t pos)
{
}

template <typename T>
auto vbegin(const T& t) -> const decltype(t.vbegin())
{
  return t.vbegin();
}

template <typename T>
auto vend(const T& t) -> const decltype(t.vend())
{
  return t.vend();
}
template <typename T>
auto hebegin(const T& t) -> const decltype(t.hebegin())
{
  return t.hebegin();
}

template <typename T>
auto hend(const T& t) -> const decltype(t.hend())
{
  return t.hend();
}

template <typename T>
auto degree(const T& t) -> const decltype(t.degree())
{
  return t.degree();
}

template <typename T>
auto vertex_at(T& t, size_t pos) -> decltype(t.vertex_at(pos))
{
  return t.vertex_at(pos);
}
template <typename T>
auto edge_at(T& t, size_t pos) -> decltype(t.edge_at(pos))
{
  return t.edge_at(pos);
}

auto property(auto&& t) 
{
  return t.property();
}
}  // namespace imp

#endif