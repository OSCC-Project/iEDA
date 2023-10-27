#ifndef IMP_HYPERGRAPH_H
#define IMP_HYPERGRAPH_H
#include <cassert>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
/**
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

template <typename T>
class GraphIterator
{
 public:
  GraphIterator(const std::vector<T*>& data, size_t index) : _data(data), _index(index) {}

  const T& operator*() const { return (_index < _data.size()) ? *(_data[_index]) : *(*(_data.end())); }
  T& operator*() { return (_index < _data.size()) ? *(_data[_index]) : *(*(_data.end())); }

  GraphIterator& operator+=(size_t n)
  {
    _index += n;
    return *this;
  }

  GraphIterator operator+(size_t n)
  {
    size_t index = _index + n;
    return GraphIterator<T>(_data, index);
  }

  GraphIterator& operator++()
  {
    ++_index;
    return *this;
  }
  bool operator==(const GraphIterator& other) const { return &_data == &other._data && _index == other._index; }
  bool operator!=(const GraphIterator& other) const { return &_data != &other._data || _index != other._index; }

 private:
  const std::vector<T*>& _data;
  size_t _index;
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
  using VertexIterator = GraphIterator<Vertex>;
  using HedgeIterator = GraphIterator<Hedge>;
  using EdgeIterator = GraphIterator<Edge>;

 public:
  HyperGraph(const GraphProperty& graph_property = GraphProperty());
  HyperGraph(size_t n, const GraphProperty& graph_property = GraphProperty());
  HyperGraph(const std::vector<VertexProperty>& vertex_properties, const GraphProperty& graph_property = GraphProperty());

  const Vertex& vertex_at(size_t n) const;
  Vertex& vertex_at(size_t n);

  const Edge& edge_at(size_t n) const;
  Edge& edge_at(size_t n);

  const Hedge& hyper_edge_at(size_t n) const;
  Hedge& hyper_edge_at(size_t n);

  size_t add_vertex(const VertexProperty& vertex_property = VertexProperty());
  std::vector<size_t> add_vertexs(const std::vector<VertexProperty>& vertex_properties);
  std::vector<size_t> add_vertexs(size_t n);

  size_t add_edge(size_t vertex_id, size_t hedge_id, const EdgeProperty& edge_property = EdgeProperty());

  size_t add_hyper_edge(const std::vector<size_t>& vertex_ids, const std::vector<EdgeProperty>& edge_properties,
                        const HedgeProperty& hedge_property = HedgeProperty());
  size_t add_hyper_edge(const std::vector<size_t>& vertex_ids);

  size_t add_hyper_edges(const std::vector<size_t>& eptr, const std::vector<size_t>& eind,
                         const std::vector<HedgeProperty>& hedge_properties, const std::vector<EdgeProperty>& edge_properties);

  const VertexIterator vbegin() const { return VertexIterator(_vertices, 0); }
  const VertexIterator vend() const { return VertexIterator(_vertices, _vertices.size()); }

  VertexIterator vbegin() { return VertexIterator(_vertices, 0); }
  VertexIterator vend() { return VertexIterator(_vertices, _vertices.size()); }

  HedgeIterator hebegin() { return HedgeIterator(_hyper_edges, 0); }
  HedgeIterator hend() { return HedgeIterator(_hyper_edges, _hyper_edges.size()); }

  EdgeIterator ebegin() { return EdgeIterator(_edges, 0); }
  EdgeIterator eend() { return EdgeIterator(_edges, _edges.size()); }

  // Noted that the id will chage after removing;
  bool remove_vertex(size_t n);
  bool remove_edge(size_t n);
  bool remove_hyper_edge(size_t n);

  VertexIterator remove_vertex(const VertexIterator& pos);
  EdgeIterator remove_edge(const EdgeIterator& pos);
  HedgeIterator remove_hyper_edge(const HedgeIterator& pos);

  void reserve_vertices(size_t n) { _vertices.reserve(n); }
  void reserve_hedges(size_t n) { _hyper_edges.reserve(n); }
  void reserve_edges(size_t n) { _edges.reserve(n); }

 public:
  struct Vertex
  {
    const HedgeIterator hebegin() const { return HedgeIterator(hyper_edges, 0); }
    HedgeIterator hebegin() { return HedgeIterator(hyper_edges, 0); }

    const HedgeIterator hend() const { return HedgeIterator(hyper_edges, hyper_edges.size()); }
    HedgeIterator hend() { return HedgeIterator(hyper_edges, hyper_edges.size()); }

    size_t degree() const { return hyper_edges.size(); }
    size_t id;
    std::vector<Hedge*> hyper_edges;
    std::vector<Edge*> edges;
    VertexProperty property{};
  };

  struct Hedge
  {
    const VertexIterator vbegin() const { return VertexIterator(vertices, 0); }
    VertexIterator vbegin() { return VertexIterator(vertices, 0); }

    const VertexIterator vend() const { return VertexIterator(vertices, vertices.size()); }
    VertexIterator vend() { return VertexIterator(vertices, vertices.size()); }

    const Vertex& vertex_at(size_t n) const { return *(vertices.at(n)); }
    Vertex& vertex_at(size_t n) { return *(vertices.at(n)); }

    const Edge& edge_at(size_t n) const { return *(edges.at(n)); }
    Edge& edge_at(size_t n) { return *(edges.at(n)); }

    size_t degree() const { return vertices.size(); }

    size_t id;
    std::vector<Vertex*> vertices;
    std::vector<Edge*> edges;
    HedgeProperty property{};
  };

  struct Edge
  {
    size_t id;
    Vertex* vertex;
    Hedge* hedge;
    EdgeProperty property{};
  };

 private:
  GraphProperty _property;
  std::vector<Vertex*> _vertices;
  std::vector<Hedge*> _hyper_edges;
  std::vector<Edge*> _edges;
  // std::shared_ptr<IdAllocator> _id_allocator;
};

template <HyperGraphConfig Config>
inline HyperGraph<Config>::HyperGraph(const GraphProperty& graph_property) : _property(graph_property)
{
}

template <HyperGraphConfig Config>
inline HyperGraph<Config>::HyperGraph(size_t n, const GraphProperty& graph_property) : HyperGraph(graph_property)
{
  add_vertexs(n);
}

template <HyperGraphConfig Config>
inline HyperGraph<Config>::HyperGraph(const std::vector<VertexProperty>& vertex_properties, const GraphProperty& graph_property)
    : HyperGraph(graph_property)
{
  add_vertexs(vertex_properties);
}
// template <HyperGraphConfig Config>
// inline const Vertex<typename Config::VertexProperty>& HyperGraph<Config>::vertex_at(size_t vertex_id) const
// {
//   return *(_vertices.at(vertex_id));
// }
template <HyperGraphConfig Config>
inline const HyperGraph<Config>::Vertex& HyperGraph<Config>::vertex_at(size_t n) const
{
  return *(_vertices.at(n));
}
template <HyperGraphConfig Config>
inline HyperGraph<Config>::Vertex& HyperGraph<Config>::vertex_at(size_t n)
{
  return *(_vertices.at(n));
}

template <HyperGraphConfig Config>
inline const HyperGraph<Config>::Edge& HyperGraph<Config>::edge_at(size_t n) const
{
  return *(_edges.at(n));
}

template <HyperGraphConfig Config>
inline HyperGraph<Config>::Edge& HyperGraph<Config>::edge_at(size_t n)
{
  return *(_edges.at(n));
}

template <HyperGraphConfig Config>
inline const HyperGraph<Config>::Hedge& HyperGraph<Config>::hyper_edge_at(size_t n) const
{
  return *(_hyper_edges.at(n));
}

template <HyperGraphConfig Config>
inline HyperGraph<Config>::Hedge& HyperGraph<Config>::hyper_edge_at(size_t n)
{
  return *(_hyper_edges.at(n));
}

template <HyperGraphConfig Config>
inline size_t HyperGraph<Config>::add_vertex(const VertexProperty& vertex_property)
{
  size_t id = _vertices.size();
  _vertices.push_back(new Vertex(id, {}, {}, vertex_property));
  return id;
}

template <HyperGraphConfig Config>
inline std::vector<size_t> HyperGraph<Config>::add_vertexs(const std::vector<VertexProperty>& vertex_properties)
{
  std::vector<size_t> ids;
  _vertices.reserve(_vertices.size() + vertex_properties.size());
  for (auto&& i : vertex_properties) {
    ids.push_back(add_vertex(i));
  }
  return ids;
}

template <HyperGraphConfig Config>
inline std::vector<size_t> HyperGraph<Config>::add_vertexs(size_t n)
{
  return add_vertexs(std::vector<VertexProperty>(n));
}

template <HyperGraphConfig Config>
inline size_t HyperGraph<Config>::add_edge(size_t vertex_id, size_t hedge_id, const EdgeProperty& edge_property)
{
  assert((_vertices.size() >= vertex_id) && (_hyper_edges.size() >= hedge_id));
  size_t id = _edges.size();
  Vertex* vertex = _vertices[vertex_id];
  Hedge* hedge = _hyper_edges[hedge_id];
  _edges.push_back(new Edge(id, vertex, hedge, edge_property));
  vertex->hyper_edges.push_back(hedge);
  vertex->edges.push_back(_edges.back());
  hedge->vertices.push_back(vertex);
  hedge->edges.push_back(_edges.back());
  return id;
}

template <HyperGraphConfig Config>
inline size_t HyperGraph<Config>::add_hyper_edge(const std::vector<size_t>& vertex_ids, const std::vector<EdgeProperty>& edge_properties,
                                                 const HedgeProperty& hedge_property)
{
  size_t id = _hyper_edges.size();
  _hyper_edges.push_back(new Hedge(id, {}, {}, hedge_property));
  for (size_t i = 0; i < vertex_ids.size(); i++) {
    assert(_vertices.size() >= vertex_ids[i]);
    add_edge(vertex_ids[i], id, edge_properties[i]);
  }
  return id;
}

template <HyperGraphConfig Config>
inline size_t HyperGraph<Config>::add_hyper_edge(const std::vector<size_t>& vertex_ids)
{
  return add_hyper_edge(vertex_ids, std::vector<EdgeProperty>(vertex_ids.size()));
}

template <HyperGraphConfig Config>
Config::VertexProperty& VertexProperty(const HyperGraph<Config>& graph, size_t n)
{
  return graph.vertex_at(n).property;
}

template <HyperGraphConfig Config>
Config::HedgeProperty HedgeProperty(HyperGraph<Config> graph, size_t n)
{
}

template <HyperGraphConfig Config>
Config::EdgeProperty EdgeProperty(HyperGraph<Config> graph, size_t n)
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
  return t.vbegin();
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
auto vertex_at(T& t, size_t n) -> decltype(t.vertex_at(n))
{
  return t.vertex_at(n);
}
template <typename T>
auto edge_at(T& t, size_t n) -> decltype(t.edge_at(n))
{
  return t.edge_at(n);
}

template <typename T>
auto property(const T& t) -> const decltype(t.property)&
{
  return t.property;
}
}  // namespace imp

#endif