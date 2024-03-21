#ifndef IMP_HYPERGRAPHALG_H
#define IMP_HYPERGRAPHALG_H
#include <concepts>
#include <map>
#include <set>
#include <tuple>
#include <unordered_set>

#include "HyperGraph.hh"
namespace imp {

template <HyperGraphConfig Config>
std::pair<HyperGraph<Config>, std::vector<size_t>> sub_graph(const HyperGraph<Config>& graph, const std::vector<size_t>& sub_vertices)
{
  HyperGraph<Config> sub_graph;
  std::set<size_t> hyper_edge_set;
  std::vector<size_t> cuts;
  std::unordered_map<size_t, size_t> vertices_map;
  for (size_t i = 0; i < sub_vertices.size(); i++) {
    auto&& vertex = graph.vertex_at(sub_vertices[i]);
    sub_graph.add_vertex(vertex.property());
    vertices_map[sub_vertices[i]] = i;
    std::for_each(hebegin(vertex), hend(vertex), [&](auto&& hedge) { hyper_edge_set.insert(hedge.pos()); });
  }
  for (auto&& i : hyper_edge_set) {
    auto&& hedge = graph.hyper_edge_at(i);
    std::vector<size_t> vertices;
    std::vector<typename Config::EdgeProperty> edge_properties;
    for (size_t i = 0; i < degree(hedge); i++) {
      auto&& vertex = vertex_at(hedge, i);
      auto&& edge = edge_at(hedge, i);
      if (vertices_map.contains(vertex.pos())) {
        vertices.push_back(vertices_map[vertex.pos()]);
        edge_properties.push_back(edge.property());
      }
    }
    if (vertices.size() < degree(hedge))
      cuts.push_back(i);
    if (vertices.size() < 2) {
      continue;
    }
    sub_graph.add_hyper_edge(vertices, edge_properties, hedge.property());
  }
  return {sub_graph, cuts};
}

template <HyperGraphConfig Config>
typename Config::VertexProperty defaultMerge(const HyperGraph<Config>& graph, const std::vector<size_t>& vertices)
{
  return typename Config::VertexProperty();
}

template <HyperGraphConfig Config>
typename Config::HedgeProperty defaultHedge(const HyperGraph<Config>& graph, size_t id)
{
  return graph.hyper_edge_at(id).property();
}

template <HyperGraphConfig Config>
typename Config::EdgeProperty defaultEdge(const HyperGraph<Config>& graph, size_t id)
{
  return graph.edge_at(id).property();
}

template <HyperGraphConfig Config, typename MergeVertices = decltype(defaultMerge<Config>),
          typename CreatHyperEdge = decltype(defaultHedge<Config>), typename CreatEdge = decltype(defaultEdge<Config>)>
HyperGraph<Config> clustering(const HyperGraph<Config>& graph, const std::vector<size_t>& parts, MergeVertices merge = defaultMerge<Config>,
                              CreatHyperEdge creat_hedge = defaultHedge<Config>, CreatEdge creat_edge = defaultEdge<Config>)
{
  HyperGraph<Config> cluster_graph(graph.property());

  size_t num_vertices = *std::max_element(std::begin(parts), std::end(parts)) + size_t(1);
  std::vector<std::vector<size_t>> cluster(num_vertices);
  for (size_t i = 0; i < parts.size(); i++) {
    cluster[parts[i]].push_back(i);
  }
  for (auto&& i : cluster) {
    auto&& vertex_property = merge(graph, i);
    cluster_graph.add_vertex(vertex_property);
  }
  std::for_each(hebegin(graph), hend(graph), [&](auto&& hedge) {
    std::multimap<size_t, size_t> vertices;
    std::unordered_set<size_t> ids;
    for (size_t i = 0; i < degree(hedge); i++) {
      ids.insert(parts[vertex_at(hedge, i).pos()]);
      vertices.insert({parts[vertex_at(hedge, i).pos()], edge_at(hedge, i).pos()});
    }
    if (ids.size() > 1) {
      std::vector<size_t> vertices_id;
      std::vector<typename Config::EdgeProperty> edge_properties;
      for (auto&& [vertex_id, edge_id] : vertices) {
        vertices_id.push_back(vertex_id);
        edge_properties.push_back(creat_edge(graph, edge_id));
      }
      cluster_graph.add_hyper_edge(vertices_id, edge_properties, creat_hedge(graph, hedge.pos()));
    }
  });

  return cluster_graph;
}

template <typename T>
None NoneWeight(T)
{
  return None();
}

template <HyperGraphConfig Config, typename vertex_weight = decltype(NoneWeight<typename Config::VertexProperty>),
          typename hedge_weight = decltype(NoneWeight<typename Config::HedgeProperty>),
          typename edge_weight = decltype(NoneWeight<typename Config::EdgeProperty>)>
  requires std::is_invocable_v<vertex_weight, typename Config::VertexProperty>
           && std::is_invocable_v<hedge_weight, typename Config::HedgeProperty>
           && std::is_invocable_v<edge_weight, typename Config::EdgeProperty>
auto vectorize(const HyperGraph<Config>& graph, vertex_weight vwgt = NoneWeight<typename Config::VertexProperty>,
               hedge_weight hewgt = NoneWeight<typename Config::HedgeProperty>,
               edge_weight ewgt = NoneWeight<typename Config::EdgeProperty>)
{
  using vwgt_type = decltype(vwgt(typename Config::VertexProperty()));
  using hewgt_type = decltype(hewgt(typename Config::HedgeProperty()));
  using ewgt_type = decltype(ewgt(typename Config::EdgeProperty()));
  std::vector<size_t> eptr(graph.heSize() + 1);
  std::vector<size_t> eind(graph.eSize());
  std::vector<vwgt_type> vwgts;
  std::vector<hewgt_type> hewgts;
  std::vector<ewgt_type> ewgts;
  constexpr bool has_vwgt = !std::is_same_v<vertex_weight, decltype(NoneWeight<typename Config::VertexProperty>)>;
  constexpr bool has_hewgt = !std::is_same_v<hedge_weight, decltype(NoneWeight<typename Config::HedgeProperty>)>;
  constexpr bool has_ewgt = !std::is_same_v<edge_weight, decltype(NoneWeight<typename Config::EdgeProperty>)>;
  if constexpr (has_vwgt)
    vwgts.resize(graph.vSize());
  if constexpr (has_hewgt)
    hewgts.resize(graph.heSize());
  if constexpr (has_ewgt)
    ewgts.resize(graph.eSize());
  eptr[0] = 0;
  for (size_t i = 0; i < graph.heSize(); i++) {
    auto hy_edge = graph.hyper_edge_at(i);
    eptr[i + 1] = eptr[i] + hy_edge.degree();
    for (size_t j = 0; j < hy_edge.degree(); j++) {
      auto edge = hy_edge.edge_at(j);
      eind[eptr[i] + j] = edge.vertex().pos();
      if constexpr (has_ewgt)
        ewgts[eptr[i] + j] = ewgt(edge.property());
    }
    if constexpr (has_hewgt)
      hewgts[i] = hewgt(hy_edge.property());
  }
  if constexpr (has_vwgt) {
    for (size_t i = 0; i < graph.vSize(); i++) {
      vwgts[i] = vwgt(graph.vertex_at(i).property());
    }
  }
  if constexpr (has_vwgt && has_hewgt && has_ewgt)
    return std::make_tuple(std::move(eptr), std::move(eind), std::move(vwgts), std::move(hewgts), std::move(ewgts));
  else if constexpr (has_vwgt && has_hewgt)
    return std::make_tuple(std::move(eptr), std::move(eind), std::move(vwgts), std::move(hewgts));
  else if constexpr (has_vwgt && has_ewgt)
    return std::make_tuple(std::move(eptr), std::move(eind), std::move(vwgts), std::move(ewgts));
  else if constexpr (has_hewgt && has_ewgt)
    return std::make_tuple(std::move(eptr), std::move(eind), std::move(hewgts), std::move(ewgts));
  else if constexpr (has_vwgt)
    return std::make_tuple(std::move(eptr), std::move(eind), std::move(vwgts));
  else if constexpr (has_hewgt)
    return std::make_tuple(std::move(eptr), std::move(eind), std::move(hewgts));
  else if constexpr (has_ewgt)
    return std::make_tuple(std::move(eptr), std::move(eind), std::move(ewgts));
  else
    return std::make_tuple(std::move(eptr), std::move(eind));
}
}  // namespace imp
#endif
