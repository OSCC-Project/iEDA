#ifndef IMP_HYPERGRAPHALG_H
#define IMP_HYPERGRAPHALG_H
#include <set>
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
    sub_graph.add_vertex(property(vertex));
    vertices_map[sub_vertices[i]] = i;
    std::for_each(hebegin(vertex), hend(vertex), [&](decltype(*hebegin(vertex))&& hedge) { hyper_edge_set.insert(hedge.id); });
  }
  for (auto&& i : hyper_edge_set) {
    auto&& hedge = graph.hyper_edge_at(i);
    std::vector<size_t> vertices;
    std::vector<typename Config::EdgeProperty> edge_properties;
    for (size_t i = 0; i < degree(hedge); i++) {
      auto&& vertex = vertex_at(hedge, i);
      auto&& edge = edge_at(hedge, i);
      if (vertices_map.contains(vertex.id)) {
        vertices.push_back(vertices_map[vertex.id]);
        edge_properties.push_back(property(edge));
      }
    }
    if (vertices.size() < degree(hedge))
      cuts.push_back(i);
    if (vertices.size() < 2) {
      continue;
    }
    sub_graph.add_hyper_edge(vertices, edge_properties, property(hedge));
  }
  return {sub_graph, cuts};
}

template <HyperGraphConfig Config, typename Mergefunction>
HyperGraph<Config> clustering(const HyperGraph<Config>& graph, const std::vector<size_t>& parts, Mergefunction f)
{
}
}  // namespace imp
#endif
