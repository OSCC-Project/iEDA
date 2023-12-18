#include "HyperGraph.hh"

#include <algorithm>
#include <iostream>
#include <random>

#include "HyperGraphAlgorithm.hh"
struct MyConfig
{
  typedef std::string VertexProperty;
  typedef std::string HedgeProperty;
  typedef std::string EdgeProperty;
  typedef std::string GraphProperty;
};

int main()
{
  using namespace imp;
  const size_t V = 10000;
  const size_t E = 10000;
  const size_t min_degree = 2;
  const size_t max_degree = 6;

  std::vector<std::string> names(V);
  std::generate(std::begin(names), std::end(names), []() {
    static int index = 0;
    return "v" + std::to_string(index++);
  });
  HyperGraph<MyConfig> G;
  G.add_vertexs(names);

  std::vector<size_t> vertices;
  std::for_each(G.vbegin(), G.vend(), [&](decltype(*G.vbegin()) v) { vertices.push_back(v.id); });

  std::random_device seed;
  std::mt19937 gen1(seed());
  std::mt19937 gen2(seed());
  std::poisson_distribution<size_t> rand_degree(min_degree);
  std::uniform_int_distribution<size_t> rand_v(0, V - max_degree - 1);
  std::shuffle(std::begin(vertices), std::end(vertices), gen1);

  G.reserve_hedges(E);
  G.reserve_edges(E * 2);

  // add random hyperedges
  for (size_t i = 0; i < E; i++) {
    size_t degree = rand_degree(gen1);
    degree = std::clamp(degree, min_degree, max_degree);
    size_t first = rand_v(gen2);
    size_t second = first + degree;
    std::vector<size_t> vertex_ids(std::begin(vertices) + first, std::begin(vertices) + second);
    G.add_hyper_edge(vertex_ids);
  }

  G.add_hyper_edge({0, 1, 2, 3});

  auto sub = sub_graph(G, {0, 1, 2, 3});

  return 0;
}
