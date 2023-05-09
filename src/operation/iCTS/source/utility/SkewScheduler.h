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
#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>
#include <boost/property_map/property_map.hpp>
#include <string>
#include <unordered_map>

#include "CTSAPI.hpp"
#include "Skew.h"
#include "UstNode.h"
#include "pgl.h"

namespace icts {

class SkewCommit {
 public:
  SkewCommit(const int &i, const int &j, const Skew &skew_constaint)
      : _i(i), _j(j), _skew_constraint(skew_constaint) {}
  std::pair<int, int> get_ij() const { return std::make_pair(_i, _j); }
  Skew get_skew_constraint() const { return _skew_constraint; }

 private:
  int _i;
  int _j;
  Skew _skew_constraint;
};

class ShortestDistanceMatrix {
 public:
  typedef std::pair<int, int> Edge;
  typedef boost::adjacency_list<
      boost::vecS, boost::vecS, boost::directedS, boost::no_property,
      boost::property<boost::edge_weight_t, double,
                      boost::property<boost::edge_weight2_t, double>>>
      Graph;
  typedef boost::property_map<Graph, boost::edge_weight_t>::type WeightMap;
  typedef boost::graph_traits<Graph>::edge_iterator EdgeIterator;

  ShortestDistanceMatrix() {}

  ShortestDistanceMatrix(const int &n) { matrixInit(n); }

  ShortestDistanceMatrix(const ShortestDistanceMatrix &other) {
    _n = other._n;
    _dist_matrix = other._dist_matrix;
  }

  void matrixInit(const int &n) {
    _n = n;
    _dist_matrix = std::vector<std::vector<double>>(
        _n, std::vector<double>(_n, std::numeric_limits<double>::max()));
  }

  void add_edge(const Edge &edge, const double &weight = 1.0) {
    _edges.emplace_back(edge);
    _weights.emplace_back(weight);
  }

  void add_edges(const std::vector<Edge> &edges,
                 const std::vector<double> &weights) {
    _edges = edges;
    _weights = weights;
  }

  bool buildGraph() {
    reset_matrix();
#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
    // VC++ can't handle the iterator constructor
    Graph graph(_n);
    for (std::size_t j = 0; j < _edges.size(); ++j) {
      add_edge(_edges[j].first, _edges[j].second, graph);
    }
#else
    Graph graph(_edges.begin(), _edges.end(), _n);
#endif
    WeightMap weight_map = boost::get(boost::edge_weight, graph);
    EdgeIterator e =
        EdgeIterator(graph.vertex_set().begin(), graph.vertex_set().begin(),
                     graph.vertex_set().end(), graph);
    EdgeIterator e_end =
        EdgeIterator(graph.vertex_set().begin(), graph.vertex_set().end(),
                     graph.vertex_set().end(), graph);
    std::map<Edge, double> base_weight_map;
    for (size_t idx = 0; idx < _edges.size(); ++idx) {
      base_weight_map[_edges[idx]] = _weights[idx];
    }
    for (boost::tie(e, e_end) = boost::edges(graph); e != e_end; ++e) {
      auto i = boost::source(*e, graph);
      auto j = boost::target(*e, graph);
      auto origin_weight = base_weight_map.at(std::pair<int, int>(i, j));
      weight_map[*e] = origin_weight;
    }
    auto success = boost::johnson_all_pairs_shortest_paths(graph, _dist_matrix);
    if (success) {
      for (int i = 0; i < _n; ++i) {
        _dist_matrix[i][i] = std::numeric_limits<double>::max();
      }
      for (int k = 0; k < _n; ++k) {
        for (int i = 0; i < _n; ++i) {
          auto sum = _dist_matrix[i][k] + _dist_matrix[k][i];
          if (sum < _dist_matrix[i][i]) {
            _dist_matrix[i][i] = sum;
          }
        }
      }
    }
    return success;
  }

  void floydWarshall() {
    reset_matrix();
    for (size_t i = 0; i < _weights.size(); ++i) {
      auto edge = _edges[i];
      auto weight = _weights[i];
      _dist_matrix[edge.first][edge.second] = weight;
    }
    for (int k = 0; k < _n; ++k) {
      for (int i = 0; i < _n; ++i) {
        for (int j = 0; j < _n; ++j) {
          auto sum = _dist_matrix[i][k] + _dist_matrix[k][j];
          if (sum < _dist_matrix[i][j]) {
            _dist_matrix[i][j] = sum;
          }
        }
      }
    }
  }

  void writeMatrix(const std::string &file = "matrix.txt") {
    std::ofstream out(file);
    for (int i = 0; i < _n; ++i) {
      for (int j = 0; j < _n; ++j) {
        if (_dist_matrix[i][j] == std::numeric_limits<double>::max()) {
          out << "inf ";
        } else {
          out << _dist_matrix[i][j] << " ";
        }
      }
      out << std::endl;
    }
  }

  void updateWeight(const SkewCommit &skew_commit) {
    auto i = skew_commit.get_ij().first;
    auto j = skew_commit.get_ij().second;
    auto skew = skew_commit.get_skew_constraint();
    for (size_t idx = 0; idx <= _edges.size() - 1; ++idx) {
      if (_edges[idx].first == i && _edges[idx].second == j) {
        _weights[idx] = skew;
        break;
      }
    }
  }

  bool updateMatrix(const SkewCommit &skew_commit) {
    bool update = false;
    auto i = skew_commit.get_ij().first;
    auto j = skew_commit.get_ij().second;
    if (!(0 <= i && i <= _n - 1 && 0 <= j && j <= _n - 1)) {
      return false;
    }
    auto skew = skew_commit.get_skew_constraint();
    _dist_matrix[i][j] = -skew;
    _dist_matrix[j][i] = skew;
    for (int k = 0; k < _n; ++k) {
      for (int l = 0; l < _n; ++l) {
        if (k == l) {
          continue;
        }
        auto new_dist = {_dist_matrix[k][l],
                         _dist_matrix[k][i] - skew + _dist_matrix[j][l],
                         _dist_matrix[k][j] + skew + _dist_matrix[i][l]};
        auto min_dist = *std::min_element(new_dist.begin(), new_dist.end());
        if (_dist_matrix[k][l] != min_dist) {
          _dist_matrix[k][l] = min_dist;
          update = true;
        }
      }
    }
    return update;
  }

  SkewRange get_feasible_skew_range(const int &i, const int &j) const {
    if (!(0 <= i && i <= _n - 1) || !(0 <= j && j <= _n - 1)) {
      return SkewRange(-1 * _skew_bound, _skew_bound);
    }
    return SkewRange(-1 * _dist_matrix[i][j], _dist_matrix[j][i]);
  }
  void set_skew_bound(const PropagationTime &skew_bound) {
    _skew_bound = skew_bound;
  }
  std::vector<std::vector<double>> get_dist_matrix() const {
    return _dist_matrix;
  }

 private:
  void reset_matrix() {
    _dist_matrix = std::vector<std::vector<double>>(
        _n, std::vector<double>(_n, std::numeric_limits<double>::max()));
  }

  int _n = 0;
  PropagationTime _skew_bound = 0.0;
  std::vector<std::vector<double>> _dist_matrix;

  std::vector<Edge> _edges;
  std::vector<double> _weights;
};

class SkewScheduler {
 public:
  using SkewConstraintMap =
      std::map<std::pair<std::string, std::string>, std::pair<double, double>>;

  SkewScheduler(const SkewConstraintMap &skew_constraints,
                const PropagationTime &skew_bound) {
    _skew_matrix = ShortestDistanceMatrix();
    _skew_matrix.set_skew_bound(skew_bound);
    _vertex_map.clear();
    int id_count = 0;
    // Number the sinks' name
    for (auto itr = skew_constraints.begin(); itr != skew_constraints.end();
         ++itr) {
      auto name_pair = itr->first;

      auto i_id = 0;
      auto i_name = name_pair.first;
      i_id = _vertex_map.find(i_name) == _vertex_map.end()
                 ? _vertex_map[i_name] = id_count++
                 : _vertex_map[i_name];

      auto j_id = 0;
      auto j_name = name_pair.second;
      j_id = _vertex_map.find(j_name) == _vertex_map.end()
                 ? _vertex_map[j_name] = id_count++
                 : _vertex_map[j_name];

      auto skew_range = itr->second;
      _skew_matrix.add_edge(Pair<int>(i_id, j_id), -1 * skew_range.first);
      _skew_matrix.add_edge(Pair<int>(j_id, i_id), skew_range.second);
    }
    _skew_matrix.matrixInit(id_count);
  }
  void buildSkewConstraint() {
    _skew_matrix.buildGraph();
    // _skew_matrix.floydWarshall();
    // _skew_matrix.writeMatrix(_config->get_sta_workspace() + "/" +
    //                          "johnson.txt");
  }

  bool haveNegativeWeightCycle() { return !_skew_matrix.buildGraph(); }

  bool updateDistanceMatrix(const SkewCommit &skew_commit) {
    return _skew_matrix.updateMatrix(skew_commit);
  }

  int find_matrix_id(const std::string &pin_name) const {
    if (_vertex_map.find(pin_name) == _vertex_map.end()) {
      return -1;
    }
    return _vertex_map.at(pin_name);
  }

  SkewRange get_feasible_skew_range(const std::string &i_name,
                                    const std::string &j_name) const {
    auto i_id = find_matrix_id(i_name);
    auto j_id = find_matrix_id(j_name);
    return _skew_matrix.get_feasible_skew_range(i_id, j_id);
  }

  SkewRange get_feasible_skew_range(const int &i, const int &j) const {
    return _skew_matrix.get_feasible_skew_range(i, j);
  }

  std::optional<UstDelay> find_before_delay(const std::string &name) const {
    std::optional<UstDelay> delay;
    if (_final_delay_map.count(name) > 0) {
      delay = _final_delay_map.at(name);
    } else {
      delay = std::nullopt;
    }
    return delay;
  }

  void insert_delay(const std::string &name, const UstDelay &delay) {
    if (name == "") {
      return;
    }
    if (_final_delay_map.find(name) == _final_delay_map.end()) {
      _final_delay_map[name] = delay;
      return;
    }
    assert(false);
  }
  std::vector<std::vector<double>> get_dist_matrix() const {
    return _skew_matrix.get_dist_matrix();
  }

 private:
  ShortestDistanceMatrix _skew_matrix;
  std::unordered_map<std::string, int> _vertex_map;
  std::map<std::string, UstDelay> _final_delay_map;
};

}  // namespace icts