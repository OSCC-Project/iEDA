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
#include "compare/ResistanceSolver.hh"

#include <Eigen/Dense>
#include <Eigen/SparseLU>

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/CompareMath.hh"

namespace ircx {
namespace compare_spef {
namespace {

class NetResistanceContext
{
 public:
  explicit NetResistanceContext(const Net& net)
  {
    for (const auto& resistor : net.resistors) {
      if (resistor.resistance <= math::kEpsilon) {
        continue;
      }
      addNode(resistor.node1);
      addNode(resistor.node2);
      resistors_.push_back(&resistor);
    }
    buildComponents();
  }

  auto solve(const std::string& from_node, const std::string& to_node) -> std::optional<double>
  {
    if (from_node == to_node) {
      return 0.0;
    }

    const auto from_it = node_to_index_.find(from_node);
    const auto to_it = node_to_index_.find(to_node);
    if (from_it == node_to_index_.end() || to_it == node_to_index_.end()) {
      return std::nullopt;
    }
    if (!sameComponent(from_it->second, to_it->second)) {
      return std::nullopt;
    }

    GroundSolve ground_solve = buildGroundSolve(to_it->second);
    if (!ground_solve.invertible) {
      return std::nullopt;
    }

    const int from_unknown = ground_solve.unknown_index[from_it->second];
    if (from_unknown < 0) {
      return std::nullopt;
    }

    return solveFromUnknown(ground_solve, from_unknown);
  }

  auto solveMany(const std::vector<NodePair>& pairs, const std::vector<std::size_t>& pair_indices) -> std::vector<std::optional<double>>
  {
    std::vector<std::optional<double>> values;
    values.resize(pair_indices.size());

    const auto endpoint_counts = countPairEndpoints(pairs, pair_indices);
    const auto grouped_requests = groupSolveRequests(pairs, pair_indices, endpoint_counts, values);
    solveGroupedRequests(grouped_requests, values);
    return values;
  }

 private:
  struct SolveRequest
  {
    std::size_t output_index = 0;
    std::size_t source_index = 0;
  };

  struct GroundSolve
  {
    std::size_t matrix_size = 0;
    std::vector<int> unknown_index;
    std::unique_ptr<Eigen::SparseLU<Eigen::SparseMatrix<double>>> solver;
    bool invertible = false;
  };

  using EndpointCounts = std::unordered_map<std::string, std::size_t>;
  using GroundRequests = std::unordered_map<std::size_t, std::vector<SolveRequest>>;

  auto countPairEndpoints(const std::vector<NodePair>& pairs, const std::vector<std::size_t>& pair_indices) const -> EndpointCounts
  {
    EndpointCounts endpoint_counts;
    endpoint_counts.reserve(pair_indices.size() * 2);
    for (std::size_t pair_index : pair_indices) {
      const NodePair& pair = pairs[pair_index];
      endpoint_counts[pair.first]++;
      endpoint_counts[pair.second]++;
    }
    return endpoint_counts;
  }

  auto groupSolveRequests(const std::vector<NodePair>& pairs, const std::vector<std::size_t>& pair_indices,
                          const EndpointCounts& endpoint_counts, std::vector<std::optional<double>>& values) const -> GroundRequests
  {
    GroundRequests requests_by_ground;
    requests_by_ground.reserve(pair_indices.size());
    for (std::size_t output_index = 0; output_index < pair_indices.size(); ++output_index) {
      const NodePair& pair = pairs[pair_indices[output_index]];
      if (pair.first == pair.second) {
        values[output_index] = 0.0;
        continue;
      }

      const auto first_it = node_to_index_.find(pair.first);
      const auto second_it = node_to_index_.find(pair.second);
      if (first_it == node_to_index_.end() || second_it == node_to_index_.end()) {
        continue;
      }
      if (!sameComponent(first_it->second, second_it->second)) {
        continue;
      }

      const std::size_t first_count = endpoint_counts.at(pair.first);
      const std::size_t second_count = endpoint_counts.at(pair.second);
      if (first_count > second_count) {
        requests_by_ground[first_it->second].push_back(SolveRequest{.output_index = output_index, .source_index = second_it->second});
      } else {
        requests_by_ground[second_it->second].push_back(SolveRequest{.output_index = output_index, .source_index = first_it->second});
      }
    }
    return requests_by_ground;
  }

  void solveGroupedRequests(const GroundRequests& requests_by_ground, std::vector<std::optional<double>>& values)
  {
    for (const auto& [ground, requests] : requests_by_ground) {
      GroundSolve ground_solve = buildGroundSolve(ground);
      if (!ground_solve.invertible || requests.empty() || ground_solve.matrix_size == 0) {
        continue;
      }

      Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(ground_solve.matrix_size, requests.size());
      for (std::size_t column = 0; column < requests.size(); ++column) {
        const int source_unknown = ground_solve.unknown_index[requests[column].source_index];
        if (source_unknown >= 0) {
          rhs(source_unknown, column) = 1.0;
        }
      }

      const Eigen::MatrixXd solution = ground_solve.solver->solve(rhs);
      if (ground_solve.solver->info() != Eigen::Success) {
        continue;
      }
      for (std::size_t column = 0; column < requests.size(); ++column) {
        const SolveRequest& request = requests[column];
        const int source_unknown = ground_solve.unknown_index[request.source_index];
        if (source_unknown < 0) {
          continue;
        }
        const double value = solution(source_unknown, column);
        if (std::isfinite(value)) {
          values[request.output_index] = std::abs(value);
        }
      }
    }
  }

  auto solveFromUnknown(const GroundSolve& ground_solve, int from_unknown) const -> std::optional<double>
  {
    if (ground_solve.matrix_size == 0) {
      return std::nullopt;
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(ground_solve.matrix_size);
    rhs[from_unknown] = 1.0;
    const Eigen::VectorXd solution = ground_solve.solver->solve(rhs);
    if (ground_solve.solver->info() != Eigen::Success) {
      return std::nullopt;
    }
    const double value = solution[from_unknown];
    if (!std::isfinite(value)) {
      return std::nullopt;
    }
    return std::abs(value);
  }

  void addNode(const std::string& node)
  {
    if (!node_to_index_.contains(node)) {
      node_to_index_[node] = node_to_index_.size();
    }
  }

  void buildComponents()
  {
    component_by_node_.assign(node_to_index_.size(), -1);
    if (node_to_index_.empty()) {
      return;
    }

    std::vector<std::vector<std::size_t>> adjacency(node_to_index_.size());
    for (const auto* resistor : resistors_) {
      const std::size_t idx1 = node_to_index_.at(resistor->node1);
      const std::size_t idx2 = node_to_index_.at(resistor->node2);
      adjacency[idx1].push_back(idx2);
      adjacency[idx2].push_back(idx1);
    }

    int next_component = 0;
    std::vector<std::size_t> stack;
    for (std::size_t index = 0; index < component_by_node_.size(); ++index) {
      if (component_by_node_[index] >= 0) {
        continue;
      }

      component_by_node_[index] = next_component;
      stack.push_back(index);
      while (!stack.empty()) {
        const std::size_t node = stack.back();
        stack.pop_back();
        for (std::size_t adjacent : adjacency[node]) {
          if (component_by_node_[adjacent] < 0) {
            component_by_node_[adjacent] = next_component;
            stack.push_back(adjacent);
          }
        }
      }
      next_component++;
    }
  }

  auto sameComponent(std::size_t node1, std::size_t node2) const -> bool
  {
    return node1 < component_by_node_.size() && node2 < component_by_node_.size() && component_by_node_[node1] >= 0
           && component_by_node_[node1] == component_by_node_[node2];
  }

  auto buildGroundSolve(std::size_t ground) const -> GroundSolve
  {
    GroundSolve solve;
    solve.unknown_index.assign(node_to_index_.size(), -1);
    if (ground >= component_by_node_.size() || component_by_node_[ground] < 0) {
      return solve;
    }

    const int ground_component = component_by_node_[ground];
    std::size_t component_node_count = 0;
    for (int component : component_by_node_) {
      if (component == ground_component) {
        component_node_count++;
      }
    }
    solve.matrix_size = component_node_count - 1;
    if (solve.matrix_size == 0) {
      solve.invertible = true;
      return solve;
    }

    int next_unknown = 0;
    for (std::size_t index = 0; index < node_to_index_.size(); ++index) {
      if (index != ground && component_by_node_[index] == ground_component) {
        solve.unknown_index[index] = next_unknown++;
      }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(resistors_.size() * 4);
    for (const auto* resistor : resistors_) {
      const std::size_t idx1 = node_to_index_.at(resistor->node1);
      const std::size_t idx2 = node_to_index_.at(resistor->node2);
      if (component_by_node_[idx1] != ground_component || component_by_node_[idx2] != ground_component) {
        continue;
      }
      const double g = 1.0 / resistor->resistance;
      const int u = solve.unknown_index[idx1];
      const int v = solve.unknown_index[idx2];
      if (u >= 0) {
        triplets.emplace_back(u, u, g);
      }
      if (v >= 0) {
        triplets.emplace_back(v, v, g);
      }
      if (u >= 0 && v >= 0) {
        triplets.emplace_back(u, v, -g);
        triplets.emplace_back(v, u, -g);
      }
    }

    Eigen::SparseMatrix<double> conductance(solve.matrix_size, solve.matrix_size);
    conductance.setFromTriplets(triplets.begin(), triplets.end());
    conductance.makeCompressed();
    solve.solver = std::make_unique<Eigen::SparseLU<Eigen::SparseMatrix<double>>>();
    solve.solver->compute(conductance);
    solve.invertible = solve.solver->info() == Eigen::Success;
    return solve;
  }

  std::unordered_map<std::string, std::size_t> node_to_index_;
  std::vector<const Resistor*> resistors_;
  std::vector<int> component_by_node_;
};

}  // namespace

auto ResistanceSolver::equivalentResistance(const Net& net, const std::string& from_node,
                                            const std::string& to_node) const -> std::optional<double>
{
  NetResistanceContext context(net);
  return context.solve(from_node, to_node);
}

auto ResistanceSolver::equivalentResistances(const Net& net, const std::vector<NodePair>& pairs) const -> std::vector<std::optional<double>>
{
  std::vector<std::size_t> pair_indices;
  pair_indices.reserve(pairs.size());
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    pair_indices.push_back(index);
  }

  return equivalentResistances(net, pairs, pair_indices);
}

auto ResistanceSolver::equivalentResistances(const Net& net, const std::vector<NodePair>& pairs,
                                             const std::vector<std::size_t>& pair_indices) const -> std::vector<std::optional<double>>
{
  NetResistanceContext context(net);
  return context.solveMany(pairs, pair_indices);
}

}  // namespace compare_spef
}  // namespace ircx
