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
/**
 * @file KMeans.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-16
 * @brief K-Means clustering template for topology.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "Geometry.hh"
#include "Point.hh"

namespace icts {

template <typename Value>
class KMeans
{
 public:
  struct Result
  {
    std::vector<Point<double>> centers;
    std::vector<std::size_t> labels;
  };

  KMeans() = default;
  ~KMeans() = default;

  template <typename PointGetter>
  auto run(const std::vector<Value>& values, std::size_t k, PointGetter getter, std::size_t max_iter = 10,
           double converge_threshold = 1.0) const -> Result
  {
    Result result;
    if (values.empty() || k == 0) {
      return result;
    }

    if (k > values.size()) {
      k = values.size();
    }

    result.labels.assign(values.size(), 0);
    result.centers = initCenters(values, k, getter);

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
      bool changed = false;

      for (std::size_t i = 0; i < values.size(); ++i) {
        const auto point = toPoint(getter(values[i]));
        std::size_t best_index = 0;
        double best_dist = std::numeric_limits<double>::max();
        for (std::size_t c = 0; c < result.centers.size(); ++c) {
          const double dist = geometry::Manhattan(point, result.centers[c]);
          if (dist < best_dist) {
            best_dist = dist;
            best_index = c;
          }
        }
        if (result.labels[i] != best_index) {
          result.labels[i] = best_index;
          changed = true;
        }
      }

      std::vector<double> sum_x(result.centers.size(), 0.0);
      std::vector<double> sum_y(result.centers.size(), 0.0);
      std::vector<std::size_t> counts(result.centers.size(), 0);

      for (std::size_t i = 0; i < values.size(); ++i) {
        const auto label_index = result.labels[i];
        const auto point = toPoint(getter(values[i]));
        sum_x[label_index] += point.get_x();
        sum_y[label_index] += point.get_y();
        ++counts[label_index];
      }

      double max_shift = 0.0;
      for (std::size_t c = 0; c < result.centers.size(); ++c) {
        if (counts[c] == 0) {
          continue;
        }
        Point<double> new_center(sum_x[c] / static_cast<double>(counts[c]), sum_y[c] / static_cast<double>(counts[c]));
        max_shift = std::max(max_shift, geometry::Manhattan(result.centers[c], new_center));
        result.centers[c] = new_center;
      }

      if (!changed || max_shift < converge_threshold) {
        break;
      }
    }

    return result;
  }

 private:
  template <typename PointGetter>
  auto initCenters(const std::vector<Value>& values, std::size_t k, PointGetter getter) const -> std::vector<Point<double>>
  {
    std::vector<Point<double>> centers;
    centers.reserve(k);
    centers.push_back(toPoint(getter(values.front())));

    while (centers.size() < k) {
      double best_dist = -1.0;
      std::size_t best_index = 0;
      for (std::size_t i = 0; i < values.size(); ++i) {
        const auto point = toPoint(getter(values[i]));
        double nearest = std::numeric_limits<double>::max();
        for (const auto& center : centers) {
          nearest = std::min(nearest, geometry::Manhattan(point, center));
        }
        if (nearest > best_dist) {
          best_dist = nearest;
          best_index = i;
        }
      }
      centers.push_back(toPoint(getter(values[best_index])));
    }

    return centers;
  }

  template <typename PointType>
  auto toPoint(const PointType& point) const -> Point<double>
  {
    return Point<double>(point.get_x(), point.get_y());
  }
};

}  // namespace icts
