#pragma once

#include <climits>
#include <vector>

#include "Traits.h"
#include "pgl.h"

namespace icts {
using std::pair;
using std::vector;
class ClusterCenter {
 public:
  ClusterCenter() = default;
  ClusterCenter(const Point& location, int count = 0)
      : _location(location), _count(count) {}
  ~ClusterCenter() = default;

  // getter
  Point get_location() const { return _location; }
  int get_count() const { return _count; }

  // setter
  void set_location(const Point& location) { _location = location; }
  void set_count(int count) { _count = count; }

  // operator
  void increase_count(int addend) { _count += addend; }

 private:
  Point _location;
  int _count;
};

template <typename Value>
class ClusterPoint {
 public:
  ClusterPoint() = default;
  ClusterPoint(Value value, int index = 0) : _value(value), _index(index) {}
  ~ClusterPoint() = default;

  // getter
  Value get_value() const { return _value; }
  int get_index() const { return _index; }

  // setter
  void set_value(Value value) { _value = value; }
  void set_index(int index) { _index = index; }

 private:
  Value _value;
  int _index;
};

template <typename Value>
class Kmeans {
 public:
  Kmeans() = default;
  Kmeans(const Kmeans&) = delete;
  ~Kmeans() = default;
  Kmeans& operator=(const Kmeans&) = delete;

  vector<vector<Value>> operator()(const vector<Value>& points,
                                   int cluster_size, int cluster_num = 0);

 private:
  vector<vector<Value>> kmeans(const vector<Value>& points, int cluster_size,
                               int cluster_num = 0);
  void init(const std::vector<Value>& points, int cluster_size,
            int cluster_num = 0);
  void partition_point();
  void compute_center_coords(vector<pair<double, double>>& center_coords);
  bool update_center_coords(
      std::vector<std::pair<double, double>>& center_coords);
  void adjust_index(ClusterPoint<Value>& cluster_point);

  Point get_location(const Value& val) const {
    return DataTraits<Value>::getPoint(val);
  }

 private:
  vector<ClusterPoint<Value>> _cluster_points;
  vector<ClusterCenter> _cluster_centers;
  int _cluster_num;
  int _cluster_size;
  int _size_upper_bound;
};

template <typename Value>
vector<vector<Value>> Kmeans<Value>::operator()(const vector<Value>& points,
                                                int cluster_size,
                                                int cluster_num) {
  vector<vector<Value>> clusters = kmeans(points, cluster_size, cluster_num);
  vector<vector<Value>> ret_clusters;
  for (auto& cluster : clusters) {
    if (static_cast<int>(cluster.size()) > _size_upper_bound) {
      auto new_clusters = kmeans(cluster, cluster_size);
      for (auto& new_cluster : new_clusters) {
        ret_clusters.emplace_back(new_cluster);
      }
    } else {
      ret_clusters.emplace_back(cluster);
    }
  }
  return clusters;
}

template <typename Value>
vector<vector<Value>> Kmeans<Value>::kmeans(const vector<Value>& points,
                                            int cluster_size, int cluster_num) {
  init(points, cluster_size, cluster_num);
  bool stop_flag = true;
  while (stop_flag) {
    partition_point();

    vector<pair<double, double>> center_coords(_cluster_num);
    compute_center_coords(center_coords);

    stop_flag = update_center_coords(center_coords);
  }

  vector<vector<Value>> clusters(_cluster_num);
  for (auto& cluster_point : _cluster_points) {
    auto index = cluster_point.get_index();
    auto value = cluster_point.get_value();
    clusters[index].push_back(value);
  }

  auto remove_rule = [](auto& cluster) { return cluster.size() == 0; };
  clusters.erase(std::remove_if(clusters.begin(), clusters.end(), remove_rule),
                 clusters.end());

  return clusters;
}

template <typename Value>
void Kmeans<Value>::init(const std::vector<Value>& points, int cluster_size,
                         int cluster_num) {
  _cluster_centers.clear();
  _cluster_points.clear();

  int points_num = points.size();
  _cluster_num = cluster_num == 0
                     ? (points_num + cluster_size - 1) / cluster_size
                     : cluster_num;
  _cluster_size = cluster_size;
  _size_upper_bound = cluster_size * 3;
  // auto setp = points_num / _cluster_num;
  // set initial center point according to input data
  for (int i = 0; i < _cluster_num; ++i) {
    auto point = points[i * cluster_size];
    auto center = get_location(point);
    _cluster_centers.emplace_back(ClusterCenter(center));
  }

  for (auto point : points) {
    _cluster_points.emplace_back(ClusterPoint<Value>(point));
  }
}

template <typename Value>
void Kmeans<Value>::partition_point() {
  // clear cluster
  for (auto& cluster : _cluster_centers) {
    cluster.set_count(0);
  }

  // partiotion
  // #pragma omp parallel for
  for (auto& cluster_point : _cluster_points) {
    auto cur_loc = get_location(cluster_point.get_value());

    auto itr = std::min_element(
        _cluster_centers.begin(), _cluster_centers.end(),
        [&cur_loc](const auto& center_point1, const auto& conter_point2) {
          return pgl::manhattan_distance(center_point1.get_location(),
                                         cur_loc) <
                 pgl::manhattan_distance(conter_point2.get_location(), cur_loc);
        });
    auto center_index = itr - _cluster_centers.begin();
    cluster_point.set_index(center_index);

    // #pragma omp critical
    _cluster_centers[center_index].increase_count(1);
  }
}

template <typename Value>
void Kmeans<Value>::compute_center_coords(
    vector<pair<double, double>>& center_coords) {
  for (auto& cluster_point : _cluster_points) {
    auto index = cluster_point.get_index();
    auto location = get_location(cluster_point.get_value());
    double coord_x = location.x();
    double coord_y = location.y();

    auto count = _cluster_centers[index].get_count();
    center_coords[index].first += coord_x / count;
    center_coords[index].second += coord_y / count;
  }
}

template <typename Value>
bool Kmeans<Value>::update_center_coords(
    vector<pair<double, double>>& center_coords) {
  bool is_update = false;
  for (int idx = 0; idx < _cluster_num; ++idx) {
    int coord_x = static_cast<int>(center_coords[idx].first + 0.5);
    int coord_y = static_cast<int>(center_coords[idx].second + 0.5);

    auto& cluster_center = _cluster_centers[idx];
    auto center_location = cluster_center.get_location();

    if (coord_x != center_location.x() || coord_y != center_location.y()) {
      is_update = true;
    }
    _cluster_centers[idx].set_location(Point(coord_x, coord_y));
  }
  return is_update;
}

template <typename Value>
void Kmeans<Value>::adjust_index(ClusterPoint<Value>& cluster_point) {
  auto index = cluster_point.get_index();
  auto cur_location = get_location(cluster_point.get_value());

  double min_dist = INT_MAX;
  for (int i = 0; i < _cluster_size; ++i) {
    if (i == index) {
      continue;
    }
    auto center_location = _cluster_centers[i].get_location();
    auto dist = manhattan_distance(cur_location, center_location);
    if (dist < min_dist) {
      min_dist = dist;
      cluster_point.set_index(i);
    }
  }
}

}  // namespace icts