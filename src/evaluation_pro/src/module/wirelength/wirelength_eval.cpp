
#include "wirelength_eval.h"

#include <cmath>
#include <queue>

#include "init_egr.h"
#include "init_flute.h"

namespace ieval {

WirelengthEval::WirelengthEval()
{
  InitFlute init_flute;
  init_flute.readLUT();
}

WirelengthEval::~WirelengthEval()
{
}

int32_t WirelengthEval::evalTotalHPWL(PointSets point_sets)
{
  int32_t total_hpwl = 0;

  for (const auto& point_set : point_sets) {
    total_hpwl += evalNetHPWL(point_set);
  }

  return total_hpwl;
}

int32_t WirelengthEval::evalTotalFLUTE(PointSets point_sets)
{
  int32_t total_stwl = 0;

  for (const auto& point_set : point_sets) {
    total_stwl += evalNetFLUTE(point_set);
  }

  return total_stwl;
}

int32_t WirelengthEval::evalTotalHTree(PointSets point_sets)
{
  int32_t total_htree = 0;

  for (const auto& point_set : point_sets) {
    total_htree += evalNetHTree(point_set);
  }

  return total_htree;
}

int32_t WirelengthEval::evalTotalVTree(PointSets point_sets)
{
  int32_t total_vtree = 0;

  for (const auto& point_set : point_sets) {
    total_vtree += evalNetVTree(point_set);
  }

  return total_vtree;
}

int32_t WirelengthEval::evalNetHPWL(PointSet point_set)
{
  int32_t net_hpwl = 0;

  int32_t max_x = 0;
  int32_t max_y = 0;
  int32_t min_x = INT32_MAX;
  int32_t min_y = INT32_MAX;
  for (size_t i = 0; i < point_set.size(); ++i) {
    max_x = std::max(point_set[i].first, max_x);
    max_y = std::max(point_set[i].second, max_y);
    min_x = std::min(point_set[i].first, min_x);
    min_y = std::min(point_set[i].second, min_y);
  }
  net_hpwl = (max_x - min_x) + (max_y - min_y);

  return net_hpwl;
}

int32_t WirelengthEval::evalNetFLUTE(PointSet point_set)
{
  int32_t net_stwl = 0;

  int num_pin = point_set.size();
  if (num_pin == 2) {
    net_stwl = abs(point_set[0].first - point_set[1].first) + abs(point_set[0].second - point_set[1].second);
  } else if (num_pin > 2) {
    int* x = new int[num_pin];
    int* y = new int[num_pin];

    int j = 0;
    for (auto& point : point_set) {
      x[j] = (int) point.first;
      y[j] = (int) point.second;
      j++;
    }

    InitFlute init_flute;
    Flute::Tree flute_tree = init_flute.flute(num_pin, x, y, 8);

    net_stwl = flute_tree.length;

    delete[] x;
    delete[] y;
    init_flute.freeTree(flute_tree);
  }

  return net_stwl;
}

int32_t WirelengthEval::evalNetHTree(PointSet point_set)
{
  int32_t net_htree = 0;

  int32_t x_direction_gravity = 0;
  for (auto& point : point_set) {
    x_direction_gravity += point.first;
  }
  x_direction_gravity /= point_set.size();

  int32_t x_direction_length = 0;
  int32_t pins_max_y = 0;
  int32_t pins_min_y = INT32_MAX;
  for (auto& point : point_set) {
    x_direction_length += abs(point.first - x_direction_gravity);
    if (point.second > pins_max_y) {
      pins_max_y = point.second;
    }
    if (point.second < pins_min_y) {
      pins_min_y = point.second;
    }
  }
  net_htree = (x_direction_length + pins_max_y - pins_min_y);

  return net_htree;
}

int32_t WirelengthEval::evalNetVTree(PointSet point_set)
{
  int32_t net_vtree = 0;

  int32_t y_direction_gravity = 0;
  for (auto& point : point_set) {
    y_direction_gravity += point.second;
  }
  y_direction_gravity /= point_set.size();

  int32_t y_direction_length = 0;
  int32_t pins_max_x = 0;
  int32_t pins_min_x = INT32_MAX;
  for (auto& point : point_set) {
    y_direction_length += abs(point.second - y_direction_gravity);
    if (point.first > pins_max_x) {
      pins_max_x = point.first;
    }
    if (point.first < pins_min_x) {
      pins_min_x = point.first;
    }
  }
  net_vtree = (y_direction_length + pins_max_x - pins_min_x);

  return net_vtree;
}

int32_t WirelengthEval::evalPathHPWL(PointSet point_set, PointPair point_pair)
{
  int32_t path_hpwl = 0;

  int32_t x1 = point_pair.first.first;
  int32_t y1 = point_pair.first.second;
  int32_t x2 = point_pair.second.first;
  int32_t y2 = point_pair.second.second;

  path_hpwl = abs(x1 - x2) + abs(y1 - y2);

  return path_hpwl;
}

int32_t WirelengthEval::evalPathFLUTE(PointSet point_set, PointPair point_pair)
{
  int32_t path_stwl = 0;

  int num_pin = point_set.size();
  if (num_pin == 2) {
    path_stwl = abs(point_set[0].first - point_set[1].first) + abs(point_set[0].second - point_set[1].second);
  } else if (num_pin > 2) {
    int* x = new int[num_pin];
    int* y = new int[num_pin];

    int j = 0;
    for (auto& point : point_set) {
      x[j] = (int) point.first;
      y[j] = (int) point.second;
      j++;
    }

    InitFlute init_flute;
    Flute::Tree flute_tree = init_flute.flute(num_pin, x, y, 8);

    std::pair<int32_t, int32_t> point1 = point_pair.first;
    std::pair<int32_t, int32_t> point2 = point_pair.second;

    // 找到指定点对在树中的索引
    int start_index = -1, end_index = -1;
    for (int i = 0; i < flute_tree.deg; i++) {
      if (flute_tree.branch[i].x == point1.first && flute_tree.branch[i].y == point1.second) {
        start_index = i;
      }
      if (flute_tree.branch[i].x == point2.first && flute_tree.branch[i].y == point2.second) {
        end_index = i;
      }
      if (start_index != -1 && end_index != -1) {
        break;
      }
    }

    if (start_index != -1 && end_index != -1) {
      std::vector<bool> visited(2 * flute_tree.deg - 2, false);
      std::vector<int> parent(2 * flute_tree.deg - 2, -1);
      std::queue<int> q;

      // 广度优先搜索
      q.push(start_index);
      visited[start_index] = true;

      while (!q.empty()) {
        int current = q.front();
        q.pop();

        if (current == end_index) {
          break;  // 找到终点，结束搜索
        }

        // 检查所有相邻的点
        for (int i = 0; i < 2 * flute_tree.deg - 2; i++) {
          if (!visited[i] && (flute_tree.branch[i].n == current || flute_tree.branch[current].n == i)) {
            visited[i] = true;
            parent[i] = current;
            q.push(i);
          }
        }
      }

      // 如果找到路径，计算长度
      if (visited[end_index]) {
        int current = end_index;
        while (current != start_index) {
          int prev = parent[current];
          path_stwl += std::abs(flute_tree.branch[current].x - flute_tree.branch[prev].x)
                       + std::abs(flute_tree.branch[current].y - flute_tree.branch[prev].y);
          current = prev;
        }
      }
    }

    delete[] x;
    delete[] y;
    init_flute.freeTree(flute_tree);
  }

  return path_stwl;
}

int32_t WirelengthEval::evalPathHTree(PointSet point_set, PointPair point_pair)
{
  int32_t path_htree = 0;

  int32_t x_direction_gravity = 0;
  for (auto& point : point_set) {
    x_direction_gravity += point.first;
  }
  x_direction_gravity /= point_set.size();

  std::pair<int32_t, int32_t> point1 = point_pair.first;
  std::pair<int32_t, int32_t> point2 = point_pair.second;

  int32_t x_direction_length = abs(point1.first - x_direction_gravity) + abs(point2.first - x_direction_gravity);

  path_htree = x_direction_length + abs(point1.second - point2.second);

  return path_htree;
}

int32_t WirelengthEval::evalPathVTree(PointSet point_set, PointPair point_pair)
{
  int32_t path_vtree = 0;

  int32_t y_direction_gravity = 0;
  for (auto& point : point_set) {
    y_direction_gravity += point.second;
  }
  y_direction_gravity /= point_set.size();

  std::pair<int32_t, int32_t> point1 = point_pair.first;
  std::pair<int32_t, int32_t> point2 = point_pair.second;

  int32_t y_direction_length = abs(point1.second - y_direction_gravity) + abs(point2.second - y_direction_gravity);

  path_vtree = y_direction_length + abs(point1.first - point2.first);

  return path_vtree;
}

float WirelengthEval::evalTotalEGRWL(std::string guide_path)
{
  float total_egrwl = 0;

  InitEGR init_EGR;
  total_egrwl = init_EGR.parseEGRWL(guide_path);

  return total_egrwl;
}

float WirelengthEval::evalNetEGRWL(std::string guide_path, std::string net_name)
{
  float net_egrwl = 0;

  InitEGR init_EGR;
  net_egrwl = init_EGR.parseNetEGRWL(guide_path, net_name);

  return net_egrwl;
}

float WirelengthEval::evalPathEGRWL(std::string guide_path, std::string net_name, std::string load_name)
{
  float path_egrwl = 0;

  InitEGR init_EGR;
  path_egrwl = init_EGR.parsePathEGRWL(guide_path, net_name, load_name);

  return path_egrwl;
}

}  // namespace ieval
