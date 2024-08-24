
#include "wirelength_eval.h"

#include <cmath>

#include "init_flute.h"

namespace ieval {

WirelengthEval::WirelengthEval()
{
}

WirelengthEval::~WirelengthEval()
{
}

int32_t WirelengthEval::evalTotalHPWL(PointSets point_sets)
{
  int32_t total_hpwl = 0;

  for (const auto& point_set : point_sets) {
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
    total_hpwl += (max_x - min_x) + (max_y - min_y);
  }

  return total_hpwl;
}

int32_t WirelengthEval::evalTotalFLUTE(PointSets point_sets)
{
  int32_t total_stwl = 0;

  InitFlute init_flute;
  init_flute.runFlute();

  for (const auto& point_set : point_sets) {
    int num_pin = point_set.size();
    if (num_pin == 2) {
      int32_t wirelength = fabs(point_set[0].first - point_set[1].first) + fabs(point_set[0].second - point_set[1].second);
      total_stwl += wirelength;
    } else if (num_pin > 2) {
      int* x = new int[num_pin];
      int* y = new int[num_pin];

      int j = 0;
      for (auto& point : point_set) {
        x[j] = (int) point.first;
        y[j] = (int) point.second;
        j++;
      }

      Flute::Tree flutetree = Flute::flute(num_pin, x, y, 8);
      delete[] x;
      delete[] y;

      int branchnum = 2 * flutetree.deg - 2;
      for (int j = 0; j < branchnum; ++j) {
        int n = flutetree.branch[j].n;
        if (j == n) {
          continue;
        }
        int32_t wirelength = fabs(flutetree.branch[j].x - flutetree.branch[n].x) + fabs(flutetree.branch[j].y - flutetree.branch[n].y);
        total_stwl += wirelength;
      }
      free(flutetree.branch);
    }
  }

  return total_stwl;
}

int32_t WirelengthEval::evalTotalHTree(PointSets point_sets)
{
  int32_t total_htree = 0;

  for (const auto& point_set : point_sets) {
    int32_t HTree = 0;
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
    HTree = (x_direction_length + pins_max_y - pins_min_y);
    total_htree += HTree;
  }

  return total_htree;
}

int32_t WirelengthEval::evalTotalVTree(PointSets point_sets)
{
  int32_t total_vtree = 0;

  for (const auto& point_set : point_sets) {
    int32_t VTree = 0;
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
    VTree = (y_direction_length + pins_max_x - pins_min_x);
    total_vtree += VTree;
  }

  return total_vtree;
}

int32_t WirelengthEval::evalNetHPWL(PointSet point_set)
{
  int32_t stwl;

  return stwl;
}

int32_t WirelengthEval::evalNetFLUTE(PointSet point_set)
{
  int32_t stwl;

  return stwl;
}

int32_t WirelengthEval::evalNetHTree(PointSet point_set)
{
  int32_t stwl;

  return stwl;
}

int32_t WirelengthEval::evalNetVTree(PointSet point_set)
{
  int32_t stwl;

  return stwl;
}

int32_t WirelengthEval::evalPathHPWL(PointSet point_set)
{
  int32_t stwl;

  return stwl;
}

int32_t WirelengthEval::evalPathFLUTE(PointSet point_set)
{
  int32_t stwl;

  return stwl;
}

int32_t WirelengthEval::evalPathHTree(PointSet point_set)
{
  int32_t stwl;

  return stwl;
}

int32_t WirelengthEval::evalPathVTree(PointSet point_set)
{
  int32_t stwl;

  return stwl;
}

}  // namespace ieval
