/*
 * @FilePath: wirelength_eval.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include "wirelength_eval.h"

#include <cmath>
#include <iostream>
#include <queue>
#include <stdexcept>

#include "init_egr.h"
#include "init_flute.h"
#include "init_idb.h"

namespace ieval {

#define EVAL_INIT_IDB_INST (ieval::InitIDB::getInst())
#define EVAL_INIT_FLUTE_INST (ieval::InitFlute::getInst())
#define EVAL_INIT_EGR_INST (ieval::InitEGR::getInst())

WirelengthEval* WirelengthEval::_wirelength_eval = nullptr;

WirelengthEval::WirelengthEval()
{
}

WirelengthEval::~WirelengthEval()
{
}

WirelengthEval* WirelengthEval::getInst()
{
  if (_wirelength_eval == nullptr) {
    _wirelength_eval = new WirelengthEval();
  }

  return _wirelength_eval;
}

void WirelengthEval::destroyInst()
{
  if (_wirelength_eval != nullptr) {
    delete _wirelength_eval;
    _wirelength_eval = nullptr;
  }
}

int64_t WirelengthEval::evalTotalHPWL(PointSets point_sets)
{
  int64_t total_hpwl = 0;

  for (const auto& point_set : point_sets) {
    total_hpwl += evalNetHPWL(point_set);
  }

  return total_hpwl;
}

int64_t WirelengthEval::evalTotalFLUTE(PointSets point_sets)
{
  int64_t total_stwl = 0;

  for (const auto& point_set : point_sets) {
    total_stwl += evalNetFLUTE(point_set);
  }

  return total_stwl;
}

int64_t WirelengthEval::evalTotalHTree(PointSets point_sets)
{
  int64_t total_htree = 0;

  for (const auto& point_set : point_sets) {
    total_htree += evalNetHTree(point_set);
  }

  return total_htree;
}

int64_t WirelengthEval::evalTotalVTree(PointSets point_sets)
{
  int64_t total_vtree = 0;

  for (const auto& point_set : point_sets) {
    total_vtree += evalNetVTree(point_set);
  }

  return total_vtree;
}

int64_t WirelengthEval::evalTotalHPWL()
{
  return evalTotalHPWL(EVAL_INIT_IDB_INST->getPointSets());
}

int64_t WirelengthEval::evalTotalFLUTE()
{
  return evalTotalFLUTE(EVAL_INIT_IDB_INST->getPointSets());
}

int64_t WirelengthEval::evalTotalHTree()
{
  return evalTotalHTree(EVAL_INIT_IDB_INST->getPointSets());
}

int64_t WirelengthEval::evalTotalVTree()
{
  return evalTotalVTree(EVAL_INIT_IDB_INST->getPointSets());
}

int64_t WirelengthEval::evalTotalEGRWL()
{
  return evalTotalEGRWL(EVAL_INIT_EGR_INST->getEGRDirPath() + "/early_router/route.guide");
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

    Flute::Tree flute_tree = EVAL_INIT_FLUTE_INST->flute(num_pin, x, y, 8);

    net_stwl = flute_tree.length;

    delete[] x;
    delete[] y;
    EVAL_INIT_FLUTE_INST->freeTree(flute_tree);
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

    Flute::Tree flute_tree = EVAL_INIT_FLUTE_INST->flute(num_pin, x, y, 8);

    std::pair<int32_t, int32_t> point1 = point_pair.first;
    std::pair<int32_t, int32_t> point2 = point_pair.second;

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

      q.push(start_index);
      visited[start_index] = true;

      while (!q.empty()) {
        int current = q.front();
        q.pop();

        if (current == end_index) {
          break;
        }

        for (int i = 0; i < 2 * flute_tree.deg - 2; i++) {
          if (!visited[i] && (flute_tree.branch[i].n == current || flute_tree.branch[current].n == i)) {
            visited[i] = true;
            parent[i] = current;
            q.push(i);
          }
        }
      }

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
    EVAL_INIT_FLUTE_INST->freeTree(flute_tree);
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

double WirelengthEval::evalTotalEGRWL(std::string guide_path)
{
  return EVAL_INIT_EGR_INST->parseEGRWL(guide_path);
}

float WirelengthEval::evalNetEGRWL(std::string guide_path, std::string net_name)
{
  return EVAL_INIT_EGR_INST->parseNetEGRWL(guide_path, net_name);
}

float WirelengthEval::evalPathEGRWL(std::string guide_path, std::string net_name, std::string load_name)
{
  return EVAL_INIT_EGR_INST->parsePathEGRWL(guide_path, net_name, load_name);
}

int32_t WirelengthEval::getDesignUnit()
{
  return EVAL_INIT_IDB_INST->getDesignUnit();
}

std::vector<std::pair<int32_t, int32_t>> WirelengthEval::getNetPointSet(std::string net_name)
{
  auto name_point_set = EVAL_INIT_IDB_INST->getNamePointSet();
  if (name_point_set.find(net_name) != name_point_set.end()) {
    return name_point_set[net_name];
  } else {
    std::cout << "Net " << net_name << " not found!" << std::endl;
    return {};
  }
}

std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> WirelengthEval::getNamePointSet()
{
  return EVAL_INIT_IDB_INST->getNamePointSet();
}

void WirelengthEval::initIDB()
{
  EVAL_INIT_IDB_INST->initPointSets();
}

void WirelengthEval::destroyIDB()
{
  EVAL_INIT_IDB_INST->destroyInst();
}

void WirelengthEval::initEGR()
{
  EVAL_INIT_EGR_INST->runEGR();
}

void WirelengthEval::destroyEGR()
{
  EVAL_INIT_EGR_INST->destroyInst();
}

std::string WirelengthEval::getEGRDirPath()
{
  return EVAL_INIT_EGR_INST->getEGRDirPath();
}

void WirelengthEval::initFlute()
{
  EVAL_INIT_FLUTE_INST->readLUT();
}

void WirelengthEval::destroyFlute()
{
  EVAL_INIT_FLUTE_INST->destroyInst();
}

void WirelengthEval::evalNetInfo()
{
  auto name_pointset = getNamePointSet();
  EVAL_INIT_EGR_INST->parseGuideFile(EVAL_INIT_EGR_INST->getEGRDirPath() + "/early_router/route.guide");
  for (const auto& [net_name, point_set] : name_pointset) {
    _name_hpwl[net_name] = evalNetHPWL(point_set);
    _name_flute[net_name] = evalNetFLUTE(point_set);
    _name_grwl[net_name] = EVAL_INIT_EGR_INST->getNetEGRWL(net_name) * getDesignUnit();
  }
}

void WirelengthEval::evalNetFlute()
{
  auto name_pointset = getNamePointSet();
  for (const auto& [net_name, point_set] : name_pointset) {
    _name_flute[net_name] = evalNetFLUTE(point_set);
  }
}

int32_t WirelengthEval::findHPWL(std::string net_name)
{
  auto it = _name_hpwl.find(net_name);
  if (it != _name_hpwl.end()) {
    return it->second;
  }
  throw std::runtime_error("HPWL not found for net: Net " + net_name);
}

int32_t WirelengthEval::findFLUTE(std::string net_name)
{
  auto it = _name_flute.find(net_name);
  if (it != _name_flute.end()) {
    return it->second;
  }
  throw std::runtime_error("FLUTE not found for net: Net " + net_name);
}

int32_t WirelengthEval::findGRWL(std::string net_name)
{
  auto it = _name_grwl.find(net_name);
  if (it != _name_grwl.end()) {
    return it->second;
  }
  throw std::runtime_error("GRWL not found for net: Net " + net_name);
}

}  // namespace ieval
