/**
 * @file CostCalculator.h
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace icts {

struct IdPair {
  int id_1;
  int id_2;
  int get_first() const { return id_1; }
  int get_second() const { return id_2; }
  bool operator<(const IdPair &p) const {
    if (id_1 < p.get_first()) {
      return true;
    }
    if (id_1 == p.get_first() && id_2 < p.get_second()) {
      return true;
    }
    return false;
  }
  bool operator==(const IdPair &p) const {
    return id_1 == p.id_1 && id_2 == p.id_2;
  }
};

struct CostNode {
  double cost;
  IdPair pair;
  IdPair get_pair() const { return pair; }
  double get_cost() const { return cost; }
  bool operator<(const CostNode &n) const {
    if (std::fabs(cost - n.get_cost()) <
        std::numeric_limits<double>::epsilon()) {
      return pair < n.get_pair();
    }
    return cost < n.get_cost();
  }
};

class CostCalculator {
 public:
  CostCalculator() = default;
  ~CostCalculator() = default;

  void insertId(const int &i) { _id_set.insert(i); }

  void initID(const int &n) {
    for (int i = 0; i < n; ++i) {
      insertId(i);
    }
  }

  void reset() {
    _id_set.clear();
    _cost_set.clear();
    _cost_map.clear();
  }

  void costReset() {
    _cost_set.clear();
    _cost_map.clear();
  }

  void insertCost(const int &i, const int &j, const double &cost) {
    _cost_map[IdPair{i, j}] = cost;
    _cost_map[IdPair{j, i}] = cost;
    _cost_set.insert(CostNode{cost, IdPair{i, j}});
  }

  void popMinCost() {
    auto itr = _cost_set.begin();
    auto id_1 = itr->get_pair().get_first();
    auto id_2 = itr->get_pair().get_second();
    auto cost_remove_by = [id_1, id_2](const CostNode &n) {
      return n.get_pair().get_first() == id_1 ||
             n.get_pair().get_first() == id_2 ||
             n.get_pair().get_second() == id_1 ||
             n.get_pair().get_second() == id_2;
    };
    std::erase_if(_cost_set, cost_remove_by);
    auto id_remove_by = [id_1, id_2](const int &i) {
      return i == id_1 || i == id_2;
    };
    std::erase_if(_id_set, id_remove_by);
  }

  CostNode minCostNode() const { return *_cost_set.begin(); }

  double minCost() const {
    if (_cost_set.empty()) {
      return std::numeric_limits<double>::max();
    }
    return _cost_set.begin()->cost;
  }

  IdPair minCostPair() const { return _cost_set.begin()->pair; }

  std::set<int> get_id_set() const { return _id_set; }

 private:
  std::set<int> _id_set;
  std::multiset<CostNode> _cost_set;
  std::map<IdPair, double> _cost_map;
};

}  // namespace icts