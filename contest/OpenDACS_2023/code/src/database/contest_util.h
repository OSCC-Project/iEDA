#pragma once

#include <map>
#include <queue>
#include <set>
#include <unordered_set>
#include <vector>

#include "contest_coord.h"
#include "contest_segment.h"

namespace ieda_contest {

class ContestUtil
{
 public:
  template <typename Key>
  static bool exist(const std::vector<Key>& vector, const Key& key)
  {
    for (size_t i = 0; i < vector.size(); i++) {
      if (vector[i] == key) {
        return true;
      }
    }
    return false;
  }

  template <typename Key, typename Compare = std::less<Key>>
  static bool exist(const std::set<Key, Compare>& set, const Key& key)
  {
    return (set.find(key) != set.end());
  }

  template <typename Key, typename Hash = std::hash<Key>>
  static bool exist(const std::unordered_set<Key, Hash>& set, const Key& key)
  {
    return (set.find(key) != set.end());
  }

  template <typename Key, typename Value, typename Compare = std::less<Key>>
  static bool exist(const std::map<Key, Value, Compare>& map, const Key& key)
  {
    return (map.find(key) != map.end());
  }

  template <typename Key, typename Value, typename Hash = std::hash<Key>>
  static bool exist(const std::unordered_map<Key, Value, Hash>& map, const Key& key)
  {
    return (map.find(key) != map.end());
  }

  template <typename T>
  static std::queue<T> initQueue(const T& t)
  {
    std::vector<T> list{t};
    return initQueue(list);
  }

  template <typename T>
  static std::queue<T> initQueue(std::vector<T>& list)
  {
    std::queue<T> queue;
    addListToQueue(queue, list);
    return queue;
  }

  template <typename T>
  static T getFrontAndPop(std::queue<T>& queue)
  {
    T node = queue.front();
    queue.pop();
    return node;
  }

  template <typename T>
  static void addListToQueue(std::queue<T>& queue, std::vector<T>& list)
  {
    for (size_t i = 0; i < list.size(); i++) {
      queue.push(list[i]);
    }
  }

  template <typename T, typename Compare>
  static void swapByCMP(T& a, T& b, Compare cmp)
  {
    if (!cmp(a, b)) {
      std::swap(a, b);
    }
  }

  template <typename T>
  static void swapASC(T& a, T& b)
  {
    swapByCMP(a, b, std::less<T>());
  }

  template <typename T, typename MergeIf>
  static void merge(std::vector<T>& list, MergeIf mergeIf)
  {
    size_t save_id = 0;
    size_t sentry_id = 0;
    size_t soldier_id = sentry_id + 1;
    while (sentry_id < list.size()) {
      T& sentry = list[sentry_id];
      while (soldier_id < list.size()) {
        T& soldier = list[soldier_id];
        if (!mergeIf(sentry, soldier)) {
          break;
        }
        ++soldier_id;
      }
      list[save_id] = std::move(sentry);
      ++save_id;
      if (!(soldier_id < list.size())) {
        break;
      }
      sentry_id = soldier_id;
      soldier_id = sentry_id + 1;
    }
    list.erase(list.begin() + save_id, list.end());
  }

  // 判断线段是否为一个点
  static bool isProximal(const ContestCoord& start_coord, const ContestCoord& end_coord)
  {
    return (start_coord.get_x() == end_coord.get_x() && start_coord.get_y() == end_coord.get_y());
  }

  // 判断线段是否为水平线
  static bool isHorizontal(const ContestCoord& start_coord, const ContestCoord& end_coord)
  {
    return (start_coord.get_x() != end_coord.get_x() && start_coord.get_y() == end_coord.get_y());
  }

  // 判断线段是否为竖直线
  static bool isVertical(const ContestCoord& start_coord, const ContestCoord& end_coord)
  {
    return (start_coord.get_x() == end_coord.get_x() && start_coord.get_y() != end_coord.get_y());
  }

  static bool passCheckingConnectivity(std::vector<ContestCoord>& key_coord_list, std::vector<ContestSegment>& segment_list)
  {
    // 判断是否有斜线段
    if (!passCheckingOblique(segment_list)) {
      std::cout << "There are oblique segments in segment_list!" << std::endl;
      return false;
    }
    // 删除点线段
    erasePointSegment(segment_list);
    // 融合重叠的线段
    mergeOverlapSegment(key_coord_list, segment_list);
    // 是否到达所有的关键坐标
    return passCheckingKeyConnectivity(key_coord_list, segment_list);
  }

  // 判断是否有斜线
  static bool passCheckingOblique(std::vector<ContestSegment>& segment_list)
  {
    for (ContestSegment& segment : segment_list) {
      ContestCoord& first_coord = segment.get_first();
      ContestCoord& second_coord = segment.get_second();

      int diff_num = 0;
      if (first_coord.get_x() != second_coord.get_x()) {
        diff_num++;
      }
      if (first_coord.get_y() != second_coord.get_y()) {
        diff_num++;
      }
      if (first_coord.get_layer_idx() != second_coord.get_layer_idx()) {
        diff_num++;
      }
      if (diff_num > 1) {
        return false;
      }
    }
    return true;
  }

  // 删除点线段
  static void erasePointSegment(std::vector<ContestSegment>& segment_list)
  {
    std::vector<ContestSegment> new_segment_list;
    for (ContestSegment& segment : segment_list) {
      if (segment.get_first() == segment.get_second()) {
        continue;
      }
      new_segment_list.push_back(segment);
    }
    segment_list = new_segment_list;
  }

  // 融合重叠的线段
  static void mergeOverlapSegment(std::vector<ContestCoord>& key_coord_list, std::vector<ContestSegment>& segment_list)
  {
    std::vector<ContestSegment> h_segment_list;
    std::vector<ContestSegment> v_segment_list;
    std::vector<ContestSegment> p_segment_list;

    for (ContestSegment& segment : segment_list) {
      if (isHorizontal(segment.get_first(), segment.get_second())) {
        h_segment_list.push_back(segment);
      } else if (isVertical(segment.get_first(), segment.get_second())) {
        v_segment_list.push_back(segment);
      } else if (isProximal(segment.get_first(), segment.get_second())) {
        p_segment_list.push_back(segment);
      }
    }
    // 先切柱子
    std::vector<ContestSegment> p_segment_list_temp;
    for (ContestSegment& p_segment : p_segment_list) {
      int x = p_segment.get_first().get_x();
      int y = p_segment.get_first().get_y();
      int first_layer_idx = p_segment.get_first().get_layer_idx();
      int second_layer_idx = p_segment.get_second().get_layer_idx();
      swapASC(first_layer_idx, second_layer_idx);
      for (int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
        p_segment_list_temp.emplace_back(ContestCoord(x, y, layer_idx), ContestCoord(x, y, layer_idx + 1));
      }
    }
    p_segment_list = p_segment_list_temp;

    // 初始化平面切点
    std::map<int, std::set<int>> x_cut_list_map;
    std::map<int, std::set<int>> y_cut_list_map;

    for (ContestSegment& h_segment : h_segment_list) {
      ContestCoord& first_coord = h_segment.get_first();
      ContestCoord& second_coord = h_segment.get_second();
      int layer_idx = first_coord.get_layer_idx();

      x_cut_list_map[layer_idx].insert(first_coord.get_x());
      x_cut_list_map[layer_idx].insert(second_coord.get_x());
      y_cut_list_map[layer_idx].insert(first_coord.get_y());
    }
    for (ContestSegment& v_segment : v_segment_list) {
      ContestCoord& first_coord = v_segment.get_first();
      ContestCoord& second_coord = v_segment.get_second();
      int layer_idx = first_coord.get_layer_idx();

      y_cut_list_map[layer_idx].insert(first_coord.get_y());
      y_cut_list_map[layer_idx].insert(second_coord.get_y());
      x_cut_list_map[layer_idx].insert(first_coord.get_x());
    }
    for (ContestSegment& p_segment : p_segment_list) {
      ContestCoord& first_coord = p_segment.get_first();
      int first_layer_idx = first_coord.get_layer_idx();

      ContestCoord& second_coord = p_segment.get_second();
      int second_layer_idx = second_coord.get_layer_idx();

      x_cut_list_map[first_layer_idx].insert(first_coord.get_x());
      y_cut_list_map[first_layer_idx].insert(first_coord.get_y());
      x_cut_list_map[second_layer_idx].insert(second_coord.get_x());
      y_cut_list_map[second_layer_idx].insert(second_coord.get_y());
    }
    for (ContestCoord& key_coord : key_coord_list) {
      int layer_idx = key_coord.get_layer_idx();
      x_cut_list_map[layer_idx].insert(key_coord.get_x());
      y_cut_list_map[layer_idx].insert(key_coord.get_y());
    }

    // 切割平面的h线
    std::vector<ContestSegment> h_segment_list_temp;
    for (ContestSegment& h_segment : h_segment_list) {
      int first_x = h_segment.get_first().get_x();
      int second_x = h_segment.get_second().get_x();
      int y = h_segment.get_first().get_y();
      int layer_idx = h_segment.get_first().get_layer_idx();

      swapASC(first_x, second_x);
      std::vector<int> x_list;
      for (int x_cut : x_cut_list_map[layer_idx]) {
        if (first_x <= x_cut && x_cut <= second_x) {
          x_list.push_back(x_cut);
        }
      }
      for (size_t i = 1; i < x_list.size(); i++) {
        ContestCoord first_coord(x_list[i - 1], y, layer_idx);
        ContestCoord second_coord(x_list[i], y, layer_idx);
        h_segment_list_temp.emplace_back(first_coord, second_coord);
      }
    }
    h_segment_list = h_segment_list_temp;

    // 切割平面的v线
    std::vector<ContestSegment> v_segment_list_temp;
    for (ContestSegment& v_segment : v_segment_list) {
      int first_y = v_segment.get_first().get_y();
      int second_y = v_segment.get_second().get_y();
      int x = v_segment.get_first().get_x();
      int layer_idx = v_segment.get_first().get_layer_idx();

      swapASC(first_y, second_y);
      std::vector<int> y_list;
      for (int y_cut : y_cut_list_map[layer_idx]) {
        if (first_y <= y_cut && y_cut <= second_y) {
          y_list.push_back(y_cut);
        }
      }
      for (size_t i = 1; i < y_list.size(); i++) {
        ContestCoord first_coord(x, y_list[i - 1], layer_idx);
        ContestCoord second_coord(x, y_list[i], layer_idx);
        v_segment_list_temp.emplace_back(first_coord, second_coord);
      }
    }
    v_segment_list = v_segment_list_temp;

    auto mergeSegmentList = [](std::vector<ContestSegment>& segment_list) {
      for (ContestSegment& segment : segment_list) {
        SortSegmentInner()(segment);
      }
      std::sort(segment_list.begin(), segment_list.end(), CmpSegment());
      merge(segment_list, [](ContestSegment& sentry, ContestSegment& soldier) {
        return (sentry.get_first() == soldier.get_first()) && (sentry.get_second() == soldier.get_second());
      });
    };
    mergeSegmentList(h_segment_list);
    mergeSegmentList(v_segment_list);
    mergeSegmentList(p_segment_list);

    segment_list.clear();
    segment_list.insert(segment_list.end(), h_segment_list.begin(), h_segment_list.end());
    segment_list.insert(segment_list.end(), v_segment_list.begin(), v_segment_list.end());
    segment_list.insert(segment_list.end(), p_segment_list.begin(), p_segment_list.end());
  }

  // 是否到达所有的关键坐标
  static bool passCheckingKeyConnectivity(std::vector<ContestCoord>& key_coord_list, std::vector<ContestSegment>& segment_list)
  {
    if (key_coord_list.size() == 0 || key_coord_list.size() == 1) {
      return true;
    }
    std::vector<std::pair<bool, ContestSegment>> visited_value_pair_list;
    visited_value_pair_list.reserve(segment_list.size());
    for (size_t i = 0; i < segment_list.size(); i++) {
      visited_value_pair_list.emplace_back(false, segment_list[i]);
    }
    std::map<ContestCoord, bool, CmpContestCoord> coord_visited_map;
    for (ContestCoord& key_coord : key_coord_list) {
      coord_visited_map[key_coord] = false;
    }

    ContestCoord root = key_coord_list.front();
    std::queue<ContestCoord> coord_queue = initQueue(root);
    while (!coord_queue.empty()) {
      ContestCoord coord = getFrontAndPop(coord_queue);
      if (exist(coord_visited_map, coord)) {
        coord_visited_map[coord] = true;
      }
      std::vector<ContestCoord> next_coord_list;
      for (size_t i = 0; i < visited_value_pair_list.size(); i++) {
        std::pair<bool, ContestSegment>& visited_value_pair = visited_value_pair_list[i];
        if (visited_value_pair.first == true) {
          continue;
        }
        ContestCoord& first_coord = visited_value_pair.second.get_first();
        ContestCoord& second_coord = visited_value_pair.second.get_second();
        if (coord == first_coord || coord == second_coord) {
          ContestCoord child_coord = (coord == first_coord ? second_coord : first_coord);
          next_coord_list.push_back(child_coord);
          visited_value_pair.first = true;
        }
      }
      addListToQueue(coord_queue, next_coord_list);
    }
    for (auto& [coord, visited] : coord_visited_map) {
      if (visited == false) {
        return false;
      }
    }
    return true;
  }

};  // namespace ieda_contest

}  // namespace ieda_contest
