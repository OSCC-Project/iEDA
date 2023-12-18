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
/**
 * @File Name: gridmap.h
 * @Brief : gridmap to manage 3d nodes
 * @Author : GuoFan (guofan@ustc.edu)
 * @Version : 1.0
 * @Creat Date : 2023-09-27
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <array>
#include <iostream>
#include <vector>

#include "contest_dm.h"
#include "idm.h"
#include "mapnode.hh"

namespace ieda_contest {

namespace gridmap {

template <typename T>
requires std::is_base_of_v<Node<3>, T> class Map
{
 public:
  typedef std::vector<std::vector<std::vector<T>>> container;
  typedef std::vector<std::vector<T>> container_layer;
  typedef std::vector<T> container_line;
  typedef bgm::point<int, 3, bg::cs::cartesian> point;

 public:
  Map() : _map_size(0, 0, 0), _grid_size(1, 1, 1) {}

  Map(ContestDataManager* data_manager, int x, int y, int z, int sx = 1, int sy = 1, int sz = 1)
      : _data_manager(data_manager),
        _map_size(x, y, z),
        _grid_size(sx, sy, sz),
        _map(container(x, container_layer(y, container_line(z, T()))))
  {
    setNodeData();
  }

  Map(std::vector<std::vector<std::vector<double>>> map_data, const point& grid_size = point(1, 1, 1))
  {
    _grid_size = grid_size;

    int x = map_data.size();
    if (x > 0) {
      int y = map_data[0].size();
      if (y > 0) {
        int z = map_data[0][0].size();

        _map_size.set<0>(x);
        _map_size.set<1>(x);
        _map_size.set<2>(x);
        _map = container(x, container_layer(y, container_line(z, T())));

        setNodeData(map_data);
      }
    }
  }

  /// @brief init map
  /// @param map_data map data
  void setNodeData(std::vector<std::vector<std::vector<double>>> map_data = {})
  {
    for (int i = 0; i < _map_size.get<0>(); ++i) {
      for (int j = 0; j < _map_size.get<1>(); ++j) {
        for (int k = 0; k < _map_size.get<2>(); ++k) {
          bg::set<0>(_map[i][j][k].set_position(), i);
          bg::set<1>(_map[i][j][k].set_position(), j);
          bg::set<2>(_map[i][j][k].set_position(), k);
          if (map_data.size() > 0) {
            _map[i][j][k].set_supply_resource_cnt(map_data[i][j][k]);
          }
        }
      }
    }
  }

  /// @brief fill demand resource count
  /// @param cnt demand resource count
  void fillDemandResourceCnt(int cnt)
  {
    for (auto& layer : _map) {
      for (auto& line : layer) {
        for (auto& item : line) {
          item.set_demand_resource_cnt(cnt);
        }
      }
    }
  }

  /// @brief fill supply resource count
  /// @param cnt supply resource count
  void fillSupplyResourceCnt(int cnt)
  {
    for (auto& layer : _map) {
      for (auto& line : layer) {
        for (auto& item : line) {
          item.set_supply_resource_cnt(cnt);
        }
      }
    }
  }

  /// @brief fill supply resource count in layer
  /// @param cnt supply resource count
  /// @param layer_idx layer index
  void fillLayerSupplyResourceCnt(int cnt, int layer_idx)
  {
    for (int i = 0; i < static_cast<int>(_map.size()); ++i) {
      for (int j = 0; j < static_cast<int>(_map[0].size()); ++j) {
        _map[i][j][layer_idx].set_supply_resource_cnt(cnt);
      }
    }
  }

  const point& get_map_size() { return _map_size; }

  T& operator[](const point& p)
  {
    if (p.get<0>() >= _map_size.get<0>() || p.get<1>() >= _map_size.get<1>() || p.get<2>() >= _map_size.get<2>() || p.get<0>() < 0
        || p.get<1>() < 0 || p.get<2>() < 0) {
      std::cout << "[gridmap] out of range: " << p.get<0>() << " " << p.get<1>() << " " << p.get<2>() << std::endl;
    }
    return _map[p.get<0>()][p.get<1>()][p.get<2>()];
  }

  /// @brief find 8 neighbours around p: up, down, left, right, forward, backward
  /// @param p determine who's neighbour
  /// @return all avaliable neighbours
  std::vector<T*> getNeighbours(const point& p)
  {
    std::vector<T*> neighbours;
    auto direction = _data_manager->get_database()->get_layer_idx_to_direction_map()[p.get<2>()];

    for (int i = -1; i <= 1; ++i) {
      for (int j = -1; j <= 1; ++j) {
        for (int k = -1; k <= 1; ++k) {
          int x = p.get<0>() + i;
          int y = p.get<1>() + j;
          int z = p.get<2>() + k;
          if (((i == 0 && j == 0 && k != 0) || (i == 0 && j != 0 && k == 0) || (i != 0 && j == 0 && k == 0)) && x >= 0 && y >= 0 && z >= 0
              && x < _map_size.get<0>() && y < _map_size.get<1>() && z < _map_size.get<2>()
              && ((direction == idb::IdbLayerDirection::kHorizontal && j == 0)
                  || (direction == idb::IdbLayerDirection::kVertical && i == 0))) {
            neighbours.push_back(&_map[x][y][z]);
          }
        }
      }
    }

    return neighbours;
  }

  std::vector<T*> getNeighbours(const T& n) { return getNeighbours(n.get_position()); }

  /// @brief should be called before every path finding to remove temporary data
  void refreshMap()
  {
    for (auto& layer : _map) {
      for (auto& line : layer) {
        for (auto& item : line) {
          item.refresh();
        }
      }
    }
  }

 private:
  ContestDataManager* _data_manager = nullptr;

  point _map_size;
  point _grid_size;
  container _map;
};

}  // namespace gridmap

}  // namespace ieda_contest
