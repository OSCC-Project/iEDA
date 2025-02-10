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
#include "builder.h"

namespace idb {

void IdbBuilder::buildNet()
{
  auto sort_net_points = [](IdbNet* net) {
    for (auto* wire : net->get_wire_list()->get_wire_list()) {
      for (auto* segment : wire->get_segment_list()) {
        if (segment->get_point_number() < 2) {
          continue;
        }

        auto& points = segment->get_point_list();
        std::sort(points.begin(), points.end(), [](IdbCoordinate<int>* a, IdbCoordinate<int>* b) {
          if (a->get_x() == b->get_x()) {
            return a->get_y() < b->get_y();
          } else if (a->get_y() == b->get_y()) {
            return a->get_x() < b->get_x();
          } else {
            return false;
          }
        });
      }
    }
  };

  checkNetPins();

  for (IdbNet* net : _def_service->get_design()->get_net_list()->get_net_list()) {
    buildNetFeatureCoord(net);
    buildPinFeatureCoord(net);
    sort_net_points(net);
  }
}

void IdbBuilder::checkNetPins()
{
  for (IdbNet* net : _def_service->get_design()->get_net_list()->get_net_list()) {
    if (net->get_instance_pin_list() != nullptr) {
      net->get_instance_pin_list()->checkPins();
    }

    if (net->get_io_pins() != nullptr) {
      net->get_io_pins()->checkPins();
    }
  }
}

IdbCoordinate<int32_t>* calcAvgCoord(const std::vector<IdbCoordinate<int32_t>*>& coord_list)
{
  if (coord_list.size() == 0) {
    return new IdbCoordinate<int32_t>(0, 0);
  }
  double avg_x = 0;
  double avg_y = 0;
  for (IdbCoordinate<int32_t>* coord : coord_list) {
    avg_x += coord->get_x();
    avg_y += coord->get_y();
  }
  avg_x /= coord_list.size();
  avg_y /= coord_list.size();

  return new IdbCoordinate<int32_t>((int32_t) (avg_x + 0.5), (int32_t) (avg_y + 0.5));
};

void IdbBuilder::buildNetFeatureCoord(IdbNet* net)
{
  std::vector<IdbCoordinate<int32_t>*> coord_list;

  auto* io_pins = net->get_io_pins();
  for (auto* io_pin : io_pins->get_pin_list()) {
    coord_list.push_back(io_pin->get_average_coordinate());
  }

  std::vector<IdbPin*>& pin_list = net->get_instance_pin_list()->get_pin_list();
  for (size_t i = 0; i < pin_list.size(); i++) {
    coord_list.push_back(pin_list[i]->get_average_coordinate());
  }

  IdbCoordinate<int32_t>* average_coordinate = calcAvgCoord(coord_list);

  net->set_average_coordinate(average_coordinate);
}

vector<IdbCoordinate<int32_t>> getCandidatePointList(IdbPin* idb_pin)
{
  vector<IdbCoordinate<int32_t>> point_list;
  /// there is only 1 routing layer in standard cell
  if (idb_pin->is_multi_layer()) {
    return point_list;
  }

  vector<IdbLayerShape*> layer_shape_list = idb_pin->get_port_box_list();

  /// find first routing layer
  IdbLayerShape* first_layer_shape = nullptr;
  for (IdbLayerShape* layer_shape : layer_shape_list) {
    if (layer_shape->get_layer()->is_routing()) {
      first_layer_shape = layer_shape;
      break;
    }
  }
  if (first_layer_shape == nullptr) {
    return point_list;
  }
  /// get track grid
  IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(first_layer_shape->get_layer());

  IdbTrackGrid* track_grid_prefer = routing_layer->get_prefer_track_grid();
  IdbTrackGrid* track_grid_nonprefer = routing_layer->get_nonprefer_track_grid();

  if (track_grid_prefer == nullptr || track_grid_nonprefer == nullptr) {
    return point_list;
  }

  int32_t start_x = 0;
  int32_t pitch_x = 0;
  int32_t start_y = 0;
  int32_t pitch_y = 0;

  if (track_grid_prefer->get_track()->is_track_direction_x()) {
    start_x = track_grid_prefer->get_track()->get_start();
    pitch_x = track_grid_prefer->get_track()->get_pitch();
    start_y = track_grid_nonprefer->get_track()->get_start();
    pitch_y = track_grid_nonprefer->get_track()->get_pitch();
  } else {
    start_x = track_grid_nonprefer->get_track()->get_start();
    pitch_x = track_grid_nonprefer->get_track()->get_pitch();
    start_y = track_grid_prefer->get_track()->get_start();
    pitch_y = track_grid_prefer->get_track()->get_pitch();
  }

  for (IdbRect* rect : first_layer_shape->get_rect_list()) {
    if (rect->get_low_x() < start_x || rect->get_low_y() < start_y) {
      continue;
    }

    int32_t low_index_x = (rect->get_low_x() - start_x) / pitch_x;
    int32_t high_index_x = (rect->get_high_x() - start_x) / pitch_x;
    int32_t low_index_y = (rect->get_low_y() - start_y) / pitch_y;
    int32_t high_index_y = (rect->get_high_y() - start_y) / pitch_y;

    if (low_index_x == high_index_x || low_index_y == high_index_y) {
      continue;
    }

    low_index_x = (low_index_x + 1) * pitch_x + start_x;
    high_index_x = high_index_x * pitch_x + start_x;

    low_index_y = (low_index_y + 1) * pitch_y + start_y;
    high_index_y = high_index_y * pitch_y + start_y;

    for (int i = low_index_x; i <= high_index_x; i += pitch_x) {
      for (int j = low_index_y; j <= high_index_y; j += pitch_y) {
        if (i < rect->get_low_x() || i > rect->get_high_x() || j < rect->get_low_y() || j > rect->get_high_y()) {
          std::cout << "Error pin grid coordinate, pin list empty." << std::endl;
          continue;
        }
        point_list.emplace_back(i, j);
      }
    }
  }

  return point_list;
}

int32_t calcManhattanDistance(IdbCoordinate<int32_t>& a, IdbCoordinate<int32_t>& b)
{
  return std::abs(a.get_x() - b.get_x()) + std::abs(a.get_y() - b.get_y());
}

void IdbBuilder::buildPinFeatureCoord(IdbNet* net)
{
  IdbCoordinate<int32_t>* net_avg_coord = net->get_average_coordinate();

  for (IdbPin* idb_pin : net->get_instance_pin_list()->get_pin_list()) {
    std::vector<IdbCoordinate<int32_t>> point_list = getCandidatePointList(idb_pin);

    size_t min_idx = 0;
    if (!point_list.empty()) {
      int32_t min_distance = INT32_MAX;

      for (size_t i = 0; i < point_list.size(); i++) {
        IdbCoordinate<int32_t>& point = point_list[i];
        int32_t curr_distance = calcManhattanDistance(*net_avg_coord, point);
        if (curr_distance < min_distance) {
          min_distance = curr_distance;
          min_idx = i;
        }
      }

      idb_pin->set_grid_coordinate(point_list[min_idx].get_x(), point_list[min_idx].get_y());
    }

    if (idb_pin->get_grid_coordinate()->get_x() == -1 || idb_pin->get_grid_coordinate()->get_y() == -1) {
      std::cout << "Error pin grid coordinate, pin =  " << idb_pin->get_pin_name() << std::endl;
    }
  }
}

}  // namespace idb
