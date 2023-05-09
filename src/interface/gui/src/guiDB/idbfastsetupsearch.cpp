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
#include "idbfastsetup.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void IdbSpeedUpSetup::search(std::string search_text) {
  if (search_text.empty()) {
    return;
  }

  initSearchbox();

  highLightCoordinate(search_text);
  highLightNet(search_text);
  highLightInstance(search_text);

  viewSearchBox();
}

bool IdbSpeedUpSetup::highLightCoordinate(std::string name) {
  GuiString gui_string;

  if (gui_string.isCoordinateInDef(name)) {
    std::pair<int, int> coordinate = gui_string.getCoordinateInDef(name);

    IdbRect* rect =
        new IdbRect(coordinate.first - 50, coordinate.second - 50, coordinate.first + 50, coordinate.second + 50);
    _search_item->add_rect(_transform.db_to_guidb_rect(rect));

    delete rect;
    rect = nullptr;

    return true;
  }

  if (gui_string.isCoordinateInGui(name)) {
    std::pair<qreal, qreal> coordinate = gui_string.getCoordinateInGui(name);
    _search_item->add_rect(QRectF(coordinate.first - 1, _transform.guidb_reverse_y_value(coordinate.second) - 1, 2, 2));

    return true;
  }

  return false;
}

bool IdbSpeedUpSetup::highLightNet(std::string name) {
  IdbNet* net = _design->get_net_list()->find_net(name);
  return highLightNet(net);
}

bool IdbSpeedUpSetup::highLightNetPin(IdbNet* net) {
  if (net == nullptr) {
    return false;
  }

  for (auto pin : net->get_instance_pin_list()->get_pin_list()) {
    /// show the pin shape
    if (pin != nullptr && pin->get_term()->is_instance_pin()) {
      for (IdbLayerShape* layer_shape : pin->get_port_box_list()) {
        for (IdbRect* rect : layer_shape->get_rect_list()) {
          _search_item->add_pin_rect(_transform.db_to_guidb_rect(rect));
        }
      }
    }
    /// show the fly line
    _search_item->add_point(_transform.db_to_guidb(pin->get_grid_coordinate()->get_x()),
                            _transform.db_to_guidb_rotate(pin->get_grid_coordinate()->get_y()));
  }

  /// set average coordinate
  _search_item->set_average_pin_coord(QPointF(_transform.db_to_guidb(net->get_average_coordinate()->get_x()),
                                              _transform.db_to_guidb_rotate(net->get_average_coordinate()->get_y())));

  return true;
}

bool IdbSpeedUpSetup::highLightNetWire(IdbNet* net) {
  if (net == nullptr) {
    return false;
  }

  for (IdbRegularWire* wire : net->get_wire_list()->get_wire_list()) {
    for (IdbRegularWireSegment* segment : wire->get_segment_list()) {
      if (segment == nullptr || segment->is_rect()) {
        continue;
      } else {
        /// find gui wire list ptr
        if (segment->get_point_number() >= 2)  // ensure the point number >= 2
        {
          if (segment->get_layer() == nullptr) {
            std::cout << "Error...createNetPoints : Layer not exist :  " << std::endl;
            return false;
          }

          IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(segment->get_layer());
          int32_t routing_width          = routing_layer->get_width();

          IdbCoordinate<int32_t>* point_1 = segment->get_point_start();
          IdbCoordinate<int32_t>* point_2 = segment->get_point_second();

          int32_t ll_x = 0;
          int32_t ll_y = 0;
          int32_t ur_x = 0;
          int32_t ur_y = 0;
          if (point_1->get_y() == point_2->get_y()) {
            // horizontal
            ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width / 2;
            ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width / 2;
            ur_x = std::max(point_1->get_x(), point_2->get_x()) + routing_width / 2;
            ur_y = ll_y + routing_width;
          } else if (point_1->get_x() == point_2->get_x()) {
            // vertical
            ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width / 2;
            ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width / 2;
            ur_x = ll_x + routing_width;
            ur_y = std::max(point_1->get_y(), point_2->get_y()) + routing_width / 2;
          } else {
            // only support horizontal & vertical direction
            std::cout << "Error...Regular segment only support horizontal & "
                         "vertical direction... "
                      << segment->get_layer()->get_name() << std::endl;
          }

          IdbRect* rect = new IdbRect(ll_x, ll_y, ur_x, ur_y);
          _search_item->add_rect(_transform.db_to_guidb_rect(rect));

          delete rect;
          rect = nullptr;
        }
      }
    }
  }

  return true;
}

bool IdbSpeedUpSetup::highLightNetFlyline(IdbNet* net) {
  if (net == nullptr) {
    return false;
  }

  return true;
}

bool IdbSpeedUpSetup::highLightNet(IdbNet* net) {
  if (net == nullptr) {
    return false;
  }

  return highLightNetPin(net) && highLightNetWire(net);
}

bool IdbSpeedUpSetup::highLightInstance(std::string name) {
  IdbInstance* instance = _design->get_instance_list()->find_instance(name);
  if (instance == nullptr) {
    return false;
  }

  IdbRect* bounding_box = instance->get_bounding_box();
  _search_item->add_rect(_transform.db_to_guidb_rect(bounding_box));

  return true;
}

bool IdbSpeedUpSetup::updateSearchNet(IdbNet* net) {
  initSearchbox();
  bool b_result = highLightNetWire(net);
  viewSearchBox();

  return b_result;
}
