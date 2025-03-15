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

#include "DRCInterface.hpp"

#include "DRCBox.hpp"
#include "DRCModel.hpp"
#include "DataManager.hpp"
#include "Module.hpp"
#include "idm.h"

#define NET_ID_ENVIRONMENT -1
#define NET_ID_PDN -3
#define NET_ID_VDD -4
#define NET_ID_VSS -5
#define NET_ID_OBS -6  //: iEDA/src/database/interaction/RT_DRC/DRCViolationType.h
namespace idrc {

// public

DRCInterface& DRCInterface::getInst()
{
  if (_drc_interface_instance == nullptr) {
    _drc_interface_instance = new DRCInterface();
  }
  return *_drc_interface_instance;
}

void DRCInterface::destroyInst()
{
  if (_drc_interface_instance != nullptr) {
    delete _drc_interface_instance;
    _drc_interface_instance = nullptr;
  }
}

#if 1  // 外部调用DRC的API

#if 1  // iDRC

void DRCInterface::initDRC()
{
  // 初始化规则
}

void DRCInterface::checkDef()
{
  std::vector<ids::Shape> ids_shape_list;
  IdbDesign* idb_design = dmInst->get_idb_design();

  // function defs
  auto initDataFromRect = [&](int32_t net_idx, int32_t layer_idx, bool is_routing, IdbRect* rect) {
    ids::Shape ids_shape;
    ids_shape.net_idx = net_idx;
    ids_shape.layer_idx = layer_idx;
    ids_shape.is_routing = is_routing;
    ids_shape.ll_x = rect->get_low_x();
    ids_shape.ll_y = rect->get_low_y();
    ids_shape.ur_x = rect->get_high_x();
    ids_shape.ur_y = rect->get_high_y();
    ids_shape_list.push_back(ids_shape);
  };

  auto initDataFromShape = [&](IdbLayerShape* idb_shape, int net_idx) {
    if (idb_shape == nullptr) {
      return;
    }

    IdbLayout* idb_layout = dmInst->get_idb_layout();
    IdbLayers* idb_layers = idb_layout->get_layers();
    /// shape must be on the above of bottom routing layer
    if (false == idb_layers->is_pr_layer(idb_shape->get_layer())) {
      return;
    }

    IdbLayer* layer = idb_shape->get_layer();
    for (IdbRect* rect : idb_shape->get_rect_list()) {
      initDataFromRect(net_idx, layer->get_id(), layer->is_routing(), rect);
    }
  };

  auto initDataFromPin = [&](IdbPin* idb_pin, int default_id) {
    int net_id = default_id;
    if (idb_pin->get_net() != nullptr) {
      net_id = idb_pin->get_net()->get_id();
    } else {
      IdbTerm* idb_term = idb_pin->get_term();
      net_id = idb_term->is_power() ? NET_ID_VDD : (idb_term->is_ground() ? NET_ID_VSS : default_id);
    }

    for (IdbLayerShape* layer_shape : idb_pin->get_port_box_list()) {
      initDataFromShape(layer_shape, net_id);
    }
  };

  auto initDataFromVia = [&](IdbVia* idb_via, int net_id) {
    /// cut
    IdbLayerShape cut_layer_shape = idb_via->get_cut_layer_shape();
    initDataFromShape(&cut_layer_shape, net_id);
    /// bottom
    IdbLayerShape bottom_layer_shape = idb_via->get_bottom_layer_shape();
    initDataFromShape(&bottom_layer_shape, net_id);
    /// top
    IdbLayerShape top_layer_shape = idb_via->get_top_layer_shape();
    initDataFromShape(&top_layer_shape, net_id);
  };

  auto initDataFromPoints
      = [&](idb::IdbCoordinate<int>* point_1, idb::IdbCoordinate<int>* point_2, int routing_width, idb::IdbLayer* layer, int net_id, bool b_pdn) {
          /// calculate rectangle by two points
          int llx = 0, lly = 0, urx = 0, ury = 0;
          int extend_size = b_pdn ? 0 : routing_width / 2;
          if (point_1->get_y() == point_2->get_y()) {
            // horizontal
            llx = std::min(point_1->get_x(), point_2->get_x()) - extend_size;
            lly = point_1->get_y() - routing_width / 2;
            urx = std::max(point_1->get_x(), point_2->get_x()) + extend_size;
            ury = lly + routing_width;
          } else if (point_1->get_x() == point_2->get_x()) {
            // vertical
            llx = point_1->get_x() - routing_width / 2;
            lly = std::min(point_1->get_y(), point_2->get_y()) - extend_size;
            urx = llx + routing_width;
            ury = std::max(point_1->get_y(), point_2->get_y()) + extend_size;
          }

          initDataFromRect(net_id, layer->get_id(), layer->is_routing(), new IdbRect(llx, lly, urx, ury));
        };

  // Pin DATA init
  {
    IdbPins* idb_io_pins = idb_design->get_io_pin_list();
    for (IdbPin* idb_io_pin : idb_io_pins->get_pin_list()) {
      if (idb_io_pin != nullptr && idb_io_pin->get_term()->is_placed()) {
        initDataFromPin(idb_io_pin, -1);
      }
    }
  }
  // Instances DATA init
  {
    uint64_t number = 0;
    for (IdbInstance* idb_inst : idb_design->get_instance_list()->get_instance_list()) {
      if (idb_inst == nullptr || idb_inst->get_cell_master() == nullptr) {
        continue;
      }
      /// obs
      for (IdbPin* idb_pin : idb_inst->get_pin_list()->get_pin_list()) {
        initDataFromPin(idb_pin, -1);
      }
      for (IdbLayerShape* idb_obs : idb_inst->get_obs_box_list()) {
        initDataFromShape(idb_obs, NET_ID_OBS - number);
      }

      number++;
    }
  }
  // PDN data init
  {
    uint64_t number = 0;
    for (IdbSpecialNet* idb_special_net : idb_design->get_special_net_list()->get_net_list()) {
      int net_id = idb_special_net->is_vdd() ? NET_ID_VDD : NET_ID_VSS;
      for (IdbSpecialWire* idb_special_wire : idb_special_net->get_wire_list()->get_wire_list()) {
        for (IdbSpecialWireSegment* idb_segment : idb_special_wire->get_segment_list()) {
          /// add wire
          if (idb_segment->get_point_list().size() >= 2) {
            /// get routing width
            IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(idb_segment->get_layer());
            int32_t routing_width = idb_segment->get_route_width() == 0 ? routing_layer->get_width() : idb_segment->get_route_width();
            /// calculate rectangle by two points
            IdbCoordinate<int32_t>* point_1 = idb_segment->get_point_start();
            IdbCoordinate<int32_t>* point_2 = idb_segment->get_point_second();

            initDataFromPoints(point_1, point_2, routing_width, idb_segment->get_layer(), net_id, true);
          }
          /// vias
          if (idb_segment->is_via()) {
            /// add via
            initDataFromVia(idb_segment->get_via(), net_id);
          }
          number++;
        }
      }
    }
  }
  // Net data init
  {
    for (IdbNet* idb_net : idb_design->get_net_list()->get_net_list()) {
      for (IdbRegularWire* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
        for (IdbRegularWireSegment* idb_segment : idb_wire->get_segment_list()) {
          if (idb_segment->get_point_number() >= 2) {
            /// get routing width
            IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(idb_segment->get_layer());
            int32_t routing_width = routing_layer->get_width();

            /// calculate rectangle by two points
            IdbCoordinate<int32_t>* point_1 = idb_segment->get_point_start();
            IdbCoordinate<int32_t>* point_2 = idb_segment->get_point_second();

            initDataFromPoints(point_1, point_2, routing_width, idb_segment->get_layer(), idb_net->get_id(), false);
          }
          /// via
          if (idb_segment->is_via()) {
            for (IdbVia* idb_via : idb_segment->get_via_list()) {
              initDataFromVia(idb_via, idb_net->get_id());
            }
          }
          /// patch
          if (idb_segment->is_rect()) {
            IdbCoordinate<int32_t>* coordinate = idb_segment->get_point_start();
            IdbRect* rect_delta = idb_segment->get_delta_rect();
            IdbRect* rect = new IdbRect(rect_delta);
            rect->moveByStep(coordinate->get_x(), coordinate->get_y());
            bool is_routing = true;
            initDataFromRect(idb_net->get_id(),idb_segment->get_layer()->get_id(),is_routing,rect);
            delete rect;
          }
        }
      }
    }
  }
  // finish
  getViolationList(ids_shape_list);
}

void DRCInterface::destroyDRC()
{
  // 销毁规则
}

std::vector<ids::Violation> DRCInterface::getViolationList(std::vector<ids::Shape> ids_shape_list)
{
  DRCModel drc_model = initDRCModel(ids_shape_list);
  DRCMOD.check(drc_model);
  return getViolationList(drc_model);
}

DRCModel DRCInterface::initDRCModel(std::vector<ids::Shape>& ids_shape_list)
{
  DRCModel drc_model;
  for (ids::Shape& ids_shape : ids_shape_list) {
    DRCShape drc_shape;
    drc_shape.set_net_idx(ids_shape.net_idx);
    drc_shape.set_ll(ids_shape.ll_x, ids_shape.ll_y);
    drc_shape.set_ur(ids_shape.ur_x, ids_shape.ur_y);
    drc_shape.set_layer_idx(ids_shape.layer_idx);
    drc_shape.set_is_routing(ids_shape.is_routing);
    drc_model.get_drc_shape_list().push_back(drc_shape);
  }
  return drc_model;
}

std::vector<ids::Violation> DRCInterface::getViolationList(DRCModel& drc_model)
{
  std::vector<ids::Violation> ids_violation_list;
  for (DRCBox& drc_box : drc_model.get_drc_box_list()) {
    for (Violation& violation : drc_box.get_violation_list()) {
      ids::Violation ids_violation;
      ids_violation.violation_type = GetViolationTypeName()(violation.get_violation_type());
      ids_violation.ll_x = violation.get_ll_x();
      ids_violation.ll_y = violation.get_ll_y();
      ids_violation.ur_x = violation.get_ur_x();
      ids_violation.ur_y = violation.get_ur_y();
      ids_violation.layer_idx = violation.get_layer_idx();
      ids_violation.is_routing = violation.get_is_routing();
      ids_violation.violation_net_set = violation.get_violation_net_set();
      ids_violation.required_size = violation.get_required_size();
      ids_violation_list.push_back(ids_violation);
    }
  }
  return ids_violation_list;
}

#endif

#endif

// private

DRCInterface* DRCInterface::_drc_interface_instance = nullptr;

}  // namespace idrc
