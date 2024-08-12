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
#include "DRCEngine.hpp"

#include "GDSPlotter.hpp"
#include "Utility.hpp"

namespace irt {

// public

void DRCEngine::initInst()
{
  if (_de_instance == nullptr) {
    _de_instance = new DRCEngine();
  }
}

DRCEngine& DRCEngine::getInst()
{
  if (_de_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_de_instance;
}

void DRCEngine::destroyInst()
{
  if (_de_instance != nullptr) {
    delete _de_instance;
    _de_instance = nullptr;
  }
}

// function

std::vector<Violation> DRCEngine::getViolationList()
{
  return std::vector<Violation>();
}

// std::vector<Violation> DRCEngine::getViolationList(std::vector<idb::IdbLayerShape*>& env_shape_list,
//                                                      std::map<int32_t, std::vector<idb::IdbLayerShape*>>& net_pin_shape_map,
//                                                      std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>>& net_wire_via_map,
//                                                      std::string stage)
// {
//   std::set<idrc::ViolationEnumType> check_select;
//   if (stage == "TA") {
//     check_select.insert(idrc::ViolationEnumType::kShort);
//   } else if (stage == "DR") {
//     check_select.insert(idrc::ViolationEnumType::kShort);
//     check_select.insert(idrc::ViolationEnumType::kDefaultSpacing);
//   } else {
//     RTLOG.error(Loc::current(), "Currently not supporting other stages");
//   }
//   /**
//    * env_shape_list 存储 obstacle obs pin_shape
//    * net_idb_segment_map 存储 wire via patch
//    */
//   ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
//   std::map<std::string, int32_t>& routing_layer_name_to_idx_map = RTDM.getDatabase().get_routing_layer_name_to_idx_map();
//   std::map<std::string, int32_t>& cut_layer_name_to_idx_map = RTDM.getDatabase().get_cut_layer_name_to_idx_map();

//   std::vector<Violation> violation_list;
//   idrc::DrcApi drc_api;
//   drc_api.init();
//   for (auto& [type, idrc_violation_list] : drc_api.check(env_shape_list, net_pin_shape_map, net_wire_via_map, check_select)) {
//     for (idrc::DrcViolation* idrc_violation : idrc_violation_list) {
//       // self的drc违例先过滤
//       if (idrc_violation->get_net_ids().size() < 2) {
//         continue;
//       }
//       // 由于pin_shape之间的drc违例存在，第一布线层的drc违例先过滤
//       idb::IdbLayer* idb_layer = idrc_violation->get_layer();
//       if (idb_layer->is_routing()) {
//         if (routing_layer_name_to_idx_map[idb_layer->get_name()] == 0) {
//           continue;
//         }
//       }
//       EXTLayerRect ext_layer_rect;
//       if (idrc_violation->is_rect()) {
//         idrc::DrcViolationRect* idrc_violation_rect = static_cast<idrc::DrcViolationRect*>(idrc_violation);
//         ext_layer_rect.set_real_ll(idrc_violation_rect->get_llx(), idrc_violation_rect->get_lly());
//         ext_layer_rect.set_real_ur(idrc_violation_rect->get_urx(), idrc_violation_rect->get_ury());
//       } else {
//         RTLOG.error(Loc::current(), "Type not supported!");
//       }
//       ext_layer_rect.set_grid_rect(RTUTIL.getClosedGCellGridRect(ext_layer_rect.get_real_rect(), gcell_axis));
//       ext_layer_rect.set_layer_idx(idb_layer->is_routing() ? routing_layer_name_to_idx_map[idb_layer->get_name()]
//                                                            : cut_layer_name_to_idx_map[idb_layer->get_name()]);

//       Violation violation;
//       violation.set_violation_shape(ext_layer_rect);
//       violation.set_is_routing(idb_layer->is_routing());
//       violation.set_violation_net_set(idrc_violation->get_net_ids());
//       violation_list.push_back(violation);
//     }
//   }
//   return violation_list;
// }

// private

DRCEngine* DRCEngine::_de_instance = nullptr;

}  // namespace irt
