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

#include "idrc_api.h"

#include "idm.h"
#include "idrc.h"
#include "idrc_config.h"
#include "idrc_data.h"
#include "idrc_dm.h"
#include "tech_rules.h"

namespace idrc {

void DrcApi::init(std::string config)
{
  DrcConfigInst->init(config);

  /// tech rule is a singleton pattern, so it must be inited if drc starts
  DrcTechRuleInst->init();
}

void DrcApi::exit()
{
  DrcTechRuleInst->destroyInst();
}

std::map<ViolationEnumType, std::vector<DrcViolation*>> DrcApi::check(std::vector<idb::IdbLayerShape*>& env_shape_list,
                                                                      std::map<int, std::vector<idb::IdbLayerShape*>>& pin_data,
                                                                      std::map<int, std::vector<idb::IdbRegularWireSegment*>>& routing_data,
                                                                      std::set<ViolationEnumType> check_select)
{
  DrcManager drc_manager;

  auto* data_manager = drc_manager.get_data_manager();
  // auto* rule_manager = drc_manager.get_rule_manager();
  auto* condition_manager = drc_manager.get_condition_manager();
  auto* violation_manager = drc_manager.get_violation_manager();
  if (data_manager == nullptr /*|| rule_manager == nullptr */ || condition_manager == nullptr || violation_manager == nullptr) {
    return {};
  }

  condition_manager->set_check_select(check_select);

  /// set drc rule stratagy by rt paramter
  /// tbd
  // rule_manager->get_stratagy()->set_stratagy_type(DrcStratagyType::kCheckFast);

  data_manager->set_env_shapes(&env_shape_list);
  data_manager->set_pin_data(&pin_data);
  data_manager->set_routing_data(&routing_data);

  auto check_type = (env_shape_list.size() + pin_data.size() + routing_data.size()) > 0 ? DrcCheckerType::kRT : DrcCheckerType::kDef;

  condition_manager->set_check_type(check_type);

#ifdef DEBUG_IDRC_API
  if (DrcCheckerType::kDef == check_type) {
    std::cout << "idrc : check def" << std::endl;
  }
#endif

#ifdef DEBUG_IDRC_API
  ieda::Stats stats_engine;
#endif
  drc_manager.dataInit(check_type);
#ifdef DEBUG_IDRC_API
  if (DrcCheckerType::kDef == check_type) {
    std::cout << "idrc : engine start"
              << " runtime = " << stats_engine.elapsedRunTime() << " memory = " << stats_engine.memoryDelta() << std::endl;
  }
#endif

#ifdef DEBUG_IDRC_API
  ieda::Stats stats_build_condition;
#endif
  drc_manager.dataOperate();
#ifdef DEBUG_IDRC_API
  if (DrcCheckerType::kDef == check_type) {
    std::cout << "idrc : build condition"
              << " runtime = " << stats_build_condition.elapsedRunTime() << " memory = " << stats_build_condition.memoryDelta()
              << std::endl;
  }
#endif

#ifdef DEBUG_IDRC_API
  ieda::Stats stats_check;
#endif
  drc_manager.dataCheck();

#ifdef DEBUG_IDRC_API
  if (DrcCheckerType::kDef == check_type) {
    std::cout << "idrc : check"
              << " runtime = " << stats_check.elapsedRunTime() << " memory = " << stats_check.memoryDelta() << std::endl;
  }
#endif

  return violation_manager->get_violation_map(drc_manager.get_engine()->get_engine_manager());
}
/**
 * check DRC violation for DEF file
 * initialize data from idb
 */
std::map<ViolationEnumType, std::vector<DrcViolation*>> DrcApi::checkDef()
{
  std::vector<idb::IdbLayerShape*> env_shape_list;
  std::map<int, std::vector<idb::IdbLayerShape*>> pin_data;
  std::map<int, std::vector<idb::IdbRegularWireSegment*>> routing_data;
  return check(env_shape_list, pin_data, routing_data);
}

void plotGDS(std::string gds_file, std::map<int32_t, std::map<int32_t, std::vector<ieda_solver::GeometryRect>>>& layer_type_rect_list_map)
{
  std::ofstream gds_stream(gds_file);
  if (gds_stream.is_open()) {
    gds_stream << "HEADER 600" << std::endl;
    gds_stream << "BGNLIB" << std::endl;
    gds_stream << "LIBNAME GDSLib" << std::endl;
    gds_stream << "UNITS 0.001 1e-9" << std::endl;
    gds_stream << "BGNSTR" << std::endl;
    gds_stream << "STRNAME top" << std::endl;

    for (auto& [layer_idx, type_rect_list_map] : layer_type_rect_list_map) {
      for (auto& [type, rect_list] : type_rect_list_map) {
        for (auto& rect : rect_list) {
          int32_t lb_x = ieda_solver::lowLeftX(rect);
          int32_t rt_x = ieda_solver::upRightX(rect);
          int32_t lb_y = ieda_solver::lowLeftY(rect);
          int32_t rt_y = ieda_solver::upRightY(rect);

          gds_stream << "BOUNDARY" << std::endl;
          gds_stream << "LAYER " << layer_idx << std::endl;
          gds_stream << "DATATYPE " << int32_t(type) << std::endl;
          gds_stream << "XY" << std::endl;
          gds_stream << lb_x << " : " << lb_y << std::endl;
          gds_stream << rt_x << " : " << lb_y << std::endl;
          gds_stream << rt_x << " : " << rt_y << std::endl;
          gds_stream << lb_x << " : " << rt_y << std::endl;
          gds_stream << lb_x << " : " << lb_y << std::endl;
          gds_stream << "ENDEL" << std::endl;
        }
      }
    }
    gds_stream << "ENDSTR" << std::endl;
    gds_stream << "ENDLIB" << std::endl;
    gds_stream.close();
    std::cout << "[Info] Result has been written to '" << gds_file << "'!" << std::endl;
  } else {
    std::cout << "[Error] Failed to open gds file '" << gds_file << "'!" << std::endl;
  }
}

void DrcApi::diagnosis(std::string third_json_file, std::string idrc_json_file, std::string output_dir)
{
  std::map<std::string, ViolationEnumType> third_name_to_type_map{{"SHORT", ViolationEnumType::kShort},
                                                                  {"SPACING", ViolationEnumType::kPRLSpacing},
                                                                  {"MINSTEP", ViolationEnumType::kMinStep},
                                                                  {"EndOfLine", ViolationEnumType::kEOL},
                                                                  {"MINHOLE", ViolationEnumType::kAreaEnclosed}};
  std::map<std::string, ViolationEnumType> idrc_name_to_type_map{{"Corner Fill", ViolationEnumType::kCornerFill},
                                                                 {"Default Spacing", ViolationEnumType::kDefaultSpacing},
                                                                 {"Enclosed Area", ViolationEnumType::kAreaEnclosed},
                                                                 {"JogToJog Spacing", ViolationEnumType::kJogToJog},
                                                                 {"Metal EOL Spacing", ViolationEnumType::kEOL},
                                                                 {"Metal Notch Spacing", ViolationEnumType::kNotch},
                                                                 {"Metal Parallel Run Length Spacing", ViolationEnumType::kPRLSpacing},
                                                                 {"Metal Short", ViolationEnumType::kShort},
                                                                 {"MinStep", ViolationEnumType::kMinStep},
                                                                 {"Minimum Area", ViolationEnumType::kArea}};
  std::map<std::string, int32_t> layer_name_to_id_map{{"M1", 1}, {"M2", 2}, {"M3", 3}, {"M4", 4}, {"M5", 5},
                                                      {"M6", 6}, {"M7", 7}, {"M8", 8}, {"M9", 9}};

  std::string json_entry_key = "type_sorted_tech_DRCs_list";
  std::string json_drc_list_key = "tech_DRCs_list";

  auto parse_json = [&](std::string& file_path, std::map<int32_t, std::map<int32_t, std::vector<ieda_solver::GeometryRect>>>& result,
                        std::map<std::string, ViolationEnumType>& name_to_type_map, int32_t scale) {
    std::cout << "idrc : parse json: " << file_path << std::endl;
    std::ifstream json_stream(file_path);
    json json_data;
    json_stream >> json_data;

    for (auto& entry : json_data[json_entry_key]) {
      auto type = name_to_type_map[entry["type"]];
      for (auto& violation_item : entry[json_drc_list_key]) {
        auto& violation = violation_item["tech_DRCs"];
        auto layer = layer_name_to_id_map[violation["layer"]];
        double origin_llx = violation["llx"];
        double origin_lly = violation["lly"];
        double origin_urx = violation["urx"];
        double origin_ury = violation["ury"];
        int32_t llx = static_cast<int32_t>(origin_llx * scale);
        int32_t lly = static_cast<int32_t>(origin_lly * scale);
        int32_t urx = static_cast<int32_t>(origin_urx * scale);
        int32_t ury = static_cast<int32_t>(origin_ury * scale);
        result[layer][(int32_t) type].emplace_back(llx, lly, urx, ury);
      }
    }
  };

  std::map<int32_t, std::map<int32_t, std::vector<ieda_solver::GeometryRect>>> third_layer_type_rect_list_map;
  std::map<int32_t, std::map<int32_t, std::vector<ieda_solver::GeometryRect>>> idrc_layer_type_rect_list_map;

  parse_json(third_json_file, third_layer_type_rect_list_map, third_name_to_type_map, 2000);
  parse_json(idrc_json_file, idrc_layer_type_rect_list_map, idrc_name_to_type_map, 1);

  std::map<int32_t, std::map<int32_t, std::vector<ieda_solver::GeometryRect>>> third_diff_idrc_layer_type_rect_list_map;
  std::map<int32_t, std::map<int32_t, std::vector<ieda_solver::GeometryRect>>> idrc_diff_third_layer_type_rect_list_map;
  {
    auto create_set = [](std::map<int32_t, std::map<int32_t, std::vector<ieda_solver::GeometryRect>>>& input,
                         std::map<int32_t, std::map<int32_t, ieda_solver::GeometryPolygonSet>>& result) {
      std::cout << "idrc : create polygon set" << std::endl;
      for (auto& [layer, violation_map] : input) {
        for (auto& [type, violation_list] : violation_map) {
          for (auto& rect : violation_list) {
            result[layer][type] += rect;
          }
        }
      }
    };

    auto diff_two_map = [](std::map<int32_t, std::map<int32_t, ieda_solver::GeometryPolygonSet>>& map1,
                           std::map<int32_t, std::map<int32_t, ieda_solver::GeometryPolygonSet>>& map2,
                           std::map<int32_t, std::map<int32_t, std::vector<ieda_solver::GeometryRect>>>& result) {
      std::cout << "idrc : boolean operations" << std::endl;
      for (auto& [layer, violation_map] : map1) {
        for (auto& [type, polyset] : violation_map) {
          auto& other_polyset = map2[layer][type];
          auto diff_set = polyset - other_polyset;
          ieda_solver::getDefaultRectangles(result[layer][type], diff_set);
        }
      }
    };

    std::map<int32_t, std::map<int32_t, ieda_solver::GeometryPolygonSet>> third_layer_type_polyset_map;
    std::map<int32_t, std::map<int32_t, ieda_solver::GeometryPolygonSet>> idrc_layer_type_polyset_map;

    create_set(third_layer_type_rect_list_map, third_layer_type_polyset_map);
    create_set(idrc_layer_type_rect_list_map, idrc_layer_type_polyset_map);

    diff_two_map(third_layer_type_polyset_map, idrc_layer_type_polyset_map, third_diff_idrc_layer_type_rect_list_map);
    diff_two_map(idrc_layer_type_polyset_map, third_layer_type_polyset_map, idrc_diff_third_layer_type_rect_list_map);
  }

  std::cout << "idrc : output GDS file" << std::endl;

  std::string third_gds_file = output_dir + "/third.gds";
  std::string idrc_gds_file = output_dir + "/idrc.gds";
  std::string third_diff_idrc_gds_file = output_dir + "/third_diff_idrc.gds";
  std::string idrc_diff_third_gds_file = output_dir + "/idrc_diff_third.gds";

  plotGDS(third_gds_file, third_layer_type_rect_list_map);
  plotGDS(idrc_gds_file, idrc_layer_type_rect_list_map);
  plotGDS(third_diff_idrc_gds_file, third_diff_idrc_layer_type_rect_list_map);
  plotGDS(idrc_diff_third_gds_file, idrc_diff_third_layer_type_rect_list_map);

  std::cout << "idrc : combine two drc data" << std::endl;

  std::string combined_gds_file = output_dir + "/combined.gds";
  std::map<int32_t, std::map<int32_t, std::vector<ieda_solver::GeometryRect>>> combined_layer_type_rect_list_map;
  for (auto& [layer, violation_map] : idrc_layer_type_rect_list_map) {
    for (auto& [type, violation_list] : violation_map) {
      int new_type = type * 10 + 1;
      combined_layer_type_rect_list_map[layer][new_type].insert(combined_layer_type_rect_list_map[layer][new_type].end(),
                                                                violation_list.begin(), violation_list.end());
    }
  }
  for (auto& [layer, violation_map] : third_layer_type_rect_list_map) {
    for (auto& [type, violation_list] : violation_map) {
      int new_type = type * 10 + 2;
      combined_layer_type_rect_list_map[layer][new_type].insert(combined_layer_type_rect_list_map[layer][new_type].end(),
                                                                violation_list.begin(), violation_list.end());
    }
  }
  plotGDS(combined_gds_file, combined_layer_type_rect_list_map);
}

}  // namespace idrc