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
#include "RuleValidator.hpp"

#include "DRCHeader.hpp"
#include "GDSPlotter.hpp"
#include "Monitor.hpp"

namespace idrc {

// public

void RuleValidator::initInst()
{
  if (_rv_instance == nullptr) {
    _rv_instance = new RuleValidator();
  }
}

RuleValidator& RuleValidator::getInst()
{
  if (_rv_instance == nullptr) {
    DRCLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_rv_instance;
}

void RuleValidator::destroyInst()
{
  if (_rv_instance != nullptr) {
    delete _rv_instance;
    _rv_instance = nullptr;
  }
}

// function

std::vector<Violation> RuleValidator::verify(std::vector<DRCShape>& drc_env_shape_list, std::vector<DRCShape>& drc_result_shape_list)
{
  Monitor monitor;
  DRCLOG.info(Loc::current(), "Starting...");
  RVModel rv_model = initRVModel(drc_env_shape_list, drc_result_shape_list);
  setRVComParam(rv_model);
  buildRVBoxList(rv_model);
  verifyRVModel(rv_model);
  // debugVerifyRVModelByGolden(rv_model);
  buildViolationList(rv_model);
  // debugPlotRVModel(rv_model, "best");
  updateSummary(rv_model);
  printSummary(rv_model);
  DRCLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
  return rv_model.get_violation_list();
}

// private

RuleValidator* RuleValidator::_rv_instance = nullptr;

RVModel RuleValidator::initRVModel(std::vector<DRCShape>& drc_env_shape_list, std::vector<DRCShape>& drc_result_shape_list)
{
  RVModel rv_model;
  rv_model.set_drc_env_shape_list(drc_env_shape_list);
  rv_model.set_drc_result_shape_list(drc_result_shape_list);
  return rv_model;
}

void RuleValidator::setRVComParam(RVModel& rv_model)
{
  int32_t only_pitch = DRCDM.getOnlyPitch();
  int32_t box_size = 500 * only_pitch;
  int32_t expand_size = 5 * only_pitch;
  /**
   * box_size, expand_size
   */
  // clang-format off
  RVComParam rv_com_param(box_size,expand_size);
  // clang-format on
  DRCLOG.info(Loc::current(), "box_size: ", rv_com_param.get_box_size());
  DRCLOG.info(Loc::current(), "expand_size: ", rv_com_param.get_expand_size());
  rv_model.set_rv_com_param(rv_com_param);
}

void RuleValidator::buildRVBoxList(RVModel& rv_model)
{
  std::vector<RVBox>& rv_box_list = rv_model.get_rv_box_list();
  int32_t box_size = rv_model.get_rv_com_param().get_box_size();
  int32_t expand_size = rv_model.get_rv_com_param().get_expand_size();

  PlanarRect bounding_box(INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN);
  int32_t offset_x = -1;
  int32_t offset_y = -1;
  int32_t grid_x_size = -1;
  int32_t grid_y_size = -1;
  {
    for (DRCShape& drc_env_shape : rv_model.get_drc_env_shape_list()) {
      bounding_box.set_ll_x(std::min(bounding_box.get_ll_x(), drc_env_shape.get_ll_x()));
      bounding_box.set_ll_y(std::min(bounding_box.get_ll_y(), drc_env_shape.get_ll_y()));
      bounding_box.set_ur_x(std::max(bounding_box.get_ur_x(), drc_env_shape.get_ur_x()));
      bounding_box.set_ur_y(std::max(bounding_box.get_ur_y(), drc_env_shape.get_ur_y()));
    }
    for (DRCShape& drc_result_shape : rv_model.get_drc_result_shape_list()) {
      bounding_box.set_ll_x(std::min(bounding_box.get_ll_x(), drc_result_shape.get_ll_x()));
      bounding_box.set_ll_y(std::min(bounding_box.get_ll_y(), drc_result_shape.get_ll_y()));
      bounding_box.set_ur_x(std::max(bounding_box.get_ur_x(), drc_result_shape.get_ur_x()));
      bounding_box.set_ur_y(std::max(bounding_box.get_ur_y(), drc_result_shape.get_ur_y()));
    }
    offset_x = bounding_box.get_ll_x();
    offset_y = bounding_box.get_ll_y();
    grid_x_size = bounding_box.getXSpan() / box_size + 1;
    grid_y_size = bounding_box.getYSpan() / box_size + 1;
  }
  rv_box_list.resize(grid_x_size * grid_y_size);
  for (int32_t grid_x = 0; grid_x < grid_x_size; grid_x++) {
    for (int32_t grid_y = 0; grid_y < grid_y_size; grid_y++) {
      RVBox& rv_box = rv_box_list[grid_x + grid_y * grid_x_size];
      rv_box.set_box_idx(grid_x + grid_y * grid_x_size);
      rv_box.get_box_rect().set_ll(grid_x * box_size + offset_x, grid_y * box_size + offset_y);
      rv_box.get_box_rect().set_ur((grid_x + 1) * box_size + offset_x, (grid_y + 1) * box_size + offset_y);
    }
  }
  for (DRCShape& drc_env_shape : rv_model.get_drc_env_shape_list()) {
    PlanarRect searched_rect = DRCUTIL.getEnlargedRect(drc_env_shape.get_rect(), expand_size);
    searched_rect = DRCUTIL.getRegularRect(searched_rect, bounding_box);
    int32_t grid_ll_x = (searched_rect.get_ll_x() - offset_x) / box_size;
    int32_t grid_ll_y = (searched_rect.get_ll_y() - offset_y) / box_size;
    int32_t grid_ur_x = (searched_rect.get_ur_x() - offset_x) / box_size;
    int32_t grid_ur_y = (searched_rect.get_ur_y() - offset_y) / box_size;
    for (int32_t grid_x = grid_ll_x; grid_x <= grid_ur_x; grid_x++) {
      for (int32_t grid_y = grid_ll_y; grid_y <= grid_ur_y; grid_y++) {
        int32_t box_idx = grid_x + grid_y * grid_x_size;
        if (static_cast<int32_t>(rv_box_list.size()) <= box_idx) {
          DRCLOG.error(Loc::current(), "rv_box_list.size() <= box_idx!");
        }
        rv_box_list[box_idx].get_drc_env_shape_list().push_back(&drc_env_shape);
      }
    }
  }
  for (DRCShape& drc_result_shape : rv_model.get_drc_result_shape_list()) {
    PlanarRect searched_rect = DRCUTIL.getEnlargedRect(drc_result_shape.get_rect(), expand_size);
    searched_rect = DRCUTIL.getRegularRect(searched_rect, bounding_box);
    int32_t grid_ll_x = (searched_rect.get_ll_x() - offset_x) / box_size;
    int32_t grid_ll_y = (searched_rect.get_ll_y() - offset_y) / box_size;
    int32_t grid_ur_x = (searched_rect.get_ur_x() - offset_x) / box_size;
    int32_t grid_ur_y = (searched_rect.get_ur_y() - offset_y) / box_size;
    for (int32_t grid_x = grid_ll_x; grid_x <= grid_ur_x; grid_x++) {
      for (int32_t grid_y = grid_ll_y; grid_y <= grid_ur_y; grid_y++) {
        int32_t box_idx = grid_x + grid_y * grid_x_size;
        if (static_cast<int32_t>(rv_box_list.size()) <= box_idx) {
          DRCLOG.error(Loc::current(), "rv_box_list.size() <= box_idx!");
        }
        rv_box_list[box_idx].get_drc_result_shape_list().push_back(&drc_result_shape);
      }
    }
  }
  for (DRCShape& drc_result_shape : rv_model.get_drc_result_shape_list()) {
    if (drc_result_shape.get_net_idx() < 0) {
      DRCLOG.error(Loc::current(), "The drc_result_shape_list exist idx < 0!");
    }
  }
}

void RuleValidator::verifyRVModel(RVModel& rv_model)
{
  Monitor monitor;
  DRCLOG.info(Loc::current(), "Starting...");
#pragma omp parallel for
  for (RVBox& rv_box : rv_model.get_rv_box_list()) {
    if (needVerifying(rv_box)) {
      buildViolationSet(rv_box);
      buildViolationList(rv_box);
      updateSummary(rv_box);
      // debugViolationByType(rv_box, ViolationType::kNone);
    }
  }
  DRCLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

bool RuleValidator::needVerifying(RVBox& rv_box)
{
  if (rv_box.get_drc_result_shape_list().empty()) {
    return false;
  }
  for (DRCShape* drc_result_shape : rv_box.get_drc_result_shape_list()) {
    if (DRCUTIL.isOpenOverlap(rv_box.get_box_rect(), drc_result_shape->get_rect())) {
      return true;
    }
  }
  return false;
}

void RuleValidator::buildViolationSet(RVBox& rv_box)
{
  std::vector<DRCShape*> temp = rv_box.get_drc_result_shape_list();
  rv_box.get_drc_result_shape_list().clear();
  verifyRVBox(rv_box);
  processRVBox(rv_box);
  for (Violation& violation : rv_box.get_violation_list()) {
    rv_box.get_env_violation_set().insert(violation);
  }
  rv_box.set_drc_result_shape_list(temp);
  rv_box.get_violation_list().clear();
}

void RuleValidator::verifyRVBox(RVBox& rv_box)
{
  std::set<ViolationType>& exist_rule_set = DRCDM.getDatabase().get_exist_rule_set();

  if (DRCUTIL.exist(exist_rule_set, ViolationType::kAdjacentCutSpacing)) {
    verifyAdjacentCutSpacing(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kCornerFillSpacing)) {
    verifyCornerFillSpacing(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kCornerSpacing)) {
    verifyCornerSpacing(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kCutEOLSpacing)) {
    verifyCutEOLSpacing(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kCutShort)) {
    verifyCutShort(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kDifferentLayerCutSpacing)) {
    verifyDifferentLayerCutSpacing(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kEnclosure)) {
    verifyEnclosure(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kEnclosureEdge)) {
    verifyEnclosureEdge(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kEnclosureParallel)) {
    verifyEnclosureParallel(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kEndOfLineSpacing)) {
    verifyEndOfLineSpacing(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kFloatingPatch)) {
    verifyFloatingPatch(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kJogToJogSpacing)) {
    verifyJogToJogSpacing(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kMaximumWidth)) {
    verifyMaximumWidth(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kMaxViaStack)) {
    verifyMaxViaStack(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kMetalShort)) {
    verifyMetalShort(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kMinHole)) {
    verifyMinHole(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kMinimumArea)) {
    verifyMinimumArea(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kMinimumCut)) {
    verifyMinimumCut(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kMinimumWidth)) {
    verifyMinimumWidth(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kMinStep)) {
    verifyMinStep(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kNonsufficientMetalOverlap)) {
    verifyNonsufficientMetalOverlap(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kNotchSpacing)) {
    verifyNotchSpacing(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kOffGridOrWrongWay)) {
    verifyOffGridOrWrongWay(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kOutOfDie)) {
    verifyOutOfDie(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kParallelRunLengthSpacing)) {
    verifyParallelRunLengthSpacing(rv_box);
  }
  if (DRCUTIL.exist(exist_rule_set, ViolationType::kSameLayerCutSpacing)) {
    verifySameLayerCutSpacing(rv_box);
  }
}

void RuleValidator::processRVBox(RVBox& rv_box)
{
  std::vector<Violation> new_violation_list;
  for (Violation& violation : rv_box.get_violation_list()) {
    if (DRCUTIL.exist(rv_box.get_env_violation_set(), violation)) {
      continue;
    }
    if (!DRCUTIL.isOpenOverlap(rv_box.get_box_rect(), violation.get_rect())) {
      continue;
    }
    new_violation_list.push_back(violation);
  }
  std::sort(new_violation_list.begin(), new_violation_list.end(), CmpViolation());
  new_violation_list.erase(std::unique(new_violation_list.begin(), new_violation_list.end()), new_violation_list.end());
  rv_box.set_violation_list(new_violation_list);
}

void RuleValidator::buildViolationList(RVBox& rv_box)
{
  verifyRVBox(rv_box);
  processRVBox(rv_box);
}

void RuleValidator::updateSummary(RVBox& rv_box)
{
  std::string& golden_directory_path = DRCDM.getConfig().golden_directory_path;
  if (golden_directory_path == "null") {
    return;
  }
  std::map<std::string, std::map<std::string, int32_t>>& type_statistics_map = rv_box.get_rv_summary().type_statistics_map;
  int32_t& total_correct_num = rv_box.get_rv_summary().total_correct_num;
  int32_t& total_incorrect_num = rv_box.get_rv_summary().total_incorrect_num;
  int32_t& total_missed_num = rv_box.get_rv_summary().total_missed_num;
  int32_t& total_statistics_num = rv_box.get_rv_summary().total_statistics_num;

  type_statistics_map.clear();
  total_correct_num = 0;
  total_incorrect_num = 0;
  total_missed_num = 0;
  total_statistics_num = 0;

  // 自己检测的违例
  std::map<ViolationType, std::set<Violation, CmpViolation>>& type_violation_map = rv_box.get_type_violation_map();
  for (Violation& violation : rv_box.get_violation_list()) {
    type_violation_map[violation.get_violation_type()].insert(violation);
  }
  // golden的违例
  std::map<ViolationType, std::set<Violation, CmpViolation>>& type_golden_violation_map = rv_box.get_type_golden_violation_map();
  {
    std::string box_golden_txt_path = DRCUTIL.getString(golden_directory_path, "golden_rv_box_", rv_box.get_box_idx(), ".txt");
    if (DRCUTIL.existFile(box_golden_txt_path)) {
      std::ifstream* box_golden_txt_file = DRCUTIL.getInputFileStream(box_golden_txt_path);
      std::string new_line;
      while (std::getline(*box_golden_txt_file, new_line)) {
        if (new_line.empty()) {
          continue;
        }
        std::set<std::string> net_idx_string_set;
        std::string required_size_string;
        std::string violation_type_name;
        std::string ll_x_string;
        std::string ll_y_string;
        std::string ur_x_string;
        std::string ur_y_string;
        std::string layer_idx_string;
        // 读取
        std::istringstream net_idx_set_stream(new_line);
        std::string net_name;
        while (std::getline(net_idx_set_stream, net_name, ',')) {
          if (!net_name.empty()) {
            net_idx_string_set.insert(net_name);
          }
        }
        std::getline(*box_golden_txt_file, new_line);
        std::istringstream drc_info_stream(new_line);
        drc_info_stream >> required_size_string >> violation_type_name;
        std::getline(*box_golden_txt_file, new_line);
        std::istringstream shape_stream(new_line);
        shape_stream >> ll_x_string >> ll_y_string >> ur_x_string >> ur_y_string >> layer_idx_string;
        // 解析
        std::set<int32_t> violation_net_set;
        for (const std::string& net_idx_string : net_idx_string_set) {
          violation_net_set.insert(std::stoi(DRCUTIL.splitString(net_idx_string, '_').back()));
        }
        int32_t required_size = std::stoi(required_size_string);
        ViolationType violation_type = GetViolationTypeByName()(violation_type_name);
        LayerRect layer_rect;
        layer_rect.set_ll(std::stoi(ll_x_string), std::stoi(ll_y_string));
        layer_rect.set_ur(std::stoi(ur_x_string), std::stoi(ur_y_string));
        layer_rect.set_layer_idx(std::stoi(layer_idx_string));

        Violation violation;
        violation.set_violation_type(violation_type);
        violation.set_rect(layer_rect);
        violation.set_layer_idx(layer_rect.get_layer_idx());
        violation.set_is_routing(true);
        violation.set_violation_net_set(violation_net_set);
        violation.set_required_size(required_size);
        type_golden_violation_map[violation.get_violation_type()].insert(violation);
      }
      DRCUTIL.closeFileStream(box_golden_txt_file);
    } else {
      DRCLOG.warn(Loc::current(), "The ", box_golden_txt_path, " is not exist!");
    }
  }
  // 统计违例对比情况
  for (auto& [type, violation_set] : type_violation_map) {
    int32_t correct_num = 0;
    int32_t incorrect_num = 0;
    for (const Violation& violation : violation_set) {
      if (DRCUTIL.exist(type_golden_violation_map[type], violation)) {
        correct_num++;
      } else {
        incorrect_num++;
      }
    }
    type_statistics_map[GetViolationTypeName()(type)]["correct_num"] = correct_num;
    type_statistics_map[GetViolationTypeName()(type)]["incorrect_num"] = incorrect_num;
    total_correct_num += correct_num;
    total_incorrect_num += incorrect_num;
  }
  for (auto& [type, golden_violation_set] : type_golden_violation_map) {
    int32_t missed_num = 0;
    for (const Violation& golden_violation : golden_violation_set) {
      if (!DRCUTIL.exist(type_violation_map[type], golden_violation)) {
        missed_num++;
      }
    }
    type_statistics_map[GetViolationTypeName()(type)]["missed_num"] = missed_num;
    total_missed_num += missed_num;
  }
  for (auto& [type, statistics] : type_statistics_map) {
    statistics["statistics_num"] = statistics["correct_num"] + statistics["incorrect_num"] + statistics["missed_num"];
  }
  total_statistics_num = total_correct_num + total_incorrect_num + total_missed_num;
}

void RuleValidator::buildViolationList(RVModel& rv_model)
{
  for (RVBox& rv_box : rv_model.get_rv_box_list()) {
    for (Violation& violation : rv_box.get_violation_list()) {
      rv_model.get_violation_list().push_back(violation);
    }
  }
}

void RuleValidator::updateSummary(RVModel& rv_model)
{
  std::string& golden_directory_path = DRCDM.getConfig().golden_directory_path;
  if (golden_directory_path == "null") {
    return;
  }
  std::map<std::string, std::map<std::string, int32_t>>& type_statistics_map = rv_model.get_rv_summary().type_statistics_map;
  int32_t& total_correct_num = rv_model.get_rv_summary().total_correct_num;
  int32_t& total_incorrect_num = rv_model.get_rv_summary().total_incorrect_num;
  int32_t& total_missed_num = rv_model.get_rv_summary().total_missed_num;
  int32_t& total_statistics_num = rv_model.get_rv_summary().total_statistics_num;

  type_statistics_map.clear();
  total_correct_num = 0;
  total_incorrect_num = 0;
  total_missed_num = 0;
  total_statistics_num = 0;

  for (RVBox& rv_box : rv_model.get_rv_box_list()) {
    for (auto& [type, statistics] : rv_box.get_rv_summary().type_statistics_map) {
      type_statistics_map[type]["correct_num"] += statistics["correct_num"];
      type_statistics_map[type]["incorrect_num"] += statistics["incorrect_num"];
      type_statistics_map[type]["missed_num"] += statistics["missed_num"];
      type_statistics_map[type]["statistics_num"] += statistics["statistics_num"];
    }
    total_correct_num += rv_box.get_rv_summary().total_correct_num;
    total_incorrect_num += rv_box.get_rv_summary().total_incorrect_num;
    total_missed_num += rv_box.get_rv_summary().total_missed_num;
    total_statistics_num += rv_box.get_rv_summary().total_statistics_num;
  }
}

void RuleValidator::printSummary(RVModel& rv_model)
{
  std::string& golden_directory_path = DRCDM.getConfig().golden_directory_path;
  if (golden_directory_path == "null") {
    return;
  }
  std::map<std::string, std::map<std::string, int32_t>>& type_statistics_map = rv_model.get_rv_summary().type_statistics_map;
  int32_t& total_correct_num = rv_model.get_rv_summary().total_correct_num;
  int32_t& total_incorrect_num = rv_model.get_rv_summary().total_incorrect_num;
  int32_t& total_missed_num = rv_model.get_rv_summary().total_missed_num;
  int32_t& total_statistics_num = rv_model.get_rv_summary().total_statistics_num;

  fort::char_table type_statistics_map_table;
  {
    type_statistics_map_table.set_cell_text_align(fort::text_align::right);
    type_statistics_map_table << fort::header << "violation_type"
                              << "correct_num" << "prop"
                              << "incorrect_num" << "prop"
                              << "missed_num" << "prop" << fort::endr;
    for (auto& [type, statistics] : type_statistics_map) {
      type_statistics_map_table << type << statistics["correct_num"] << DRCUTIL.getPercentage(statistics["correct_num"], statistics["statistics_num"])
                                << statistics["incorrect_num"] << DRCUTIL.getPercentage(statistics["incorrect_num"], statistics["statistics_num"])
                                << statistics["missed_num"] << DRCUTIL.getPercentage(statistics["missed_num"], statistics["statistics_num"]) << fort::endr;
    }
    type_statistics_map_table << fort::header << "Total" << total_correct_num << DRCUTIL.getPercentage(total_correct_num, total_statistics_num)
                              << total_incorrect_num << DRCUTIL.getPercentage(total_incorrect_num, total_statistics_num) << total_missed_num
                              << DRCUTIL.getPercentage(total_missed_num, total_statistics_num) << fort::endr;
  }
  DRCUTIL.printTableList({type_statistics_map_table});
}

#if 1  // aux

int32_t RuleValidator::getIdx(int32_t idx, int32_t coord_size)
{
  return (idx + coord_size) % coord_size;
}

#endif

#if 1  // debug

void RuleValidator::debugPlotRVModel(RVModel& rv_model, std::string flag)
{
  std::string& rv_temp_directory_path = DRCDM.getConfig().rv_temp_directory_path;

  GPGDS gp_gds;

  for (DRCShape& drc_env_shape : rv_model.get_drc_env_shape_list()) {
    GPStruct drc_env_shape_struct(DRCUTIL.getString("drc_env_shape(net_", drc_env_shape.get_net_idx(), ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kEnvShape));
    gp_boundary.set_rect(drc_env_shape.get_rect());
    if (drc_env_shape.get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(drc_env_shape.get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(drc_env_shape.get_layer_idx()));
    }
    drc_env_shape_struct.push(gp_boundary);
    gp_gds.addStruct(drc_env_shape_struct);
  }

  for (DRCShape& drc_result_shape : rv_model.get_drc_result_shape_list()) {
    GPStruct drc_result_shape_struct(DRCUTIL.getString("drc_result_shape(net_", drc_result_shape.get_net_idx(), ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kResultShape));
    gp_boundary.set_rect(drc_result_shape.get_rect());
    if (drc_result_shape.get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(drc_result_shape.get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(drc_result_shape.get_layer_idx()));
    }
    drc_result_shape_struct.push(gp_boundary);
    gp_gds.addStruct(drc_result_shape_struct);
  }

  for (Violation& violation : rv_model.get_violation_list()) {
    std::string net_idx_name = DRCUTIL.getString("net");
    for (int32_t violation_net_idx : violation.get_violation_net_set()) {
      net_idx_name = DRCUTIL.getString(net_idx_name, ",", violation_net_idx);
    }
    GPStruct violation_struct(DRCUTIL.getString("violation(", net_idx_name, ")(rs,", violation.get_required_size(), ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(DRCGP.convertGPDataType(violation.get_violation_type())));
    gp_boundary.set_rect(violation.get_rect());
    if (violation.get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(violation.get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(violation.get_layer_idx()));
    }
    violation_struct.push(gp_boundary);
    gp_gds.addStruct(violation_struct);
  }

  std::string gds_file_path = DRCUTIL.getString(rv_temp_directory_path, flag, "_rv_model.gds");
  DRCGP.plot(gp_gds, gds_file_path);
}

void RuleValidator::debugPlotRVBox(RVBox& rv_box, std::string flag)
{
  std::string& rv_temp_directory_path = DRCDM.getConfig().rv_temp_directory_path;

  GPGDS gp_gds;

  GPStruct base_region_struct("base_region");
  GPBoundary gp_boundary;
  gp_boundary.set_layer_idx(0);
  gp_boundary.set_data_type(0);
  gp_boundary.set_rect(rv_box.get_box_rect());
  base_region_struct.push(gp_boundary);
  gp_gds.addStruct(base_region_struct);

  for (DRCShape* drc_env_shape : rv_box.get_drc_env_shape_list()) {
    GPStruct drc_env_shape_struct(DRCUTIL.getString("drc_env_shape(net_", drc_env_shape->get_net_idx(), ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kEnvShape));
    gp_boundary.set_rect(drc_env_shape->get_rect());
    if (drc_env_shape->get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(drc_env_shape->get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(drc_env_shape->get_layer_idx()));
    }
    drc_env_shape_struct.push(gp_boundary);
    gp_gds.addStruct(drc_env_shape_struct);
  }

  for (DRCShape* drc_result_shape : rv_box.get_drc_result_shape_list()) {
    GPStruct drc_result_shape_struct(DRCUTIL.getString("drc_result_shape(net_", drc_result_shape->get_net_idx(), ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kResultShape));
    gp_boundary.set_rect(drc_result_shape->get_rect());
    if (drc_result_shape->get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(drc_result_shape->get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(drc_result_shape->get_layer_idx()));
    }
    drc_result_shape_struct.push(gp_boundary);
    gp_gds.addStruct(drc_result_shape_struct);
  }

  for (Violation& violation : rv_box.get_violation_list()) {
    std::string net_idx_name = DRCUTIL.getString("net");
    for (int32_t violation_net_idx : violation.get_violation_net_set()) {
      net_idx_name = DRCUTIL.getString(net_idx_name, ",", violation_net_idx);
    }
    GPStruct violation_struct(DRCUTIL.getString("violation(", net_idx_name, ")(rs,", violation.get_required_size(), ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(DRCGP.convertGPDataType(violation.get_violation_type())));
    gp_boundary.set_rect(violation.get_rect());
    if (violation.get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(violation.get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(violation.get_layer_idx()));
    }
    violation_struct.push(gp_boundary);
    gp_gds.addStruct(violation_struct);
  }

  std::string gds_file_path = DRCUTIL.getString(rv_temp_directory_path, flag, "_rv_box_", rv_box.get_box_idx(), ".gds");

  DRCGP.plot(gp_gds, gds_file_path);
}

void RuleValidator::debugViolationByType(RVBox& rv_box, ViolationType violation_type)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::string& golden_directory_path = DRCDM.getConfig().golden_directory_path;
  if (golden_directory_path == "null") {
    return;
  }
#pragma omp critical
  {
    DRCLOG.warn(Loc::current(), "");
    DRCLOG.warn(Loc::current(), "***** Begin Box ", rv_box.get_box_idx(), " *****");
    DRCLOG.warn(Loc::current(), DRCUTIL.getSpaceByTabNum(1), GetViolationTypeName()(violation_type));
    DRCLOG.warn(Loc::current(), DRCUTIL.getSpaceByTabNum(2), "incorrect");
    int32_t incorrect_number = 0;
    for (auto& [type, violation_set] : rv_box.get_type_violation_map()) {
      if (violation_type != type) {
        continue;
      }
      for (const Violation& violation : violation_set) {
        if (!DRCUTIL.exist(rv_box.get_type_golden_violation_map()[type], violation)) {
          std::string violation_info = "";
          violation_info = DRCUTIL.getString(violation_info, DRCUTIL.getSpaceByTabNum(3));
          violation_info = DRCUTIL.getString(violation_info, violation.get_ll_x() / 1000.0, ",", violation.get_ll_y() / 1000.0, "  ");
          violation_info = DRCUTIL.getString(violation_info, violation.get_ur_x() / 1000.0, ",", violation.get_ur_y() / 1000.0, "  ");
          violation_info = DRCUTIL.getString(violation_info, routing_layer_list[violation.get_layer_idx()].get_layer_name(), "  ");
          violation_info = DRCUTIL.getString(violation_info, "{ ");
          for (int32_t violation_net_idx : violation.get_violation_net_set()) {
            violation_info = DRCUTIL.getString(violation_info, violation_net_idx, " ");
          }
          violation_info = DRCUTIL.getString(violation_info, "}", "  ");
          violation_info = DRCUTIL.getString(violation_info, violation.get_required_size(), "  ");
          violation_info = DRCUTIL.getString(violation_info, violation.get_is_routing() ? "true" : "false");
          DRCLOG.warn(Loc::current(), violation_info);
          incorrect_number++;
        }
      }
    }
    if (incorrect_number > 0) {
      DRCLOG.warn(Loc::current(), DRCUTIL.getSpaceByTabNum(3), "incorrect_number: ", incorrect_number);
    }
    DRCLOG.warn(Loc::current(), DRCUTIL.getSpaceByTabNum(2), "missed");
    int32_t missed_number = 0;
    for (auto& [type, golden_violation_set] : rv_box.get_type_golden_violation_map()) {
      if (violation_type != type) {
        continue;
      }
      for (const Violation& golden_violation : golden_violation_set) {
        if (!DRCUTIL.exist(rv_box.get_type_violation_map()[type], golden_violation)) {
          std::string violation_info = "";
          violation_info = DRCUTIL.getString(violation_info, DRCUTIL.getSpaceByTabNum(3));
          violation_info = DRCUTIL.getString(violation_info, golden_violation.get_ll_x() / 1000.0, ",", golden_violation.get_ll_y() / 1000.0, "  ");
          violation_info = DRCUTIL.getString(violation_info, golden_violation.get_ur_x() / 1000.0, ",", golden_violation.get_ur_y() / 1000.0, "  ");
          violation_info = DRCUTIL.getString(violation_info, routing_layer_list[golden_violation.get_layer_idx()].get_layer_name(), "  ");
          violation_info = DRCUTIL.getString(violation_info, "{ ");
          for (int32_t violation_net_idx : golden_violation.get_violation_net_set()) {
            violation_info = DRCUTIL.getString(violation_info, violation_net_idx, " ");
          }
          violation_info = DRCUTIL.getString(violation_info, "}", "  ");
          violation_info = DRCUTIL.getString(violation_info, golden_violation.get_required_size(), "  ");
          violation_info = DRCUTIL.getString(violation_info, golden_violation.get_is_routing() ? "true" : "false");
          DRCLOG.warn(Loc::current(), violation_info);
          missed_number++;
        }
      }
    }
    if (missed_number > 0) {
      DRCLOG.warn(Loc::current(), DRCUTIL.getSpaceByTabNum(3), "missed_number: ", missed_number);
    }
    DRCLOG.warn(Loc::current(), "***** End Box ", rv_box.get_box_idx(), " *****");
    DRCLOG.warn(Loc::current(), "");
    if (incorrect_number + missed_number > 0) {
      debugPlotRVBox(rv_box, "best");
    }
  }
}

void RuleValidator::debugVerifyRVModelByGolden(RVModel& rv_model)
{
#pragma omp parallel for
  for (RVBox& rv_box : rv_model.get_rv_box_list()) {
    if (needVerifying(rv_box)) {
      debugVerifyRVBoxByGolden(rv_box);
      processRVBox(rv_box);
      debugOutputViolationByGolden(rv_box);
      debugPlotRVBox(rv_box, "golden");
    }
  }
}

void RuleValidator::debugVerifyRVBoxByGolden(RVBox& rv_box)
{
  int32_t micron_dbu = DRCDM.getDatabase().get_micron_dbu();
  int32_t manufacture_grid = DRCDM.getDatabase().get_off_grid_or_wrong_way_rule().manufacture_grid;
  Die& die = DRCDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<std::string, int32_t>& routing_layer_name_to_idx_map = DRCDM.getDatabase().get_routing_layer_name_to_idx_map();
  std::map<std::string, int32_t>& cut_layer_name_to_idx_map = DRCDM.getDatabase().get_cut_layer_name_to_idx_map();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::string& rv_temp_directory_path = DRCDM.getConfig().rv_temp_directory_path;
  std::string top_name = DRCUTIL.getString("rv_box_", rv_box.get_box_idx());
  std::string top_dir_path = DRCUTIL.getString(rv_temp_directory_path, top_name);
  std::string def_file_path = DRCUTIL.getString(top_dir_path, "/clean.def");
  std::string netlist_file_path = DRCUTIL.getString(top_dir_path, "/clean.v");
  std::string prepared_file_path = DRCUTIL.getString(top_dir_path, "/prepared");
  std::string finished_file_path = DRCUTIL.getString(top_dir_path, "/finished");
  std::string violation_file_path = DRCUTIL.getString(top_dir_path, "/drc.irt");

  auto getViolationList = [&](const std::vector<DRCShape*>& drc_env_shape_list, const std::vector<DRCShape*>& drc_result_shape_list) {
    std::vector<Violation> violation_list;
    // build
    {
      // 删除再构建top文件夹
      DRCUTIL.removeDir(top_dir_path);
      DRCUTIL.createDir(top_dir_path);
      // 修改文件夹权限
      std::filesystem::perms permissions = std::filesystem::perms::owner_all | std::filesystem::perms::group_all | std::filesystem::perms::others_all;
      DRCUTIL.changePermissions(top_dir_path, permissions);
    }
    // write
    {
      std::set<int32_t> net_idx_set;
      // 获取所有net
      {
        for (DRCShape* drc_env_shape : drc_env_shape_list) {
          net_idx_set.insert(drc_env_shape->get_net_idx());
        }
        for (DRCShape* drc_result_shape : drc_result_shape_list) {
          net_idx_set.insert(drc_result_shape->get_net_idx());
        }
      }
      // 构建def
      {
        std::ofstream* def_file = DRCUTIL.getOutputFileStream(def_file_path);
        // 构建header
        DRCUTIL.pushStream(def_file, "VERSION 5.8 ;", "\n");
        DRCUTIL.pushStream(def_file, "DESIGN ", top_name, " ;", "\n");
        DRCUTIL.pushStream(def_file, "UNITS DISTANCE MICRONS ", micron_dbu, " ;", "\n");
        DRCUTIL.pushStream(def_file, "DIEAREA ( ", die.get_ll_x(), " ", die.get_ll_y(), " ) ( ", die.get_ur_x(), " ", die.get_ur_y(), " ) ;", "\n");
        DRCUTIL.pushStream(def_file, "\n");
        // 构建via
        std::map<LayerRect, std::string, CmpLayerRectByXASC> cut_shape_via_map;
        for (DRCShape* drc_env_shape : drc_env_shape_list) {
          if (!drc_env_shape->get_is_routing()) {
            PlanarRect& real_rect = drc_env_shape->get_rect();
            int32_t layer_idx = drc_env_shape->get_layer_idx();
            PlanarCoord mid_point = real_rect.getMidPoint();
            if (cut_to_adjacent_routing_map[layer_idx].size() < 2) {
              continue;
            }
            LayerRect cut_shape(real_rect.get_ll_x() - mid_point.get_x(), real_rect.get_ll_y() - mid_point.get_y(), real_rect.get_ur_x() - mid_point.get_x(),
                                real_rect.get_ur_y() - mid_point.get_y(), layer_idx);
            cut_shape_via_map[cut_shape] = "";
          }
        }
        for (DRCShape* drc_result_shape : drc_result_shape_list) {
          if (!drc_result_shape->get_is_routing()) {
            PlanarRect& real_rect = drc_result_shape->get_rect();
            int32_t layer_idx = drc_result_shape->get_layer_idx();
            PlanarCoord mid_point = real_rect.getMidPoint();
            if (cut_to_adjacent_routing_map[layer_idx].size() < 2) {
              continue;
            }
            LayerRect cut_shape(real_rect.get_ll_x() - mid_point.get_x(), real_rect.get_ll_y() - mid_point.get_y(), real_rect.get_ur_x() - mid_point.get_x(),
                                real_rect.get_ur_y() - mid_point.get_y(), layer_idx);
            cut_shape_via_map[cut_shape] = "";
          }
        }
        DRCUTIL.pushStream(def_file, "VIAS ", cut_shape_via_map.size(), " ;", "\n");
        for (auto& [cut_shape, via_name] : cut_shape_via_map) {
          PlanarCoord mid_point = cut_shape.getMidPoint();
          via_name = DRCUTIL.getString("VIA_CUT_", cut_shape.get_layer_idx(), "_", cut_shape.get_ll_x(), "_", cut_shape.get_ll_y(), "_", cut_shape.get_ur_x(),
                                       "_", cut_shape.get_ur_y());
          DRCUTIL.pushStream(def_file, "- ", via_name, "\n");
          for (int32_t routing_layer_idx : cut_to_adjacent_routing_map[cut_shape.get_layer_idx()]) {
            DRCUTIL.pushStream(def_file, " + RECT ", routing_layer_list[routing_layer_idx].get_layer_name(), " ( ", mid_point.get_x() - manufacture_grid, " ",
                               mid_point.get_y() - manufacture_grid, " ) ( ", mid_point.get_x() + manufacture_grid, " ", mid_point.get_y() + manufacture_grid,
                               " )", "\n");
          }
          DRCUTIL.pushStream(def_file, " + RECT ", cut_layer_list[cut_shape.get_layer_idx()].get_layer_name(), " ( ", cut_shape.get_ll_x(), " ",
                             cut_shape.get_ll_y(), " ) ( ", cut_shape.get_ur_x(), " ", cut_shape.get_ur_y(), " )", "\n");
          DRCUTIL.pushStream(def_file, " ;", "\n");
        }
        DRCUTIL.pushStream(def_file, "END VIAS", "\n");
        DRCUTIL.pushStream(def_file, "\n");
        // 构建net
        std::map<int32_t, std::vector<DRCShape*>> net_shape_map;
        for (DRCShape* drc_env_shape : drc_env_shape_list) {
          net_shape_map[drc_env_shape->get_net_idx()].push_back(drc_env_shape);
        }
        for (DRCShape* drc_result_shape : drc_result_shape_list) {
          net_shape_map[drc_result_shape->get_net_idx()].push_back(drc_result_shape);
        }
        DRCUTIL.pushStream(def_file, "NETS ", net_idx_set.size(), " ;", "\n");

        for (auto& [net_idx, shape_list] : net_shape_map) {
          std::string flag = "  + ROUTED";

          if (net_idx == -1) {
            DRCUTIL.pushStream(def_file, "- net_blockage\n");
          } else {
            DRCUTIL.pushStream(def_file, "- net_", net_idx, "\n");
          }
          for (DRCShape* shape : shape_list) {
            PlanarRect& real_rect = shape->get_rect();
            int32_t layer_idx = shape->get_layer_idx();
            PlanarCoord mid_point = real_rect.getMidPoint();
            if (shape->get_is_routing()) {
              DRCUTIL.pushStream(def_file, flag, " ", routing_layer_list[layer_idx].get_layer_name(), " ( ", mid_point.get_x(), " ", mid_point.get_y(),
                                 " ) RECT ( ", real_rect.get_ll_x() - mid_point.get_x(), " ", real_rect.get_ll_y() - mid_point.get_y(), " ",
                                 real_rect.get_ur_x() - mid_point.get_x(), " ", real_rect.get_ur_y() - mid_point.get_y(), " )", "\n");
              flag = "    NEW";
            } else {
              LayerRect cut_shape(real_rect.get_ll_x() - mid_point.get_x(), real_rect.get_ll_y() - mid_point.get_y(), real_rect.get_ur_x() - mid_point.get_x(),
                                  real_rect.get_ur_y() - mid_point.get_y(), layer_idx);
              if (DRCUTIL.exist(cut_shape_via_map, cut_shape)) {
                std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[layer_idx];
                int32_t routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
                DRCUTIL.pushStream(def_file, flag, " ", routing_layer_list[routing_layer_idx].get_layer_name(), " ( ", mid_point.get_x(), " ",
                                   mid_point.get_y(), " ) ", cut_shape_via_map[cut_shape], "\n");
                flag = "    NEW";
              }
            }
          }
          DRCUTIL.pushStream(def_file, " ;\n");
        }
        DRCUTIL.pushStream(def_file, "END NETS", "\n");
        DRCUTIL.pushStream(def_file, "\n");
        // 构建footer
        DRCUTIL.pushStream(def_file, "END DESIGN", "\n");
        DRCUTIL.pushStream(def_file, "\n");
        DRCUTIL.closeFileStream(def_file);
      }
      // 构建netlist
      {
        std::ofstream* netlist_file = DRCUTIL.getOutputFileStream(netlist_file_path);
        // 构建header
        DRCUTIL.pushStream(netlist_file, "module ", top_name, " ();", "\n");
        DRCUTIL.pushStream(netlist_file, "\n");
        // 构建net
        for (int32_t net_idx : net_idx_set) {
          if (net_idx == -1) {
            DRCUTIL.pushStream(netlist_file, "wire net_blockage ;", "\n");
          } else {
            DRCUTIL.pushStream(netlist_file, "wire net_", net_idx, " ;", "\n");
          }
        }
        DRCUTIL.pushStream(netlist_file, "\n");
        // 构建footer
        DRCUTIL.pushStream(netlist_file, "endmodule", "\n");
        DRCUTIL.pushStream(netlist_file, "\n");
        DRCUTIL.closeFileStream(netlist_file);
      }
      // 构建任务状态文件
      {
        std::ofstream* prepared_file = DRCUTIL.getOutputFileStream(prepared_file_path);
        DRCUTIL.pushStream(prepared_file, " ");
        DRCUTIL.closeFileStream(prepared_file);
      }
    }
    // read
    {
      // 等待直到任务结束
      int32_t waiting_time = 0;
      while (!DRCUTIL.existFile(finished_file_path)) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        waiting_time++;
        if (waiting_time % 500 == 0) {
          DRCLOG.warn(Loc::current(), "The task ", top_name, " waited for ", waiting_time, " seconds");
        }
      }
      // 从中得到违例信息
      if (DRCUTIL.existFile(violation_file_path)) {
        std::ifstream* violation_file = DRCUTIL.getInputFileStream(violation_file_path);

        std::string new_line;
        while (std::getline(*violation_file, new_line)) {
          if (new_line.empty()) {
            continue;
          }
          std::set<std::string> net_name_set;
          std::string required;
          std::string drc_rule_name;
          std::string layer_name;
          std::string ll_x_string;
          std::string ll_y_string;
          std::string ur_x_string;
          std::string ur_y_string;
          // 读取
          std::istringstream net_name_set_stream(new_line);
          std::string net_name;
          while (std::getline(net_name_set_stream, net_name, ',')) {
            if (!net_name.empty()) {
              net_name_set.insert(net_name);
            }
          }
          std::getline(*violation_file, new_line);
          std::istringstream drc_info_stream(new_line);
          drc_info_stream >> required >> drc_rule_name;
          std::getline(*violation_file, new_line);
          std::istringstream shape_stream(new_line);
          shape_stream >> layer_name >> ll_x_string >> ll_y_string >> ur_x_string >> ur_y_string;
          // 解析
          ViolationType violation_type = GetViolationTypeByName()(drc_rule_name);
          if (violation_type == ViolationType::kNone) {
            DRCLOG.warn(Loc::current(), "Unknow rule! '", drc_rule_name, "'");
          }
          LayerRect layer_rect;
          layer_rect.set_ll(static_cast<int32_t>(std::round(std::stod(ll_x_string) * micron_dbu)),
                            static_cast<int32_t>(std::round(std::stod(ll_y_string) * micron_dbu)));
          layer_rect.set_ur(static_cast<int32_t>(std::round(std::stod(ur_x_string) * micron_dbu)),
                            static_cast<int32_t>(std::round(std::stod(ur_y_string) * micron_dbu)));
          if (DRCUTIL.exist(routing_layer_name_to_idx_map, layer_name)) {
            layer_rect.set_layer_idx(routing_layer_name_to_idx_map[layer_name]);
          } else if (DRCUTIL.exist(cut_layer_name_to_idx_map, layer_name)) {
            std::vector<int32_t> routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_name_to_idx_map[layer_name]];
            layer_rect.set_layer_idx(std::min(routing_layer_idx_list.front(), routing_layer_idx_list.back()));
          } else {
            DRCLOG.error(Loc::current(), "Unknow layer! '", layer_name, "'");
          }
          std::set<int32_t> violation_net_set;
          for (const std::string& net_name : net_name_set) {
            if (net_name == "net_blockage") {
              violation_net_set.insert(-1);
            } else {
              violation_net_set.insert(std::stoi(DRCUTIL.splitString(net_name, '_').back()));
            }
          }
          if (violation_net_set.size() > 2) {
            DRCLOG.error(Loc::current(), "The violation_net_set size > 2!");
          }
          int32_t required_size;
          if (required == "null") {
            required_size = 0;
          } else {
            if (violation_type == ViolationType::kMinHole || violation_type == ViolationType::kMinimumArea) {
              required_size = static_cast<int32_t>(std::round(std::stod(required) * micron_dbu * micron_dbu));
            } else if (violation_type == ViolationType::kMaxViaStack || violation_type == ViolationType::kMinimumCut) {
              required_size = static_cast<int32_t>(std::round(std::stod(required)));
            } else {
              required_size = static_cast<int32_t>(std::round(std::stod(required) * micron_dbu));
            }
          }
          // 筛选
          if (violation_type == ViolationType::kFloatingPatch || violation_type == ViolationType::kEnclosure) {
            continue;
          }
          if (violation_net_set.size() == 1 && (*violation_net_set.begin()) == -1) {
            continue;
          }
          Violation violation;
          violation.set_violation_type(violation_type);
          violation.set_rect(layer_rect);
          violation.set_layer_idx(layer_rect.get_layer_idx());
          violation.set_is_routing(true);
          violation.set_violation_net_set(violation_net_set);
          violation.set_required_size(required_size);
          violation_list.push_back(violation);
        }
        DRCUTIL.closeFileStream(violation_file);
      } else {
        DRCLOG.warn(Loc::current(), "The task ", top_name, " violation_file_path is not exist!");
      }
      // 删除文件夹
      DRCUTIL.removeDir(top_dir_path);
      return violation_list;
    }
  };

  std::set<Violation, CmpViolation> ignore_violation_set;
  for (Violation& violation : getViolationList(rv_box.get_drc_env_shape_list(), {})) {
    ignore_violation_set.insert(violation);
  }
  for (Violation& violation : getViolationList(rv_box.get_drc_env_shape_list(), rv_box.get_drc_result_shape_list())) {
    if (DRCUTIL.exist(ignore_violation_set, violation)) {
      continue;
    }
    rv_box.get_violation_list().push_back(violation);
  }
}

void RuleValidator::debugOutputViolationByGolden(RVBox& rv_box)
{
  std::string& rv_temp_directory_path = DRCDM.getConfig().rv_temp_directory_path;

  std::ofstream* box_golden_txt_file = DRCUTIL.getOutputFileStream(DRCUTIL.getString(rv_temp_directory_path, "golden_rv_box_", rv_box.get_box_idx(), ".txt"));
  for (Violation& violation : rv_box.get_violation_list()) {
    for (int32_t net_idx : violation.get_violation_net_set()) {
      DRCUTIL.pushStream(box_golden_txt_file, net_idx, ",");
    }
    DRCUTIL.pushStream(box_golden_txt_file, "\n");
    DRCUTIL.pushStream(box_golden_txt_file, violation.get_required_size(), " ", GetViolationTypeName()(violation.get_violation_type()), "\n");
    DRCUTIL.pushStream(box_golden_txt_file, violation.get_ll_x(), " ", violation.get_ll_y(), " ", violation.get_ur_x(), " ", violation.get_ur_y(), " ",
                       violation.get_layer_idx(), "\n");
  }
  DRCUTIL.closeFileStream(box_golden_txt_file);
}

#endif

}  // namespace idrc
