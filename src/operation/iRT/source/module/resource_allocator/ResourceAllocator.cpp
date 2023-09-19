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
#include "ResourceAllocator.hpp"

#include "DRCChecker.hpp"
#include "GDSPlotter.hpp"
#include "RTUtil.hpp"

namespace irt {

// public

void ResourceAllocator::initInst()
{
  if (_ra_instance == nullptr) {
    _ra_instance = new ResourceAllocator();
  }
}

ResourceAllocator& ResourceAllocator::getInst()
{
  if (_ra_instance == nullptr) {
    LOG_INST.error(Loc::current(), "The instance not initialized!");
  }
  return *_ra_instance;
}

void ResourceAllocator::destroyInst()
{
  if (_ra_instance != nullptr) {
    delete _ra_instance;
    _ra_instance = nullptr;
  }
}

// function

void ResourceAllocator::allocate(std::vector<Net>& net_list)
{
  Monitor monitor;

  allocateNetList(net_list);

  LOG_INST.info(Loc::current(), "The ", GetStageName()(Stage::kResourceAllocator), " completed!", monitor.getStatsInfo());
}

// private

ResourceAllocator* ResourceAllocator::_ra_instance = nullptr;

void ResourceAllocator::allocateNetList(std::vector<Net>& net_list)
{
  RAModel ra_model = init(net_list);
  iterative(ra_model);
  update(ra_model);
}

#if 1  // init

RAModel ResourceAllocator::init(std::vector<Net>& net_list)
{
  RAModel ra_model = initRAModel(net_list);
  buildRAModel(ra_model);
  checkRAModel(ra_model);
  writePYScript();
  return ra_model;
}

RAModel ResourceAllocator::initRAModel(std::vector<Net>& net_list)
{
  RAModel ra_model;
  ra_model.set_ra_net_list(convertToRANetList(net_list));
  return ra_model;
}

std::vector<RANet> ResourceAllocator::convertToRANetList(std::vector<Net>& net_list)
{
  std::vector<RANet> ra_net_list;
  ra_net_list.reserve(net_list.size());
  for (size_t i = 0; i < net_list.size(); i++) {
    ra_net_list.emplace_back(convertToRANet(net_list[i]));
  }
  return ra_net_list;
}

RANet ResourceAllocator::convertToRANet(Net& net)
{
  RANet ra_net;
  ra_net.set_origin_net(&net);
  ra_net.set_net_idx(net.get_net_idx());
  ra_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    ra_net.get_ra_pin_list().push_back(RAPin(pin));
  }
  ra_net.set_bounding_box(net.get_bounding_box());
  return ra_net;
}

void ResourceAllocator::buildRAModel(RAModel& ra_model)
{
  initRANetDemand(ra_model);
  initRAGCellList(ra_model);
  updateNetFixedRectMap(ra_model);
  calcRAGCellSupply(ra_model);
  buildRelation(ra_model);
  initTempObject(ra_model);
}

void ResourceAllocator::initRANetDemand(RAModel& ra_model)
{
  for (RANet& ra_net : ra_model.get_ra_net_list()) {
    std::vector<RAPin>& ra_pin_list = ra_net.get_ra_pin_list();

    std::vector<PlanarCoord> coord_list;
    for (RAPin& ra_pin : ra_pin_list) {
      for (LayerCoord& grid_coord : ra_pin.getGridCoordList()) {
        coord_list.push_back(grid_coord.get_planar_coord());
      }
    }
    std::sort(coord_list.begin(), coord_list.end(), CmpPlanarCoordByXASC());
    coord_list.erase(std::unique(coord_list.begin(), coord_list.end()), coord_list.end());

    // 计算所需的track条数
    irt_int routing_demand = 0;
    if (coord_list.size() == 1) {
      // local net
      routing_demand = (static_cast<irt_int>(ra_pin_list.size()) / 2);
    } else {
      routing_demand = std::min(RTUtil::getHTreeLength(coord_list), RTUtil::getVTreeLength(coord_list));
    }
    ra_net.set_routing_demand(routing_demand);
  }
}

void ResourceAllocator::initRAGCellList(RAModel& ra_model)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();

  std::vector<RAGCell>& ra_gcell_list = ra_model.get_ra_gcell_list();
  ra_gcell_list.resize(die.getXSize() * die.getYSize());
  // init gcell
  for (size_t i = 0; i < ra_gcell_list.size(); i++) {
    RAGCell& ra_gcell = ra_gcell_list[i];
    irt_int grid_x = static_cast<irt_int>(i) / die.getYSize();
    irt_int grid_y = static_cast<irt_int>(i) % die.getYSize();
    ra_gcell.set_base_region(RTUtil::getRealRect(grid_x, grid_y, gcell_axis));
  }
}

void ResourceAllocator::updateNetFixedRectMap(RAModel& ra_model)
{
  std::vector<Blockage>& routing_blockage_list = DM_INST.getDatabase().get_routing_blockage_list();

  for (const Blockage& routing_blockage : routing_blockage_list) {
    LayerRect blockage_real_rect(routing_blockage.get_real_rect(), routing_blockage.get_layer_idx());
    addRectToEnv(ra_model, RASourceType::kLayoutShape, DRCRect(-1, blockage_real_rect, true));
  }
  for (RANet& ra_net : ra_model.get_ra_net_list()) {
    for (RAPin& ra_pin : ra_net.get_ra_pin_list()) {
      for (const EXTLayerRect& routing_shape : ra_pin.get_routing_shape_list()) {
        LayerRect shape_real_rect(routing_shape.get_real_rect(), routing_shape.get_layer_idx());
        addRectToEnv(ra_model, RASourceType::kLayoutShape, DRCRect(ra_net.get_net_idx(), shape_real_rect, true));
      }
    }
  }
}

void ResourceAllocator::addRectToEnv(RAModel& ra_model, RASourceType ra_source_type, DRCRect drc_rect)
{
  if (drc_rect.get_is_routing() == false) {
    return;
  }
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();

  std::vector<RAGCell>& ra_gcell_list = ra_model.get_ra_gcell_list();

  for (const LayerRect& max_scope_real_rect : DC_INST.getMaxScope(drc_rect)) {
    LayerRect max_scope_regular_rect = RTUtil::getRegularRect(max_scope_real_rect, die.get_real_rect());
    PlanarRect max_scope_grid_rect = RTUtil::getClosedGridRect(max_scope_regular_rect, gcell_axis);
    for (irt_int x = max_scope_grid_rect.get_lb_x(); x <= max_scope_grid_rect.get_rt_x(); x++) {
      for (irt_int y = max_scope_grid_rect.get_lb_y(); y <= max_scope_grid_rect.get_rt_y(); y++) {
        RAGCell& ra_gcell = ra_gcell_list[x * die.getYSize() + y];
        DC_INST.updateRectList(ra_gcell.getRegionQuery(ra_source_type), ChangeType::kAdd, drc_rect);
      }
    }
  }
}

void ResourceAllocator::updateNetReservedViaMap(RAModel& ra_model)
{
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  for (RANet& ra_net : ra_model.get_ra_net_list()) {
    std::set<LayerCoord, CmpLayerCoordByXASC> real_coord_set;
    for (RAPin& ra_pin : ra_net.get_ra_pin_list()) {
      for (LayerCoord& real_coord : ra_pin.getRealCoordList()) {
        real_coord_set.insert(real_coord);
      }
    }
    for (const LayerCoord& real_coord : real_coord_set) {
      irt_int layer_idx = real_coord.get_layer_idx();
      for (irt_int via_below_layer_idx :
           RTUtil::getReservedViaBelowLayerIdxList(layer_idx, bottom_routing_layer_idx, top_routing_layer_idx)) {
        std::vector<Segment<LayerCoord>> segment_list;
        segment_list.emplace_back(LayerCoord(real_coord.get_planar_coord(), via_below_layer_idx),
                                  LayerCoord(real_coord.get_planar_coord(), via_below_layer_idx + 1));
        for (DRCRect& drc_rect : DC_INST.getDRCRectList(ra_net.get_net_idx(), segment_list)) {
          addRectToEnv(ra_model, RASourceType::kReservedVia, drc_rect);
        }
      }
    }
  }
}

void ResourceAllocator::calcRAGCellSupply(RAModel& ra_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  double supply_utilization_rate = DM_INST.getConfig().supply_utilization_rate;

  std::vector<RAGCell>& ra_gcell_list = ra_model.get_ra_gcell_list();
// track supply
#pragma omp parallel for
  for (RAGCell& ra_gcell : ra_gcell_list) {
    irt_int resource_supply = 0;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      irt_int whole_via_demand = routing_layer.get_min_area() / routing_layer.get_min_width();
      std::vector<PlanarRect> wire_list = getWireList(ra_gcell, routing_layer);
      if (!wire_list.empty()) {
        irt_int real_whole_wire_demand = wire_list.front().getArea() / routing_layer.get_min_width();
        irt_int gcell_whole_wire_demand = 0;
        if (routing_layer.isPreferH()) {
          gcell_whole_wire_demand = ra_gcell.get_base_region().getXSpan();
        } else {
          gcell_whole_wire_demand = ra_gcell.get_base_region().getYSpan();
        }
        if (real_whole_wire_demand != gcell_whole_wire_demand) {
          LOG_INST.error(Loc::current(), "The real_whole_wire_demand and gcell_whole_wire_demand are not equal!");
        }
      }
      for (RASourceType ra_source_type : {RASourceType::kLayoutShape, RASourceType::kReservedVia}) {
        for (const auto& [net_idx, rect_set] :
             DC_INST.getLayerNetRectMap(ra_gcell.getRegionQuery(ra_source_type), true)[routing_layer.get_layer_idx()]) {
          for (const LayerRect& rect : rect_set) {
            for (const LayerRect& min_scope_real_rect : DC_INST.getMinScope(DRCRect(net_idx, rect, true))) {
              std::vector<PlanarRect> new_wire_list;
              for (PlanarRect& wire : wire_list) {
                if (RTUtil::isOpenOverlap(min_scope_real_rect, wire)) {
                  // 要切
                  std::vector<PlanarRect> split_rect_list
                      = RTUtil::getSplitRectList(wire, min_scope_real_rect, routing_layer.get_prefer_direction());
                  new_wire_list.insert(new_wire_list.end(), split_rect_list.begin(), split_rect_list.end());
                } else {
                  // 不切
                  new_wire_list.push_back(wire);
                }
              }
              wire_list = new_wire_list;
            }
          }
        }
      }
      for (PlanarRect& wire : wire_list) {
        irt_int supply = wire.getArea() / routing_layer.get_min_width();
        if (supply < whole_via_demand) {
          continue;
        }
        resource_supply += supply;
      }
    }
    resource_supply *= supply_utilization_rate;
    ra_gcell.set_resource_supply(resource_supply);
  }
}

std::vector<PlanarRect> ResourceAllocator::getWireList(RAGCell& ra_gcell, RoutingLayer& routing_layer)
{
  irt_int real_lb_x = ra_gcell.get_base_region().get_lb_x();
  irt_int real_lb_y = ra_gcell.get_base_region().get_lb_y();
  irt_int real_rt_x = ra_gcell.get_base_region().get_rt_x();
  irt_int real_rt_y = ra_gcell.get_base_region().get_rt_y();
  std::vector<irt_int> x_list = RTUtil::getOpenScaleList(real_lb_x, real_rt_x, routing_layer.getXTrackGridList());
  std::vector<irt_int> y_list = RTUtil::getOpenScaleList(real_lb_y, real_rt_y, routing_layer.getYTrackGridList());
  irt_int half_width = routing_layer.get_min_width() / 2;

  std::vector<PlanarRect> wire_list;
  if (routing_layer.isPreferH()) {
    for (irt_int y : y_list) {
      wire_list.emplace_back(real_lb_x, y - half_width, real_rt_x, y + half_width);
    }
  } else {
    for (irt_int x : x_list) {
      wire_list.emplace_back(x - half_width, real_lb_y, x + half_width, real_rt_y);
    }
  }
  return wire_list;
}

void ResourceAllocator::buildRelation(RAModel& ra_model)
{
  Monitor monitor;

  Die& die = DM_INST.getDatabase().get_die();

  std::vector<RANet>& ra_net_list = ra_model.get_ra_net_list();
  std::vector<RAGCell>& ra_gcell_list = ra_model.get_ra_gcell_list();
  std::vector<double>& result_list = ra_model.get_result_list();

  irt_int result_idx = 0;
  for (irt_int ra_net_idx = 0; ra_net_idx < static_cast<irt_int>(ra_net_list.size()); ra_net_idx++) {
    RANet& ra_net = ra_net_list[ra_net_idx];
    if (ra_net.get_net_idx() != ra_net_idx) {
      LOG_INST.error(Loc::current(), "The net_list be reordered!");
    }
    BoundingBox& bounding_box = ra_net.get_bounding_box();
    for (irt_int x = bounding_box.get_grid_lb_x(); x <= bounding_box.get_grid_rt_x(); x++) {
      for (irt_int y = bounding_box.get_grid_lb_y(); y <= bounding_box.get_grid_rt_y(); y++) {
        irt_int ra_gcell_idx = x * die.getYSize() + y;
        RAGCell& ra_gcell = ra_gcell_list[ra_gcell_idx];
        ra_net.get_ra_gcell_node_list().emplace_back(ra_gcell_idx, result_idx);
        ra_gcell.get_ra_net_node_list().emplace_back(ra_net_idx, result_idx);
        ++result_idx;
      }
    }
  }

  result_list.resize(result_idx + 1);
  for (size_t i = 0; i < result_list.size(); i++) {
    result_list[i] = 0.0;
  }

  LOG_INST.info(Loc::current(), "Establish iteration relationship completed!", monitor.getStatsInfo());
}

void ResourceAllocator::initTempObject(RAModel& ra_model)
{
  std::vector<RAGCell>& ra_gcell_list = ra_model.get_ra_gcell_list();
  std::vector<RANet>& ra_net_list = ra_model.get_ra_net_list();
  std::vector<double>& nabla_f_row = ra_model.get_nabla_f_row();
  std::vector<double>& nabla_f_col = ra_model.get_nabla_f_col();

  nabla_f_row.resize(ra_gcell_list.size());
  for (size_t i = 0; i < nabla_f_row.size(); i++) {
    nabla_f_row[i] = 0.0;
  }

  nabla_f_col.resize(ra_net_list.size());
  for (size_t i = 0; i < nabla_f_col.size(); i++) {
    nabla_f_col[i] = 0.0;
  }
}

void ResourceAllocator::checkRAModel(RAModel& ra_model)
{
  for (RAGCell& ra_gcell : ra_model.get_ra_gcell_list()) {
    if (ra_gcell.get_resource_supply() < 0) {
      LOG_INST.error(Loc::current(), "The resource_supply < 0!");
    }
  }
}

void ResourceAllocator::writePYScript()
{
  std::string ra_temp_directory_path = DM_INST.getConfig().ra_temp_directory_path;
  irt_int ra_outer_max_iter_num = DM_INST.getConfig().ra_outer_max_iter_num;

  std::ofstream* python_file = RTUtil::getOutputFileStream(RTUtil::getString(ra_temp_directory_path, "plot.py"));

  RTUtil::pushStream(python_file, "## 导入绘图需要用到的python库", "\n");
  RTUtil::pushStream(python_file, "from concurrent.futures import process", "\n");
  RTUtil::pushStream(python_file, "import numpy as np", "\n");
  RTUtil::pushStream(python_file, "import matplotlib.pyplot as plt", "\n");
  RTUtil::pushStream(python_file, "import seaborn as sns", "\n");
  RTUtil::pushStream(python_file, "import pandas as pd", "\n");
  RTUtil::pushStream(python_file, "from PIL import Image", "\n");
  RTUtil::pushStream(python_file, "import glob", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file, "for i in range(1,", ra_outer_max_iter_num + 1, "):", "\n");
  RTUtil::pushStream(python_file, "    csv_data = pd.read_csv('ra_model_'+ str(i) +'.csv')", "\n");
  RTUtil::pushStream(python_file, "    array_data = np.array(csv_data)", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file, "    # 输出热力图", "\n");
  RTUtil::pushStream(python_file, "    plt.clf()", "\n");
  RTUtil::pushStream(python_file, "    hm=sns.heatmap(array_data, vmin=0, vmax=1.1, cmap='hot_r')", "\n");
  RTUtil::pushStream(python_file, "    hm.set_title('ra_model_'+ str(i))", "\n");
  RTUtil::pushStream(python_file, "    s1 = hm.get_figure()", "\n");
  RTUtil::pushStream(python_file, "    s1.savefig('ra_model_'+ str(i) +'.png',dpi=1000)", "\n");
  RTUtil::pushStream(python_file, "    # plt.show()", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file, "images = glob.glob('ra_model_*.png')", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file, "# 提取文件名中的id数字部分,并转换为整数", "\n");
  RTUtil::pushStream(python_file, "sorted_images = sorted(images, key=lambda x: int(x.split('_')[-1].split('.')[0]))", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file, "frames = []", "\n");
  RTUtil::pushStream(python_file, "for image in sorted_images:", "\n");
  RTUtil::pushStream(python_file, "    img = Image.open(image)", "\n");
  RTUtil::pushStream(python_file, "    img = img.resize((800, 600))", "\n");
  RTUtil::pushStream(python_file, "    frames.append(img)", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file,
                     "frames[0].save('output.gif', format='GIF', append_images=frames[1:], save_all=True, duration=300, loop=0)", "\n");
  RTUtil::closeFileStream(python_file);
}

#endif

#if 1  // iterative

void ResourceAllocator::iterative(RAModel& ra_model)
{
  double ra_initial_penalty = DM_INST.getConfig().ra_initial_penalty;         //!< 罚函数的参数
  double ra_penalty_drop_rate = DM_INST.getConfig().ra_penalty_drop_rate;     //!< 罚函数的参数下降系数
  irt_int ra_outer_max_iter_num = DM_INST.getConfig().ra_outer_max_iter_num;  //!< 外层循环数

  for (irt_int outer_iter = 1; outer_iter <= ra_outer_max_iter_num; outer_iter++) {
    Monitor iter_monitor;
    LOG_INST.info(Loc::current(), "****** Start Iteration(", outer_iter, "/", ra_outer_max_iter_num, ") ******");
    double penalty_para = (1 / (2 * ra_initial_penalty));
    ra_model.set_curr_outer_iter(outer_iter);
    allocateRAModel(ra_model, penalty_para);
    processRAModel(ra_model);
    countRAModel(ra_model);
    reportRAModel(ra_model);
    // outputResourceMap(ra_model);
    ra_initial_penalty *= ra_penalty_drop_rate;
    LOG_INST.info(Loc::current(), "****** End Iteration(", outer_iter, "/", ra_outer_max_iter_num, ")", iter_monitor.getStatsInfo(),
                  " ******");
    if (stopOuterRAModel(ra_model)) {
      LOG_INST.info(Loc::current(), "****** Reached the stopping condition, ending the iteration prematurely! ******");
      ra_model.set_curr_outer_iter(-1);
      break;
    }
  }
}

/**
 * @description: 使用二次规划 罚方法
 *
 * 迭代过程
 *  f(x) = (1/2) * x' * Q * x + b' * x
 *  nabla_f = Q * x + b
 *  alpha = (nabla_f' * nabla_f) / (nabla_f' * Q * nabla_f)
 *  x = x + (-nabla_f) * alpha
 *
 * eg.
 *  RAGCell : GCM    Target : T    RANet : NM    Constraint : C
 * ────────────────────────────────────────
 * │ GCM0(T1=3) │ GCM1(T2=4) │ GCM2(T3=5) │
 * ───────────────────────────────────────────────────│
 * │     x0     │     x1     │            │ NM0(C1=3) │
 * │            │            │            │           │
 * │            │     x2     │     x3     │ NM1(C2=6) │
 * ───────────────────────────────────────────────────│
 *
 * min [(x0 - T1)^2 + (x1 + x2 - T2)^2 + (x3 - T3)^2] +
 *     (1/(2 * u)) * [(x0 + x1 - C1)^2 + (x2 + x3 - C2)^2]
 *
 */
void ResourceAllocator::allocateRAModel(RAModel& ra_model, double penalty_para)
{
  irt_int ra_inner_max_iter_num = DM_INST.getConfig().ra_inner_max_iter_num;  //!< 内层循环数

  for (irt_int inner_iter = 1; inner_iter <= ra_inner_max_iter_num; inner_iter++) {
    Monitor iter_monitor;
    ra_model.set_curr_inner_iter(inner_iter);
    calcNablaF(ra_model, penalty_para);
    double norm_nabla_f = calcAlpha(ra_model, penalty_para);
    double norm_square_step = updateResult(ra_model);
    LOG_INST.info(Loc::current(), "Iter(", inner_iter, "/", ra_inner_max_iter_num, "), norm_nabla_f=", norm_nabla_f,
                  ", norm_square_step=", norm_square_step, iter_monitor.getStatsInfo());
    if (stopInnerRAModel(ra_model)) {
      LOG_INST.info(Loc::current(), "****** Reached the stopping condition, ending the iteration prematurely! ******");
      ra_model.set_curr_inner_iter(-1);
      break;
    }
  }
}

/**
 * calculate nabla_f;
 * 因;
 * nabla_f = Q * x + b
 * 且;
 * Q = Q_ra_gcell + (1/2u) * Q_ra_net;
 * F = F_ra_gcell + (1/2u) * F_ra_net;
 * 代入;
 * nabla_f = Q_ra_gcell'*X + (1/2u)Q_ra_net'*X + F_ra_gcell + (1/2u)F_ra_net
 * 整理;
 * nabla_f = (Q_ra_gcell*X + F_ra_gcell) + (1/2u)*(Q_ra_net*X + F_ra_net)
 */
void ResourceAllocator::calcNablaF(RAModel& ra_model, double penalty_para)
{
  std::vector<RAGCell>& ra_gcell_list = ra_model.get_ra_gcell_list();
  std::vector<RANet>& ra_net_list = ra_model.get_ra_net_list();
  std::vector<double>& result_list = ra_model.get_result_list();
  std::vector<double>& nabla_f_row = ra_model.get_nabla_f_row();
  std::vector<double>& nabla_f_col = ra_model.get_nabla_f_col();
/**
 * calculate "Q_ra_gcell*X + F_ra_gcell";
 * 因;
 * Q_ra_gcell = (Q_ra_gcell_1 + Q_ra_gcell_2 + ...);
 * F_ra_gcell = (F_ra_gcell_1 + F_ra_gcell_2 + ...);
 * 代入;
 * (Q_ra_gcell*X + F_ra_gcell) = (Q_ra_gcell_1*X_1 + Q_ra_gcell_2*X_2 + ...) + (F_ra_gcell_1 + F_ra_gcell_2 + ...);
 * 整理;
 * (Q_ra_gcell*X + F_ra_gcell) = (Q_ra_gcell_1*X_1 + F_ra_gcell_1) + (Q_ra_gcell_2*X_2 + F_ra_gcell_2) + ...;
 */
#pragma omp parallel for
  for (size_t i = 0; i < ra_gcell_list.size(); i++) {
    RAGCell& ra_gcell = ra_gcell_list[i];
    std::vector<RANetNode>& ra_net_node_list = ra_gcell.get_ra_net_node_list();
    // calculate "Q_ra_gcell_i*X_i"
    double gcell_q_temp = 0;
    for (size_t j = 0; j < ra_net_node_list.size(); j++) {
      gcell_q_temp += result_list[ra_net_node_list[j].get_result_idx()];
    }
    // 由于有 f(x) = (1/2) * x' * Q * x + b' * x 中存在 "1/2" 故要 "*2"
    gcell_q_temp *= 2;
    // calculate "+F_ra_gcell_i"
    gcell_q_temp += (-2) * ra_gcell.get_resource_supply();
    // update nabla_f_row
    nabla_f_row[i] = gcell_q_temp;
  }

  /**
   * calculate "(1/2u) * (Q_ra_net*X+F_ra_net)";
   * 因;
   * Q_ra_net = (Q_ra_net_1 + Q_ra_net_2 + ...);
   * F_ra_net = (F_ra_net_1 + F_ra_net_2 + ...);
   * 代入;
   * (1/2u) * (Q_ra_net*X + F_ra_net) = (1/2u) * [(Q_ra_net_1*X_1 + Q_ra_net_2*X_2 + ...) + (F_ra_net_1 + F_ra_net_2 + ...)]
   * 整理;
   * (1/2u) * (Q_ra_net*X + F_ra_net) = (1/2u) * [(Q_ra_net_1*X_1 + F_ra_net_1) + (Q_ra_net_2*X_2 + F_ra_net_2) + ...]
   */
#pragma omp parallel for
  for (size_t i = 0; i < ra_net_list.size(); i++) {
    RANet& ra_net = ra_net_list[i];
    std::vector<RAGCellNode>& ra_gcell_node_list = ra_net.get_ra_gcell_node_list();
    // calculate "Q_ra_net_i*X_i"
    double bounding_box_q_temp = 0;
    for (size_t j = 0; j < ra_gcell_node_list.size(); j++) {
      bounding_box_q_temp += result_list[ra_gcell_node_list[j].get_result_idx()];
    }
    // 由于有 f(x) = (1/2) * x' * Q * x + b' * x 中存在 "1/2" 故要 "*2"
    bounding_box_q_temp *= 2;
    // calculate "+F_ra_net_i"
    bounding_box_q_temp += (-2) * ra_net.get_routing_demand();
    /**
     * notice:当前逻辑不改变的情况下,由于是在nabla_f上追加的值,所以不能在算完所有值后将nabla_f内的每个值相乘;
     * 会存在数据依赖,如果算完所有值再乘的结果与下例子一样;
     * 原结果由 { A+k*(B+C) } 变为 { k*(A+B+C) };
     * 现在计算情况与上例一样,例子中B交C为空集,所以不会增加多余计算,若需要高性能则可以再增加空间(空间换时间);
     */
    // calculate "(1/2u)*"
    bounding_box_q_temp *= penalty_para;
    // update nabla_f_col
    nabla_f_col[i] = bounding_box_q_temp;
  }
}
//  calculate alpha = (nabla_f' * nabla_f) / (nabla_f' * Q * nabla_f)
double ResourceAllocator::calcAlpha(RAModel& ra_model, double penalty_para)
{
  std::vector<RAGCell>& ra_gcell_list = ra_model.get_ra_gcell_list();
  std::vector<RANet>& ra_net_list = ra_model.get_ra_net_list();
  std::vector<double>& nabla_f_row = ra_model.get_nabla_f_row();
  std::vector<double>& nabla_f_col = ra_model.get_nabla_f_col();
  /**
   * calculate "||nabla_f||"
   * calculate "nabla_f' * Q * nabla_f"
   * Q = (Q_ra_gcell_1 + Q_ra_gcell_2 + ...) + (1/2u) * [Q_ra_net1 + Q_ra_net2 + ...];
   * 计算情况与下例一样,最后结果为单值
   * (1;2;3)'*[1 0 1;0 0 0;1 0 1]*(1;2;3)
   * =((1+3),0,(1+3))*(1;2;3)
   * =(1+3)*1+(1+3)*3
   * =(1+3)*(1+3)
   */
  double norm_nabla_f = 0;
  double nabla_f_q_nabla_f = 0;

#pragma omp parallel for
  for (size_t i = 0; i < ra_gcell_list.size(); i++) {
    RAGCell& ra_gcell = ra_gcell_list[i];
    std::vector<RANetNode>& ra_net_node_list = ra_gcell.get_ra_net_node_list();

    double gcell_q_temp = 0;
    double gcell_norm_nabla_f = 0;
    gcell_q_temp += (static_cast<irt_int>(ra_net_node_list.size()) * nabla_f_row[i]);
    for (size_t j = 0; j < ra_net_node_list.size(); j++) {
      RANetNode& ra_net_node = ra_net_node_list[j];
      gcell_q_temp += nabla_f_col[ra_net_node.get_ra_net_idx()];
      gcell_norm_nabla_f += std::pow(nabla_f_row[i] + nabla_f_col[ra_net_node.get_ra_net_idx()], 2);
    }
    double gcell_nabla_f_q_nabla_f = std::pow(gcell_q_temp, 2);
#pragma omp atomic
    norm_nabla_f += gcell_norm_nabla_f;
#pragma omp atomic
    nabla_f_q_nabla_f += gcell_nabla_f_q_nabla_f;
  }

#pragma omp parallel for
  for (size_t i = 0; i < ra_net_list.size(); i++) {
    RANet& ra_net = ra_net_list[i];
    std::vector<RAGCellNode>& ra_gcell_node_list = ra_net.get_ra_gcell_node_list();

    double bounding_box_q_temp = 0;
    bounding_box_q_temp += (static_cast<irt_int>(ra_gcell_node_list.size()) * nabla_f_col[i]);
    for (size_t j = 0; j < ra_gcell_node_list.size(); j++) {
      bounding_box_q_temp += nabla_f_row[ra_gcell_node_list[j].get_gcell_idx()];
    }
    double bounding_nabla_f_q_nabla_f = std::pow(bounding_box_q_temp, 2) * penalty_para;
#pragma omp atomic
    nabla_f_q_nabla_f += bounding_nabla_f_q_nabla_f;
  }
  // 由于有 f(x) = (1/2) * x' * Q * x + b' * x 中存在 "1/2" 故要 "*2"
  nabla_f_q_nabla_f *= 2;
  ra_model.set_alpha(norm_nabla_f / nabla_f_q_nabla_f);
  return norm_nabla_f;
}
// x = x + (-nabla_f) * alpha
double ResourceAllocator::updateResult(RAModel& ra_model)
{
  double norm_square_step = 0;

  std::vector<RAGCell>& ra_gcell_list = ra_model.get_ra_gcell_list();
  std::vector<double>& result_list = ra_model.get_result_list();
  std::vector<double>& nabla_f_row = ra_model.get_nabla_f_row();
  std::vector<double>& nabla_f_col = ra_model.get_nabla_f_col();

  double alpha = ra_model.get_alpha();

  for (size_t ra_gcell_idx = 0; ra_gcell_idx < ra_gcell_list.size(); ra_gcell_idx++) {
    RAGCell& ra_gcell = ra_gcell_list[ra_gcell_idx];
    std::vector<RANetNode>& ra_net_node_list = ra_gcell.get_ra_net_node_list();

    for (size_t j = 0; j < ra_net_node_list.size(); j++) {
      RANetNode& ra_net_node = ra_net_node_list[j];
      double step = (-1 * (nabla_f_row[ra_gcell_idx] + nabla_f_col[ra_net_node.get_ra_net_idx()]) * alpha);
      result_list[ra_net_node.get_result_idx()] += step;
      if (result_list[ra_net_node.get_result_idx()] < 0) {
        result_list[ra_net_node.get_result_idx()] = 0;
      }
      norm_square_step += std::pow(step, 2);
    }
  }
  return norm_square_step;
}

bool ResourceAllocator::stopInnerRAModel(RAModel& ra_model)
{
  return false;
}

void ResourceAllocator::processRAModel(RAModel& ra_model)
{
  Die& die = DM_INST.getDatabase().get_die();

  std::vector<RANet>& ra_net_list = ra_model.get_ra_net_list();
  std::vector<double>& result_list = ra_model.get_result_list();

  for (RANet& ra_net : ra_net_list) {
    BoundingBox& bounding_box = ra_net.get_bounding_box();
    irt_int grid_lb_x = bounding_box.get_grid_lb_x();
    irt_int grid_lb_y = bounding_box.get_grid_lb_y();

    double max_allocate_result = -DBL_MAX;
    for (RAGCellNode& ra_gcell_node : ra_net.get_ra_gcell_node_list()) {
      max_allocate_result = std::max(max_allocate_result, result_list[ra_gcell_node.get_result_idx()]);
    }
    GridMap<double> allocation_map(bounding_box.getXSize(), bounding_box.getYSize());
    for (RAGCellNode& ra_gcell_node : ra_net.get_ra_gcell_node_list()) {
      irt_int grid_x = ra_gcell_node.get_gcell_idx() / die.getYSize();
      irt_int grid_y = ra_gcell_node.get_gcell_idx() % die.getYSize();
      allocation_map[grid_x - grid_lb_x][grid_y - grid_lb_y] = result_list[ra_gcell_node.get_result_idx()];
    }

    double lower_cost = 0.001;
    GridMap<double> cost_map = getCostMap(allocation_map, lower_cost);
    normalizeCostMap(cost_map, lower_cost);
    for (RAPin& ra_pin : ra_net.get_ra_pin_list()) {
      for (LayerCoord& grid_coord : ra_pin.getGridCoordList()) {
        cost_map[grid_coord.get_x() - grid_lb_x][grid_coord.get_y() - grid_lb_y] = lower_cost;
      }
    }
    ra_net.set_ra_cost_map(cost_map);
  }
}

GridMap<double> ResourceAllocator::getCostMap(GridMap<double>& allocation_map, double lower_cost)
{
  GridMap<double> cost_map(allocation_map.get_x_size(), allocation_map.get_y_size());

  double min_allocation = DBL_MAX;
  for (irt_int i = 0; i < allocation_map.get_x_size(); i++) {
    for (irt_int j = 0; j < allocation_map.get_y_size(); j++) {
      allocation_map[i][j] = std::max(allocation_map[i][j], lower_cost);
      if (RTUtil::isNanOrInf(allocation_map[i][j])) {
        LOG_INST.error(Loc::current(), "The allocation is nan or inf!");
      }
      min_allocation = std::min(allocation_map[i][j], min_allocation);
    }
  }
  double sum_allocation = 0;
  double sum_mid = 0;
  for (irt_int i = 0; i < allocation_map.get_x_size(); i++) {
    for (irt_int j = 0; j < allocation_map.get_y_size(); j++) {
      sum_allocation += allocation_map[i][j];
      sum_mid += (min_allocation / allocation_map[i][j]);
    }
  }
  double sum_cost = 0;
  for (irt_int i = 0; i < allocation_map.get_x_size(); i++) {
    for (irt_int j = 0; j < allocation_map.get_y_size(); j++) {
      cost_map[i][j] = (min_allocation / allocation_map[i][j]) * (sum_allocation / sum_mid);
      sum_cost += cost_map[i][j];
    }
  }
  return cost_map;
}

void ResourceAllocator::normalizeCostMap(GridMap<double>& cost_map, double lower_cost)
{
  double min_cost = DBL_MAX;
  double max_cost = 0;
  for (irt_int i = 0; i < cost_map.get_x_size(); i++) {
    for (irt_int j = 0; j < cost_map.get_y_size(); j++) {
      min_cost = std::min(min_cost, cost_map[i][j]);
      max_cost = std::max(max_cost, cost_map[i][j]);
    }
  }
  double base = std::max(max_cost - min_cost, lower_cost);
  for (irt_int i = 0; i < cost_map.get_x_size(); i++) {
    for (irt_int j = 0; j < cost_map.get_y_size(); j++) {
      cost_map[i][j] = std::max((cost_map[i][j] - min_cost) / base, lower_cost);
      if (RTUtil::isNanOrInf(cost_map[i][j])) {
        LOG_INST.error(Loc::current(), "The cost is nan or inf!");
      }
      // 只取小数点后3位
      cost_map[i][j] = RTUtil::retainPlaces(cost_map[i][j], 3);
    }
  }
}

void ResourceAllocator::countRAModel(RAModel& ra_model)
{
  RAModelStat ra_model_stat;

  Die& die = DM_INST.getDatabase().get_die();

  GridMap<double> global_cost_map;
  global_cost_map.init(die.getXSize(), die.getYSize());

  for (RANet& ra_net : ra_model.get_ra_net_list()) {
    irt_int grid_lb_x = ra_net.get_bounding_box().get_grid_lb_x();
    irt_int grid_lb_y = ra_net.get_bounding_box().get_grid_lb_y();

    GridMap<double>& ra_cost_map = ra_net.get_ra_cost_map();
    for (irt_int x = 0; x < ra_cost_map.get_x_size(); ++x) {
      for (irt_int y = 0; y < ra_cost_map.get_y_size(); ++y) {
        global_cost_map[grid_lb_x + x][grid_lb_y + y] += ra_cost_map[x][y];
      }
    }
  }
  double max_global_cost = DBL_MIN;
  for (int x = 0; x < global_cost_map.get_x_size(); x++) {
    for (int y = 0; y < global_cost_map.get_y_size(); y++) {
      max_global_cost = std::max(max_global_cost, global_cost_map[x][y]);
    }
  }
  ra_model_stat.set_max_global_cost(max_global_cost);

  std::vector<double>& avg_cost_list = ra_model_stat.get_avg_cost_list();
  double max_avg_cost = -DBL_MAX;
  std::vector<RANet>& ra_net_list = ra_model.get_ra_net_list();
  for (RANet& ra_net : ra_net_list) {
    double total_cost = 0;
    GridMap<double>& ra_cost_map = ra_net.get_ra_cost_map();
    for (irt_int x = 0; x < ra_cost_map.get_x_size(); x++) {
      for (irt_int y = 0; y < ra_cost_map.get_y_size(); y++) {
        total_cost += ra_cost_map[x][y];
      }
    }
    double cost_value = total_cost / (ra_cost_map.get_x_size() * ra_cost_map.get_y_size());
    max_avg_cost = std::max(max_avg_cost, cost_value);
    avg_cost_list.push_back(cost_value);
  }
  ra_model_stat.set_max_avg_cost(max_avg_cost);

  ra_model.set_ra_model_stat(ra_model_stat);
}

void ResourceAllocator::reportRAModel(RAModel& ra_model)
{
  RAModelStat& ra_model_stat = ra_model.get_ra_model_stat();

  std::vector<double>& avg_cost_list = ra_model_stat.get_avg_cost_list();

  fort::char_table avg_cost_table;
  avg_cost_table << fort::header << "Avg Cost"
                 << "Net Number" << fort::endr;
  GridMap<std::string> avg_cost_map = RTUtil::getRangeRatioMap(avg_cost_list, {1.0});
  for (irt_int y = 0; y < avg_cost_map.get_y_size(); y++) {
    avg_cost_table << avg_cost_map[0][y] << avg_cost_map[1][y] << fort::endr;
  }
  avg_cost_table << fort::header << "Total" << avg_cost_list.size() << fort::endr;

  fort::char_table global_cost_table;
  global_cost_table << fort::header << "Max Global Cost" << fort::endr;
  global_cost_table << ra_model_stat.get_max_global_cost() << fort::endr;

  // print
  RTUtil::printTableList({avg_cost_table, global_cost_table});
}

bool ResourceAllocator::stopOuterRAModel(RAModel& ra_model)
{
  return false;
}

#endif

#if 1  // update

void ResourceAllocator::update(RAModel& ra_model)
{
  for (RANet& ra_net : ra_model.get_ra_net_list()) {
    ra_net.get_origin_net()->set_ra_cost_map(ra_net.get_ra_cost_map());
  }
}

#endif

#if 1  // plot ra_model

void ResourceAllocator::outputResourceMap(RAModel& ra_model)
{
  Die& die = DM_INST.getDatabase().get_die();
  std::string ra_temp_directory_path = DM_INST.getConfig().ra_temp_directory_path;

  GridMap<double> global_cost_map;
  global_cost_map.init(die.getXSize(), die.getYSize());

  for (RANet& ra_net : ra_model.get_ra_net_list()) {
    irt_int grid_lb_x = ra_net.get_bounding_box().get_grid_lb_x();
    irt_int grid_lb_y = ra_net.get_bounding_box().get_grid_lb_y();

    GridMap<double>& ra_cost_map = ra_net.get_ra_cost_map();
    for (irt_int x = 0; x < ra_cost_map.get_x_size(); ++x) {
      for (irt_int y = 0; y < ra_cost_map.get_y_size(); ++y) {
        global_cost_map[grid_lb_x + x][grid_lb_y + y] += ra_cost_map[x][y];
      }
    }
  }

  std::ofstream* csv_file
      = RTUtil::getOutputFileStream(RTUtil::getString(ra_temp_directory_path, "ra_model_", ra_model.get_curr_outer_iter(), ".csv"));
  for (irt_int y = global_cost_map.get_y_size() - 1; y >= 0; y--) {
    for (irt_int x = 0; x < global_cost_map.get_x_size(); x++) {
      RTUtil::pushStream(csv_file, global_cost_map[x][y], ",");
    }
    RTUtil::pushStream(csv_file, "\n");
  }
  RTUtil::closeFileStream(csv_file);
}

#endif

}  // namespace irt
