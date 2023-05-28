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

#include "RTUtil.hpp"

namespace irt {

// public

void ResourceAllocator::initInst(Config& config, Database& database)
{
  if (_ra_instance == nullptr) {
    _ra_instance = new ResourceAllocator(config, database);
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

void ResourceAllocator::allocate(std::vector<Net>& net_list)
{
  Monitor monitor;

  std::vector<RANet> ra_net_list = _ra_data_manager.convertToRANetList(net_list);
  allocateRANetList(ra_net_list);

  LOG_INST.info(Loc::current(), "The ", GetStageName()(Stage::kResourceAllocator), " completed!", monitor.getStatsInfo());
}

// private

ResourceAllocator* ResourceAllocator::_ra_instance = nullptr;

void ResourceAllocator::init(Config& config, Database& database)
{
  _ra_data_manager.input(config, database);
}

void ResourceAllocator::allocateRANetList(std::vector<RANet>& ra_net_list)
{
  RAModel ra_model = initRAModel(ra_net_list);
  buildRAModel(ra_model);
  checkRAModel(ra_model);
  allocateRAModel(ra_model);
  updateRAModel(ra_model);
  reportRAModel(ra_model);
}

#if 1  // build

RAModel ResourceAllocator::initRAModel(std::vector<RANet>& ra_net_list)
{
  RAModel ra_model;
  ra_model.set_ra_net_list(ra_net_list);
  return ra_model;
}

void ResourceAllocator::buildRAModel(RAModel& ra_model)
{
  initRANetDemand(ra_model);
  initRAGCellList(ra_model);
  updateLayerBlockageMap(ra_model);
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
  GCellAxis& gcell_axis = _ra_data_manager.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = _ra_data_manager.getDatabase().get_die();

  std::vector<RAGCell>& ra_gcell_list = ra_model.get_ra_gcell_list();
  ra_gcell_list.resize(die.getXSize() * die.getYSize());
  // init gcell
  for (size_t i = 0; i < ra_gcell_list.size(); i++) {
    RAGCell& ra_gcell = ra_gcell_list[i];
    irt_int grid_x = static_cast<irt_int>(i) / die.getYSize();
    irt_int grid_y = static_cast<irt_int>(i) % die.getYSize();
    ra_gcell.set_real_rect(RTUtil::getRealRect(grid_x, grid_y, gcell_axis));
  }
}

void ResourceAllocator::updateLayerBlockageMap(RAModel& ra_model)
{
  GCellAxis& gcell_axis = _ra_data_manager.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = _ra_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _ra_data_manager.getDatabase().get_routing_layer_list();
  std::vector<Blockage>& routing_blockage_list = _ra_data_manager.getDatabase().get_routing_blockage_list();
  irt_int bottom_routing_layer_idx = _ra_data_manager.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = _ra_data_manager.getConfig().top_routing_layer_idx;

  std::vector<RAGCell>& ra_gcell_list = ra_model.get_ra_gcell_list();
  for (const Blockage& routing_blockage : routing_blockage_list) {
    irt_int layer_idx = routing_blockage.get_layer_idx();
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      continue;
    }
    irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(routing_blockage.get_real_rect());
    PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(routing_blockage.get_real_rect(), min_spacing, die.get_real_rect());
    PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
    for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
      for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
        RAGCell& ra_gcell = ra_gcell_list[x * die.getYSize() + y];
        ra_gcell.get_layer_blockage_map()[layer_idx].push_back(enlarged_real_rect);
      }
    }
  }
  for (RANet& ra_net : ra_model.get_ra_net_list()) {
    for (RAPin& ra_pin : ra_net.get_ra_pin_list()) {
      for (const EXTLayerRect& routing_shape : ra_pin.get_routing_shape_list()) {
        irt_int layer_idx = routing_shape.get_layer_idx();
        if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
          continue;
        }
        irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(routing_shape.get_real_rect());
        PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(routing_shape.get_real_rect(), min_spacing, die.get_real_rect());
        PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
        for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
          for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
            RAGCell& ra_gcell = ra_gcell_list[x * die.getYSize() + y];
            ra_gcell.get_layer_blockage_map()[layer_idx].push_back(enlarged_real_rect);
          }
        }
      }
    }
  }
}

void ResourceAllocator::calcRAGCellSupply(RAModel& ra_model)
{
  std::vector<RoutingLayer>& routing_layer_list = _ra_data_manager.getDatabase().get_routing_layer_list();
  irt_int bottom_routing_layer_idx = _ra_data_manager.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = _ra_data_manager.getConfig().top_routing_layer_idx;
  std::map<irt_int, double>& layer_idx_utilization_ratio = _ra_data_manager.getConfig().layer_idx_utilization_ratio;

  std::vector<RAGCell>& ra_gcell_list = ra_model.get_ra_gcell_list();
// track supply
#pragma omp parallel for
  for (RAGCell& ra_gcell : ra_gcell_list) {
    irt_int public_track_supply = 0;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      irt_int layer_idx = routing_layer.get_layer_idx();
      if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
        continue;
      }
      double layer_utilization_ratio = 1;
      if (RTUtil::exist(layer_idx_utilization_ratio, layer_idx)) {
        layer_utilization_ratio = layer_idx_utilization_ratio[layer_idx];
      }
      std::vector<PlanarRect> wire_list = getWireList(ra_gcell, routing_layer);
      irt_int layer_public_track_supply = static_cast<irt_int>(wire_list.size());
      for (PlanarRect& wire : wire_list) {
        for (PlanarRect& blockage : ra_gcell.get_layer_blockage_map()[layer_idx]) {
          if (RTUtil::isOpenOverlap(blockage, wire)) {
            layer_public_track_supply--;
            break;
          }
        }
      }
      if (layer_public_track_supply < 0) {
        LOG_INST.error(Loc::current(), "The layer_public_track_supply < 0!");
      }
      public_track_supply += static_cast<irt_int>(layer_public_track_supply * layer_utilization_ratio);
    }
    ra_gcell.set_public_track_supply(public_track_supply);
  }
}

std::vector<PlanarRect> ResourceAllocator::getWireList(RAGCell& ra_gcell, RoutingLayer& routing_layer)
{
  irt_int real_lb_x = ra_gcell.get_real_rect().get_lb_x();
  irt_int real_lb_y = ra_gcell.get_real_rect().get_lb_y();
  irt_int real_rt_x = ra_gcell.get_real_rect().get_rt_x();
  irt_int real_rt_y = ra_gcell.get_real_rect().get_rt_y();
  std::vector<irt_int> x_list = RTUtil::getOpenScaleList(real_lb_x, real_rt_x, routing_layer.getXTrackGrid());
  std::vector<irt_int> y_list = RTUtil::getOpenScaleList(real_lb_y, real_rt_y, routing_layer.getYTrackGrid());
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

  Die& die = _ra_data_manager.getDatabase().get_die();

  std::vector<RANet>& ra_net_list = ra_model.get_ra_net_list();
  std::vector<RAGCell>& ra_gcell_list = ra_model.get_ra_gcell_list();
  std::vector<double>& result_list = ra_model.get_result_list();

  irt_int result_idx = 0;
  for (irt_int ra_net_idx = 0; ra_net_idx < static_cast<irt_int>(ra_net_list.size()); ra_net_idx++) {
    RANet& ra_net = ra_net_list[ra_net_idx];
    if (ra_net.get_net_idx() != ra_net_idx) {
      LOG_INST.error(Loc::current(), "The net_list be reordered!");
    }
    EXTPlanarRect& bounding_box = ra_net.get_bounding_box();
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

#endif

#if 1  // check ra_model

void ResourceAllocator::checkRAModel(RAModel& ra_model)
{
  std::vector<RoutingLayer>& routing_layer_list = _ra_data_manager.getDatabase().get_routing_layer_list();
  for (RAGCell& ra_gcell : ra_model.get_ra_gcell_list()) {
    PlanarRect& gcell_rect = ra_gcell.get_real_rect();
    for (auto& [layer_idx, blockage_list] : ra_gcell.get_layer_blockage_map()) {
      if (routing_layer_list.back().get_layer_idx() < layer_idx || layer_idx < routing_layer_list.front().get_layer_idx()) {
        LOG_INST.error(Loc::current(), "The layer idx is illegal!");
      }
      for (PlanarRect& blockage : blockage_list) {
        if (RTUtil::isClosedOverlap(gcell_rect, blockage)) {
          continue;
        }
        LOG_INST.error(Loc::current(), "The net blockage is outside the node region!");
      }
    }
    if (ra_gcell.get_public_track_supply() < 0) {
      LOG_INST.error(Loc::current(), "The public_track_supply < 0!");
    }
  }
}

#endif

#if 1  // allocate ra_model

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
void ResourceAllocator::allocateRAModel(RAModel& ra_model)
{
  Monitor monitor;

  RAConfig& ra_config = _ra_data_manager.getConfig();
  // 迭代参数
  double initial_penalty = ra_config.initial_penalty;      //!< 罚函数的参数
  double penalty_drop_rate = ra_config.penalty_drop_rate;  //!< 罚函数的参数下降系数
  irt_int outer_iter_num = ra_config.outer_iter_num;       //!< 外层循环数
  irt_int inner_iter_num = ra_config.inner_iter_num;       //!< 内层循环数

  for (irt_int i = 0, stage = 1; i < outer_iter_num; i++, stage++) {
    double penalty_para = (1 / (2 * initial_penalty));
    LOG_INST.info(Loc::current(), "************* Start iteration penalty_para=", penalty_para, " *************");
    for (irt_int j = 0, iter = 1; j < inner_iter_num; j++, iter++) {
      Monitor iter_monitor;

      calcNablaF(ra_model, penalty_para);
      double norm_nabla_f = calcAlpha(ra_model, penalty_para);
      double norm_square_step = updateResult(ra_model);

      LOG_INST.info(Loc::current(), "Stage(", stage, "/", outer_iter_num, ") Iter(", iter, "/", inner_iter_num,
                    "), norm_nabla_f=", norm_nabla_f, ", norm_square_step=", norm_square_step, iter_monitor.getStatsInfo());
    }
    initial_penalty *= penalty_drop_rate;
  }

  LOG_INST.info(Loc::current(), "The resource model iteration was completed!", monitor.getStatsInfo());
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
    gcell_q_temp += (-2) * ra_gcell.get_public_track_supply();
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

#endif

#if 1  // update ra_model

void ResourceAllocator::updateRAModel(RAModel& ra_model)
{
  updateAllocationMap(ra_model);
  updateOriginRACostMap(ra_model);
}

void ResourceAllocator::updateAllocationMap(RAModel& ra_model)
{
  Die& die = _ra_data_manager.getDatabase().get_die();

  std::vector<RANet>& ra_net_list = ra_model.get_ra_net_list();
  std::vector<double>& result_list = ra_model.get_result_list();

  for (RANet& ra_net : ra_net_list) {
    EXTPlanarRect& bounding_box = ra_net.get_bounding_box();
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
  if (!RTUtil::equalDoubleByError(sum_allocation, sum_cost, DBL_ERROR)) {
    LOG_INST.error(Loc::current(), "The total allocation '", sum_allocation, "' is not equal to the total cost '", sum_cost, "'!");
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

void ResourceAllocator::updateOriginRACostMap(RAModel& ra_model)
{
  for (RANet& ra_net : ra_model.get_ra_net_list()) {
    ra_net.get_origin_net()->set_ra_cost_map(ra_net.get_ra_cost_map());
  }
}

#endif

#if 1  // report ra_model

void ResourceAllocator::reportRAModel(RAModel& ra_model)
{
  countRAModel(ra_model);
  reportTable(ra_model);
}

void ResourceAllocator::countRAModel(RAModel& ra_model)
{
  RAModelStat& ra_model_stat = ra_model.get_ra_model_stat();
  std::vector<double>& avg_cost_list = ra_model_stat.get_avg_cost_list();

  double max_cost = -DBL_MAX;
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
    max_cost = std::max(max_cost, cost_value);
    avg_cost_list.push_back(cost_value);
  }
  ra_model_stat.set_max_avg_cost(max_cost);
}

void ResourceAllocator::reportTable(RAModel& ra_model)
{
  RAModelStat& ra_model_stat = ra_model.get_ra_model_stat();

  std::vector<double>& avg_cost_list = ra_model_stat.get_avg_cost_list();
  double avg_cost_range = RTUtil::getScaleRange(avg_cost_list);
  GridMap<double> avg_cost_map = RTUtil::getRangeNumRatioMap(avg_cost_list);

  fort::char_table avg_cost_table;
  avg_cost_table.set_border_style(FT_SOLID_STYLE);

  avg_cost_table << fort::header << "Avg Cost"
                 << "Net Number" << fort::endr;
  for (irt_int y_idx = 0; y_idx < avg_cost_map.get_y_size(); y_idx++) {
    double left = avg_cost_map[0][y_idx];
    double right = left + avg_cost_range;
    std::string range_str;
    if (y_idx == avg_cost_map.get_y_size() - 1) {
      range_str = RTUtil::getString("[", left, ",", ra_model_stat.get_max_avg_cost(), "]");
    } else {
      range_str = RTUtil::getString("[", left, ",", right, ")");
    }
    avg_cost_table << range_str << RTUtil::getString(avg_cost_map[1][y_idx], "(", avg_cost_map[2][y_idx], "%)") << fort::endr;
  }
  avg_cost_table << fort::header << "Total" << avg_cost_list.size() << fort::endr;

  for (std::string table_str : RTUtil::splitString(avg_cost_table.to_string(), '\n')) {
    LOG_INST.info(Loc::current(), table_str);
  }
}

#endif

}  // namespace irt
