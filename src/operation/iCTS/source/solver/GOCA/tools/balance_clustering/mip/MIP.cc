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
/**
 * @file MIP.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "MIP.hh"

#include <scip/scipdefplugins.h>

#include <algorithm>
#include <cmath>

#include "TimingPropagator.hh"
#include "log/Log.hh"
namespace icts {
/**
 * @brief init the MIP solver parameters
 *
 * @param net_num
 * @param max_fanout
 * @param max_cap
 * @param max_net_dist
 * @param p
 * @param q
 * @param r
 */
void MIP::initParameter(const size_t& net_num, const int& max_fanout, const double& max_cap, const int& max_net_dist, const double& p,
                        const double& q, const double& r)
{
  _net_num = net_num;
  _max_fanout = max_fanout;
  _max_cap = max_cap;
  _max_net_dist = max_net_dist;
  _p = p;
  _q = q;
  _r = r;

  _inst_num = _flatten_insts.size();

  std::ranges::for_each(_flatten_insts, [&](const Inst* inst) {
    auto loc = inst->get_location();
    _min_x = std::min(_min_x, loc.x());
    _min_y = std::min(_min_y, loc.y());
    _max_x = std::max(_max_x, loc.x());
    _max_y = std::max(_max_y, loc.y());
  });
}
/**
 * @brief init variable that describes the net to which the inst belongs
 *
 */
void MIP::initEtaVar()
{
  // eta[i][j]: whether inst[i] is in net[j]
  _eta = std::vector<std::vector<SCIP_VAR*>>(_inst_num, std::vector<SCIP_VAR*>(_net_num, NULL));
  for (size_t i = 0; i < _inst_num; ++i) {
    for (size_t j = 0; j < _net_num; ++j) {
      std::string name = "eta_" + std::to_string(i) + "_" + std::to_string(j);
      SCIPcreateVarBasic(_scip, &_eta[i][j], name.c_str(), 0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY);
      SCIPaddVar(_scip, _eta[i][j]);
    }
  }
}
/**
 * @brief init bounding box variable
 *
 */
void MIP::initBoundingVar()
{
  // x_ll[j], x_ur[j], y_ll[j], y_ur[j]: bounding box for net[j]
  _x_ll = std::vector<SCIP_VAR*>(_net_num, NULL);
  _x_ur = std::vector<SCIP_VAR*>(_net_num, NULL);
  _y_ll = std::vector<SCIP_VAR*>(_net_num, NULL);
  _y_ur = std::vector<SCIP_VAR*>(_net_num, NULL);
  for (size_t j = 0; j < _net_num; ++j) {
    std::string x_ll_name = "x_ll_" + std::to_string(j);
    std::string x_ur_name = "x_ur_" + std::to_string(j);
    std::string y_ll_name = "y_ll_" + std::to_string(j);
    std::string y_ur_name = "y_ur_" + std::to_string(j);
    SCIPcreateVarBasic(_scip, &_x_ll[j], x_ll_name.c_str(), _min_x, _max_x, 0.0, SCIP_VARTYPE_INTEGER);
    SCIPcreateVarBasic(_scip, &_x_ur[j], x_ur_name.c_str(), _min_x, _max_x, 0.0, SCIP_VARTYPE_INTEGER);
    SCIPcreateVarBasic(_scip, &_y_ll[j], y_ll_name.c_str(), _min_y, _max_y, 0.0, SCIP_VARTYPE_INTEGER);
    SCIPcreateVarBasic(_scip, &_y_ur[j], y_ur_name.c_str(), _min_y, _max_y, 0.0, SCIP_VARTYPE_INTEGER);
    SCIPaddVar(_scip, _x_ll[j]);
    SCIPaddVar(_scip, _x_ur[j]);
    SCIPaddVar(_scip, _y_ll[j]);
    SCIPaddVar(_scip, _y_ur[j]);
  }
}
/**
 * @brief init min/max delay variable of net
 *
 */
void MIP::initDelayVar()
{
  _net_min_delay = std::vector<SCIP_VAR*>(_net_num, NULL);
  _net_max_delay = std::vector<SCIP_VAR*>(_net_num, NULL);
  for (size_t j = 0; j < _net_num; ++j) {
    std::string net_min_delay_name = "net_min_delay_" + std::to_string(j);
    std::string net_max_delay_name = "net_max_delay_" + std::to_string(j);
    SCIPcreateVarBasic(_scip, &_net_min_delay[j], net_min_delay_name.c_str(), 0.0, SCIPinfinity(_scip), 0.0, SCIP_VARTYPE_CONTINUOUS);
    SCIPcreateVarBasic(_scip, &_net_max_delay[j], net_max_delay_name.c_str(), 0.0, SCIPinfinity(_scip), 0.0, SCIP_VARTYPE_CONTINUOUS);
    SCIPaddVar(_scip, _net_min_delay[j]);
    SCIPaddVar(_scip, _net_max_delay[j]);
  }
  std::string min_var_name = "global_min_delay";
  std::string max_var_name = "global_max_delay";
  SCIPcreateVarBasic(_scip, &_global_min_delay, min_var_name.c_str(), 0.0, SCIPinfinity(_scip), 0.0, SCIP_VARTYPE_CONTINUOUS);
  SCIPcreateVarBasic(_scip, &_global_max_delay, max_var_name.c_str(), 0.0, SCIPinfinity(_scip), 0.0, SCIP_VARTYPE_CONTINUOUS);
  SCIPaddVar(_scip, _global_min_delay);
  SCIPaddVar(_scip, _global_max_delay);
}
/**
 * @brief cluster constraint, each inst is assigned to at least one net
 *
 */
void MIP::addClusterResitrictConstraint()
{
  // cluster restrict constraint
  for (size_t i = 0; i < _inst_num; ++i) {
    SCIP_CONS* restrict_cst;
    std::string name = "restrict_cst_" + std::to_string(i);
    SCIPcreateConsBasicLinear(_scip, &restrict_cst, name.c_str(), 0, nullptr, nullptr, 1.0, 1.0);
    for (size_t j = 0; j < _net_num; ++j) {
      // trick: only consider i >= j, else eta[i][j] be seted 0
      if (i >= j) {
        SCIPaddCoefLinear(_scip, restrict_cst, _eta[i][j], 1.0);
      } else {
        SCIP_CONS* eta_symmetry_cst;
        std::string eta_symmetry_cst_name = "eta_symmetry_cst_" + std::to_string(i) + "_" + std::to_string(j);
        SCIPcreateConsBasicLinear(_scip, &eta_symmetry_cst, eta_symmetry_cst_name.c_str(), 0, nullptr, nullptr, 0.0, 0.0);
        SCIPaddCoefLinear(_scip, eta_symmetry_cst, _eta[i][j], 1.0);
        SCIPaddCons(_scip, eta_symmetry_cst);
      }
    }
    SCIPaddCons(_scip, restrict_cst);
  }
}
/**
 * @brief bounding box constraint for net, is used for describe the HPWL of net
 *
 */
void MIP::addBoundingBoxConstraint()
{
  // bounding box constraint
  for (size_t i = 0; i < _inst_num; ++i) {
    auto loc = _flatten_insts[i]->get_location();
    for (size_t j = 0; j < _net_num; ++j) {
      SCIP_CONS* x_ur_cst;
      std::string x_ur_cst_name = "x_ur_cst_" + std::to_string(i) + "_" + std::to_string(j);
      SCIPcreateConsBasicLinear(_scip, &x_ur_cst, x_ur_cst_name.c_str(), 0, nullptr, nullptr, 0.0, SCIPinfinity(_scip));
      SCIPaddCoefLinear(_scip, x_ur_cst, _x_ur[j], 1.0);
      SCIPaddCoefLinear(_scip, x_ur_cst, _eta[i][j], -1.0 * loc.x());
      SCIPaddCons(_scip, x_ur_cst);

      SCIP_CONS* y_ur_cst;
      std::string y_ur_cst_name = "y_ur_cst_" + std::to_string(i) + "_" + std::to_string(j);
      SCIPcreateConsBasicLinear(_scip, &y_ur_cst, y_ur_cst_name.c_str(), 0, nullptr, nullptr, 0.0, SCIPinfinity(_scip));
      SCIPaddCoefLinear(_scip, y_ur_cst, _y_ur[j], 1.0);
      SCIPaddCoefLinear(_scip, y_ur_cst, _eta[i][j], -1.0 * loc.y());
      SCIPaddCons(_scip, y_ur_cst);

      SCIP_CONS* x_ll_cst;
      std::string x_ll_cst_name = "x_ll_cst_" + std::to_string(i) + "_" + std::to_string(j);
      SCIPcreateConsBasicLinear(_scip, &x_ll_cst, x_ll_cst_name.c_str(), 0, nullptr, nullptr, -SCIPinfinity(_scip), loc.x() + _lambda);
      SCIPaddCoefLinear(_scip, x_ll_cst, _x_ll[j], 1.0);
      SCIPaddCoefLinear(_scip, x_ll_cst, _eta[i][j], _lambda);
      SCIPaddCons(_scip, x_ll_cst);

      SCIP_CONS* y_ll_cst;
      std::string y_ll_cst_name = "y_ll_cst_" + std::to_string(i) + "_" + std::to_string(j);
      SCIPcreateConsBasicLinear(_scip, &y_ll_cst, y_ll_cst_name.c_str(), 0, nullptr, nullptr, -SCIPinfinity(_scip), loc.y() + _lambda);
      SCIPaddCoefLinear(_scip, y_ll_cst, _y_ll[j], 1.0);
      SCIPaddCoefLinear(_scip, y_ll_cst, _eta[i][j], _lambda);
      SCIPaddCons(_scip, y_ll_cst);
    }
  }
  for (size_t j = 0; j < _net_num; ++j) {
    SCIP_CONS* x_partial_cst;
    std::string x_partial_cst_name = "x_partial_cst_" + std::to_string(j) + "_" + std::to_string(j);
    SCIPcreateConsBasicLinear(_scip, &x_partial_cst, x_partial_cst_name.c_str(), 0, nullptr, nullptr, -SCIPinfinity(_scip), 0.0);
    SCIPaddCoefLinear(_scip, x_partial_cst, _x_ll[j], 1.0);
    SCIPaddCoefLinear(_scip, x_partial_cst, _x_ur[j], -1.0);
    SCIPaddCons(_scip, x_partial_cst);

    SCIP_CONS* y_partial_cst;
    std::string y_partial_cst_name = "y_partial_cst_" + std::to_string(j) + "_" + std::to_string(j);
    SCIPcreateConsBasicLinear(_scip, &y_partial_cst, y_partial_cst_name.c_str(), 0, nullptr, nullptr, -SCIPinfinity(_scip), 0.0);
    SCIPaddCoefLinear(_scip, y_partial_cst, _y_ll[j], 1.0);
    SCIPaddCoefLinear(_scip, y_partial_cst, _y_ur[j], -1.0);
    SCIPaddCons(_scip, y_partial_cst);
  }
}
/**
 * @brief net wirelength constraint, is modeled as a linear combination of fanout and HPWL,
 *         can't exceed _max_net_dist
 *
 */
void MIP::addNetLengthConstraint()
{
  // net length constraint
  for (size_t j = 0; j < _net_num; ++j) {
    SCIP_CONS* net_len_cst;
    std::string net_len_cst_name = "net_len_cst_" + std::to_string(j);
    SCIPcreateConsBasicLinear(_scip, &net_len_cst, net_len_cst_name.c_str(), 0, nullptr, nullptr, -SCIPinfinity(_scip), _max_net_dist);
    SCIPaddCoefLinear(_scip, net_len_cst, _x_ur[j], 1.3);
    SCIPaddCoefLinear(_scip, net_len_cst, _x_ll[j], -1.3);
    SCIPaddCoefLinear(_scip, net_len_cst, _y_ur[j], 1.3);
    SCIPaddCoefLinear(_scip, net_len_cst, _y_ll[j], -1.3);
    auto fanout_net_len_factor = 2.53;
    for (size_t i = 0; i < _inst_num; ++i) {
      SCIPaddCoefLinear(_scip, net_len_cst, _eta[i][j], fanout_net_len_factor);
    }
    SCIPaddCons(_scip, net_len_cst);
  }
}
/**
 * @brief cap load constraint of net, is modeled as sum of pins' cap and wire cap,
 *         can't exceed _max_cap
 *
 */
void MIP::addCapConstraint()
{
  // cap constraint
  for (size_t j = 0; j < _net_num; ++j) {
    SCIP_CONS* cap_cst;
    std::string cap_cst_name = "cap_cst_" + std::to_string(j);
    SCIPcreateConsBasicLinear(_scip, &cap_cst, cap_cst_name.c_str(), 0, nullptr, nullptr, -SCIPinfinity(_scip), _max_cap);
    // pin cap
    for (size_t i = 0; i < _inst_num; ++i) {
      SCIPaddCoefLinear(_scip, cap_cst, _eta[i][j], _flatten_insts[i]->getCapLoad());
    }
    // wire cap
    auto cap_factor = TimingPropagator::getUnitCap() / TimingPropagator::getDbUnit();
    auto factor = 1.3;
    SCIPaddCoefLinear(_scip, cap_cst, _x_ur[j], factor * cap_factor);
    SCIPaddCoefLinear(_scip, cap_cst, _x_ll[j], -factor * cap_factor);
    SCIPaddCoefLinear(_scip, cap_cst, _y_ur[j], factor * cap_factor);
    SCIPaddCoefLinear(_scip, cap_cst, _y_ll[j], -factor * cap_factor);
    auto fanout_net_len_factor = 2.53;
    for (size_t i = 0; i < _inst_num; ++i) {
      SCIPaddCoefLinear(_scip, cap_cst, _eta[i][j], fanout_net_len_factor * cap_factor);
    }
    SCIPaddCons(_scip, cap_cst);
  }
}
/**
 * @brief fanout constraint, which can't exceed _max_fanout
 *
 */
void MIP::addFanoutConstraint()
{
  // fanout constraint
  for (size_t j = 0; j < _net_num; ++j) {
    SCIP_CONS* fanout_cst;
    std::string fanout_cst_name = "fanout_cst_" + std::to_string(j);
    SCIPcreateConsBasicLinear(_scip, &fanout_cst, fanout_cst_name.c_str(), 0, nullptr, nullptr, -SCIPinfinity(_scip), _max_fanout);
    for (size_t i = 0; i < _inst_num; ++i) {
      SCIPaddCoefLinear(_scip, fanout_cst, _eta[i][j], 1.0);
    }
    SCIPaddCons(_scip, fanout_cst);
  }
}
/**
 * @brief delay constraint, is used for describe the min/max delay of net
 *
 */
void MIP::addDelayConstraint()
{
  // delay constraint
  for (size_t i = 0; i < _inst_num; ++i) {
    auto* load_pin = _flatten_insts[i]->get_load_pin();
    auto min_delay = load_pin->get_min_delay();
    auto max_delay = load_pin->get_max_delay();
    for (size_t j = 0; j < _net_num; ++j) {
      SCIP_CONS* net_min_delay_cst;
      std::string net_min_delay_cst_name = "net_min_delay_cst_" + std::to_string(i) + "_" + std::to_string(j);
      SCIPcreateConsBasicLinear(_scip, &net_min_delay_cst, net_min_delay_cst_name.c_str(), 0, nullptr, nullptr, -SCIPinfinity(_scip),
                                _lambda + min_delay);
      SCIPaddCoefLinear(_scip, net_min_delay_cst, _net_min_delay[j], 1.0);
      SCIPaddCoefLinear(_scip, net_min_delay_cst, _eta[i][j], _lambda);
      // net wire delay
      auto wire_factor = _max_net_dist * TimingPropagator::getUnitCap() * TimingPropagator::getUnitRes()
                         / std::pow(TimingPropagator::getDbUnit(), 2);
      auto cap_factor
          = TimingPropagator::getMaxSizeLib()->get_delay_coef().back() * TimingPropagator::getUnitCap() / TimingPropagator::getDbUnit();
      auto factor = 1.3;
      SCIPaddCoefLinear(_scip, net_min_delay_cst, _x_ur[j], -factor * (wire_factor + cap_factor));
      SCIPaddCoefLinear(_scip, net_min_delay_cst, _x_ll[j], factor * (wire_factor + cap_factor));
      SCIPaddCoefLinear(_scip, net_min_delay_cst, _y_ur[j], -factor * (wire_factor + cap_factor));
      SCIPaddCoefLinear(_scip, net_min_delay_cst, _y_ll[j], factor * (wire_factor + cap_factor));
      auto fanout_net_len_factor = 2.53;
      for (size_t i = 0; i < _inst_num; ++i) {
        SCIPaddCoefLinear(_scip, net_min_delay_cst, _eta[i][j], -fanout_net_len_factor * (wire_factor + cap_factor));
      }
      // cell delay

      SCIPaddCons(_scip, net_min_delay_cst);

      SCIP_CONS* net_max_delay_cst;
      std::string net_max_delay_cst_name = "net_max_delay_cst_" + std::to_string(i) + "_" + std::to_string(j);
      SCIPcreateConsBasicLinear(_scip, &net_max_delay_cst, net_max_delay_cst_name.c_str(), 0, nullptr, nullptr, 0.0, SCIPinfinity(_scip));
      SCIPaddCoefLinear(_scip, net_max_delay_cst, _net_max_delay[j], 1.0);
      SCIPaddCoefLinear(_scip, net_max_delay_cst, _eta[i][j], -max_delay);
      // net wire delay
      SCIPaddCoefLinear(_scip, net_max_delay_cst, _x_ur[j], -factor * (wire_factor + cap_factor));
      SCIPaddCoefLinear(_scip, net_max_delay_cst, _x_ll[j], factor * (wire_factor + cap_factor));
      SCIPaddCoefLinear(_scip, net_max_delay_cst, _y_ur[j], -factor * (wire_factor + cap_factor));
      SCIPaddCoefLinear(_scip, net_max_delay_cst, _y_ll[j], factor * (wire_factor + cap_factor));
      for (size_t i = 0; i < _inst_num; ++i) {
        SCIPaddCoefLinear(_scip, net_max_delay_cst, _eta[i][j], -fanout_net_len_factor * (wire_factor + cap_factor));
      }
      SCIPaddCons(_scip, net_max_delay_cst);
    }
  }
  // global delay constraint
  for (size_t j = 0; j < _net_num; ++j) {
    SCIP_CONS* global_min_delay_cst;
    std::string global_min_delay_cst_name = "global_min_delay_cst_" + std::to_string(j);
    SCIPcreateConsBasicLinear(_scip, &global_min_delay_cst, global_min_delay_cst_name.c_str(), 0, nullptr, nullptr, -SCIPinfinity(_scip),
                              0.0);
    SCIPaddCoefLinear(_scip, global_min_delay_cst, _global_min_delay, 1.0);
    SCIPaddCoefLinear(_scip, global_min_delay_cst, _net_min_delay[j], -1.0);
    SCIPaddCons(_scip, global_min_delay_cst);

    SCIP_CONS* global_max_delay_cst;
    std::string global_max_delay_cst_name = "global_max_delay_cst_" + std::to_string(j);
    SCIPcreateConsBasicLinear(_scip, &global_max_delay_cst, global_max_delay_cst_name.c_str(), 0, nullptr, nullptr, 0.0,
                              SCIPinfinity(_scip));
    SCIPaddCoefLinear(_scip, global_max_delay_cst, _global_max_delay, 1.0);
    SCIPaddCoefLinear(_scip, global_max_delay_cst, _net_max_delay[j], -1.0);
    SCIPaddCons(_scip, global_max_delay_cst);

    SCIP_CONS* partial_delay_cst;
    std::string partial_delay_cst_name = "partial_delay_cst_" + std::to_string(j);
    SCIPcreateConsBasicLinear(_scip, &partial_delay_cst, partial_delay_cst_name.c_str(), 0, nullptr, nullptr, -SCIPinfinity(_scip), 0.0);
    SCIPaddCoefLinear(_scip, partial_delay_cst, _net_min_delay[j], 1.0);
    SCIPaddCoefLinear(_scip, partial_delay_cst, _net_max_delay[j], -1.0);
    SCIPaddCons(_scip, partial_delay_cst);
  }
}
/**
 * @brief objective function, include net wirelength, net skew and global skew
 *
 */
void MIP::addObjFunc()
{
  // net wirelength cost: \sum (p * WL[j]), WL[j] = 2.0 * HPWL[j]
  for (size_t j = 0; j < _net_num; ++j) {
    SCIPaddVarObj(_scip, _x_ur[j], _p * 2.0);
    SCIPaddVarObj(_scip, _x_ll[j], -_p * 2.0);
    SCIPaddVarObj(_scip, _y_ur[j], _p * 2.0);
    SCIPaddVarObj(_scip, _y_ll[j], -_p * 2.0);
  }
  // net skew cost:
  for (size_t j = 0; j < _net_num; ++j) {
    SCIPaddVarObj(_scip, _net_max_delay[j], _q);
    SCIPaddVarObj(_scip, _net_min_delay[j], -_q);
  }
  // skew cost: r * (delay_max - delay_min)
  SCIPaddVarObj(_scip, _global_max_delay, _r);
  SCIPaddVarObj(_scip, _global_min_delay, -_r);
}
/**
 * @brief release variables
 *
 */
void MIP::freeVars()
{
  auto free_func = [&](SCIP_VAR* var) { SCIPreleaseVar(_scip, &var); };
  // variable free
  std::ranges::for_each(_eta, [&](std::vector<SCIP_VAR*>& vars) {
    std::ranges::for_each(vars, free_func);
    vars.clear();
  });
  _eta.clear();
  std::ranges::for_each(_x_ll, free_func);
  _x_ll.clear();
  std::ranges::for_each(_x_ur, free_func);
  _x_ur.clear();
  std::ranges::for_each(_y_ll, free_func);
  _y_ll.clear();
  std::ranges::for_each(_y_ur, free_func);
  _y_ur.clear();
  std::ranges::for_each(_net_min_delay, free_func);
  _net_min_delay.clear();
  std::ranges::for_each(_net_max_delay, free_func);
  _net_max_delay.clear();
  SCIPreleaseVar(_scip, &_global_min_delay);
  _global_min_delay = nullptr;
  SCIPreleaseVar(_scip, &_global_max_delay);
  _global_max_delay = nullptr;
  std::ranges::for_each(_aux_vars, free_func);
  _aux_vars.clear();
}
/**
 * @brief flow interface
 *
 * @return std::vector<std::vector<Inst*>>
 */
std::vector<std::vector<Inst*>> MIP::run()
{
  /**
   * @brief SCIP initialization
   *
   */
  SCIPcreate(&_scip);
  // Initialize SCIP
  SCIPincludeDefaultPlugins(_scip);
  SCIPsetMessagehdlrQuiet(_scip, TRUE);
  // Create problem
  SCIPcreateProb(_scip, "MIP-Clustering", NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  // SCIPsetRealParam(_scip, "limits/gap", 0.1);
  SCIPsetRealParam(_scip, "limits/time", 40);
  SCIPsetObjsense(_scip, SCIP_OBJSENSE::SCIP_OBJSENSE_MINIMIZE);
  /**
   * @brief Create variables
   *
   */
  initEtaVar();
  initBoundingVar();
  initDelayVar();

  /**
   * @brief Create constraints
   *
   */
  addClusterResitrictConstraint();
  addBoundingBoxConstraint();
  addNetLengthConstraint();
  addCapConstraint();
  addFanoutConstraint();
  addDelayConstraint();
  // addBoundingToleranceConstraint();
  // addShapingToleranceConstraint();

  /**
   * @brief Set objective function
   *
   */
  addObjFunc();

  /**
   * @brief Solve the problem
   *
   */
  SCIPsolve(_scip);

  /**
   * @brief Get variable values
   *
   */
  // Get the solution
  SCIP_SOL* solution = SCIPgetBestSol(_scip);
  if (solution == NULL) {
    LOG_ERROR << "Can't find the MIP clustering solution";
    return {};
  }

  std::vector<std::vector<Inst*>> clusters;
  for (size_t j = 0; j < _net_num; ++j) {
    std::vector<Inst*> cluster;
    for (size_t i = 0; i < _inst_num; ++i) {
      auto eta_ij = SCIPgetSolVal(_scip, solution, _eta[i][j]);
      if (eta_ij > 0.5) {
        cluster.push_back(_flatten_insts[i]);
      }
    }
    clusters.push_back(cluster);

    // check
    LOG_INFO << "No." << j << " net has " << clusters.back().size() << " insts";
    auto x_ll = SCIPgetSolVal(_scip, solution, _x_ll[j]);
    auto x_ur = SCIPgetSolVal(_scip, solution, _x_ur[j]);
    auto y_ll = SCIPgetSolVal(_scip, solution, _y_ll[j]);
    auto y_ur = SCIPgetSolVal(_scip, solution, _y_ur[j]);
    LOG_INFO << "Bounding box: " << x_ll << " " << x_ur << " " << y_ll << " " << y_ur;
    auto net_dist = x_ur - x_ll + y_ur - y_ll;
    auto net_len = (1.3 * net_dist + 2.53 * cluster.size()) / TimingPropagator::getDbUnit();
    auto max_len = _max_net_dist / TimingPropagator::getDbUnit();
    auto net_cap = net_len * TimingPropagator::getUnitCap();
    std::ranges::for_each(cluster, [&](Inst* inst) { net_cap += inst->getCapLoad(); });
    auto net_min_delay = SCIPgetSolVal(_scip, solution, _net_min_delay[j]);
    auto net_max_delay = SCIPgetSolVal(_scip, solution, _net_max_delay[j]);
    LOG_INFO << "Net length: " << net_len << " Max length: " << max_len;
    LOG_INFO << "Net cap: " << net_cap << " Max cap: " << _max_cap;
    LOG_INFO << "Net min delay: " << net_min_delay << " Max delay: " << net_max_delay;
    auto global_min_delay = SCIPgetSolVal(_scip, solution, _global_min_delay);
    auto global_max_delay = SCIPgetSolVal(_scip, solution, _global_max_delay);
    LOG_INFO << "Global min delay: " << global_min_delay << " Max delay: " << global_max_delay << "\n\n";
  }
  /**
   * @brief Free resources
   *
   */
  freeVars();
  // SCIP free
  SCIPfree(&_scip);

  return clusters;
}

}  // namespace icts