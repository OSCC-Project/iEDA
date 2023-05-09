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
 * @file CrossTalkDelayCalc.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class two pi model of the crosstalk delay calculation.
 * @version 0.1
 * @date 2022-09-16
 */
#pragma once

#include "ElmoreDelayCalc.hh"

namespace ista {

/**
 * @brief The two pi model for crosstalk delay model.
 * reference:Jason Cong Improved Crosstalk Modeling for Noise Constrained
 * Interconnect Optimization.
 */
class TwoPiModel {
 public:
  TwoPiModel() = default;
  TwoPiModel(double C1, double Rs, double C2, double Re, double CL, double Cx);
  ~TwoPiModel() = default;

  TwoPiModel(const TwoPiModel& orig) = default;
  TwoPiModel& operator=(const TwoPiModel& orig) = default;

  TwoPiModel(TwoPiModel&& other) = default;
  TwoPiModel& operator=(TwoPiModel&& other) = default;

  void set_C1(double C1) { _C1 = C1; }
  double get_C1() const { return _C1; }

  void set_Rs(double Rs) { _Rs = Rs; }
  double get_Rs() const { return _Rs; }

  void set_C2(double C2) { _C2 = C2; }
  double get_C2() const { return _C2; }

  void set_Re(double Re) { _Re = Re; }
  double get_Re() const { return _Re; }

  void set_CL(double CL) { _CL = CL; }
  double get_CL() const { return _CL; }

  void set_Cx(double Cx) { _Cx = Cx; }
  double get_Cx() const { return _Cx; }

  void set_Rd(double Rd) { _Rd = Rd; }
  double get_Rd() const { return _Rd; }

 private:
  double _C1 = 0.0;
  double _Rs = 0.0;
  double _C2 = 0.0;
  double _Re = 0.0;
  double _CL = 0.0;
  double _Cx = 0.0;  //!< The coupled cap.
  double _Rd = 0.0;  //!< The driver inner resistance.
};

/**
 * @brief The func for crosstalk delay calc.
 *
 */
class CrossTalkDelayCalc {
 public:
  CrossTalkDelayCalc(Net* remote_net, Net* local_net)
      : _remote_net(remote_net), _local_net(local_net) {}
  ~CrossTalkDelayCalc() = default;

  CrossTalkDelayCalc(CrossTalkDelayCalc&& orig) = default;
  CrossTalkDelayCalc& operator=(CrossTalkDelayCalc&& orig) = default;

  void reduceRCTreeToTwoPiModel(RcNet* rc_net, RctNode* coupled_point);
  void calcNoiseAmplitude(double input_arrive_time, double input_transition);
  double calcNoiseVoltage(double time) const;

  auto* get_remote_net() { return _remote_net; }
  auto* get_local_net() { return _local_net; }

  double get_input_arrive_time() { return _input_arrive_time; }

 private:
  Net* _remote_net;
  Net* _local_net;

  std::optional<TwoPiModel> _two_pi_model;

  double _input_arrive_time = 0.0;
  double _tr_coeff = 1.0;  //!< please reference the paper <<Improved Crosstalk
                           //!< Modeling for Noise Constrained Interconnect
                           //!< Optimization>>. The tr is input transition time.
  double _tx_coeff = 0.0;
  double _tv_coeff = 1.0;
};

}  // namespace ista
