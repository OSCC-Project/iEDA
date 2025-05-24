// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file StaArc.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of timing arc, which is used for static timing analysis.
 * @version 0.1
 * @date 2021-02-10
 */
#pragma once

#include "StaData.hh"
#include "liberty/Lib.hh"
#include "netlist/Instance.hh"
#include "netlist/Net.hh"
#include "propagation-cuda/propagation.cuh"

namespace ista {
class StaVertex;
class StaFunc;
class Lib_Arc_GPU;

/**
 * @brief The static timing analysis DAG edge.
 *
 */
class StaArc {
 public:
  StaArc(StaVertex* src, StaVertex* snk);
  virtual ~StaArc() = default;

  StaVertex* get_src() const { return _src; }
  void set_src(StaVertex* src_vertex) { _src = src_vertex; }
  StaVertex* get_snk() const { return _snk; }
  void set_snk(StaVertex* snk_vertex) { _snk = snk_vertex; }

  virtual unsigned isInstArc() const { return 0; }
  virtual unsigned isNetArc() const { return 0; }
  virtual unsigned isBufInvArc() const { return 0; }

  virtual unsigned isDelayArc() const { return 0; }
  virtual unsigned isCheckArc() const { return 0; }
  virtual unsigned isMpwArc() const { return 0; }

  virtual unsigned isPositiveArc() const { return 0; }
  virtual unsigned isNegativeArc() const { return 0; }
  virtual unsigned isUnateArc() const { return 0; }
  virtual unsigned isSetupArc() const { return 0; }
  virtual unsigned isHoldArc() const { return 0; }
  virtual unsigned isRecoveryArc() const { return 0; }
  virtual unsigned isRemovalArc() const { return 0; }
  virtual unsigned isRisingEdgeCheck() const { return 0; }
  virtual unsigned isFallingEdgeCheck() const { return 0; }
  virtual unsigned isRisingTriggerArc() const { return 0; }
  virtual unsigned isFallingTriggerArc() const { return 0; }

  void addData(StaArcDelayData* arc_delay_data);
  void resetArcDelayBucket() { _arc_delay_bucket.freeData(); }
  unsigned isResetArcDelayBucket() { return (_arc_delay_bucket.isFreeData()); }
  int get_arc_delay(AnalysisMode analysis_mode, TransType trans_type);
  void initArcDelayData();
  StaArcDelayData* getArcDelayData(AnalysisMode analysis_mode,
                                   TransType trans_type);
  StaDataBucket& getDataBucket() { return _arc_delay_bucket; }

  [[nodiscard]] unsigned is_loop_disable() const { return _is_loop_disable; }
  void set_is_loop_disable(bool is_set) { _is_loop_disable = is_set; }

  [[nodiscard]] unsigned is_disable_arc() const { return _is_disable_arc; }
  void set_is_disable_arc(bool is_set) { _is_disable_arc = is_set; }

  unsigned exec(StaFunc& func);

  void dump();

 private:
  StaVertex* _src;
  StaVertex* _snk;
  StaDataBucket _arc_delay_bucket;

  unsigned _is_loop_disable : 1 = 0;
  unsigned _is_disable_arc : 1 = 0;
  unsigned _reserved : 30 = 0;
  FORBIDDEN_COPY(StaArc);
};

/**
 * @brief Traverse the data bucket data of the arc, usage:
 * StaArc* arc;
 * StaData* arc_delay_data;
 * FOREACH_DELAY_DATA(arc, delay_data)
 * {
 *    do_something_for_delay_data();
 * }
 */
#define FOREACH_ARC_DELAY_DATA(arc, arc_delay_data)      \
  for (StaDataBucketIterator iter(arc->getDataBucket()); \
       iter.hasNext() ? arc_delay_data = iter.next().get(), true : false;)

/**
 * @brief The static timing analysis DAG edge, which map to the net arc.
 *
 */
class StaNetArc : public StaArc {
 public:
  StaNetArc(StaVertex* driver, StaVertex* load, Net* net);
  ~StaNetArc() override = default;

  unsigned isNetArc() const override { return 1; }
  unsigned isDelayArc() const override { return 1; }
  unsigned isCheckArc() const override { return 0; }
  unsigned isMpwArc() const override { return 0; }
  unsigned isPositiveArc() const override { return 1; }
  unsigned isUnateArc() const override { return 1; }

  Net* get_net() { return _net; }
  void set_net(Net* net) { _net = net; }

  void addWaveformData(StaArcWaveformData* arc_waveform_data) {
    if (arc_waveform_data) {
      _arc_waveform_bucket.addData(arc_waveform_data, 0);
    }
  }
  void resetArcDelayBucket() { _arc_waveform_bucket.freeData(); }
  auto& getWaveformBucket() { return _arc_waveform_bucket; }

  void updateCrosstalkDelay(AnalysisMode delay_type, TransType trans_type,
                            int crosstalk_delay) {
    auto* arc_delay = getArcDelayData(delay_type, trans_type);
    if (crosstalk_delay > arc_delay->get_crosstalk_delay()) {
      arc_delay->set_crosstalk_delay(crosstalk_delay);
    }
  }
  std::optional<int> getCrossTalkDelay(AnalysisMode delay_type,
                                       TransType trans_type) {
    auto* arc_delay = getArcDelayData(delay_type, trans_type);
    if (arc_delay) {
      return arc_delay->get_crosstalk_delay();
    }
    return std::nullopt;
  }
  std::optional<double> getCrossTalkDelayNs(AnalysisMode delay_type,
                                            TransType trans_type) {
    auto crosstalk_delay_fs = getCrossTalkDelay(delay_type, trans_type);
    if (crosstalk_delay_fs) {
      return FS_TO_NS(crosstalk_delay_fs.value());
    }

    return std::nullopt;
  }

 private:
  Net* _net;

  StaDataBucket _arc_waveform_bucket;

  FORBIDDEN_COPY(StaNetArc);
};

/**
 * @brief Traverse the waveform data bucket data of the vertex, usage:
 * StaNetArc* arc;
 * StaData* arc_waveform_data;
 * FOREACH_ARC_WAVEFORM_DATA(arc, arc_waveform_data)
 * {
 *    do_something_for_waveform_data();
 * }
 */
#define FOREACH_ARC_WAVEFORM_DATA(arc, arc_waveform_data)       \
  for (StaDataBucketIterator iter(                              \
           dynamic_cast<StaNetArc*>(arc)->getWaveformBucket()); \
       iter.hasNext() ? arc_waveform_data = iter.next().get(), true : false;)

/**
 * @brief The static timing analysis DAG edge, which map to inst arc.
 *
 */
class StaInstArc : public StaArc {
 public:
  StaInstArc(StaVertex* src, StaVertex* snk, LibArc* lib_arc, Instance* inst);
  ~StaInstArc() override = default;
  // ~StaInstArc() override = default;

  unsigned isInstArc() const override { return 1; }

  LibArc* get_lib_arc() { return _lib_arc; }
  void set_lib_arc(LibArc* lib_arc) { _lib_arc = lib_arc; }

  unsigned isDelayArc() const override { return _lib_arc->isDelayArc(); }
  unsigned isCheckArc() const override { return _lib_arc->isCheckArc(); }
  unsigned isMpwArc() const override { return _lib_arc->isMpwArc(); }
  unsigned isBufInvArc() const override {
    return _lib_arc->get_owner_cell()->isBuffer() ||
           _lib_arc->get_owner_cell()->isInverter();
  }

  unsigned isSetupArc() const override { return _lib_arc->isSetupArc(); }
  unsigned isHoldArc() const override { return _lib_arc->isHoldArc(); }
  unsigned isRecoveryArc() const override { return _lib_arc->isRecoveryArc(); }
  unsigned isRemovalArc() const override { return _lib_arc->isRemovalArc(); }

  unsigned isPositiveArc() const override { return _lib_arc->isPositiveArc(); }
  unsigned isNegativeArc() const override { return _lib_arc->isNegativeArc(); }
  unsigned isUnateArc() const override { return _lib_arc->isUnateArc(); }

  unsigned isRisingEdgeCheck() const override {
    return _lib_arc->isRisingEdgeCheck();
  }
  unsigned isFallingEdgeCheck() const override {
    return _lib_arc->isFallingEdgeCheck();
  }

  unsigned isRisingTriggerArc() const override {
    return _lib_arc->isRisingTriggerArc();
  }
  unsigned isFallingTriggerArc() const override {
    return _lib_arc->isFallingTriggerArc();
  }

  LibArc::TimingType getTimingType() { return _lib_arc->get_timing_type(); }

  auto* get_inst() { return _inst; }

#if CUDA_PROPAGATION
  Lib_Arc_GPU* get_lib_gpu_arc() const { return _lib_gpu_arc; }
  void set_lib_gpu_arc(Lib_Arc_GPU* lib_gpu_arc) { _lib_gpu_arc = lib_gpu_arc; }

  int get_lib_arc_id() const { return _lib_arc_id; }
  void set_lib_arc_id(int arc_id) { _lib_arc_id = arc_id; }
#endif

 private:
  LibArc* _lib_arc;  //!< The mapped to lib arc.
  Instance* _inst;   //!< The owned inst.

#if CUDA_PROPAGATION
  Lib_Arc_GPU* _lib_gpu_arc = nullptr;  //!< The gpu lib arc.
  int _lib_arc_id = -1; //!< The arc id for gpu lib data.
#endif

  FORBIDDEN_COPY(StaInstArc);
};

}  // namespace ista
