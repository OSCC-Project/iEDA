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
 * @file StaVetex.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The static timing analysis DAG vertex, which map to inst pin or port.
 * @version 0.1
 * @date 2021-02-10
 */

#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "StaData.hh"
#include "netlist/Netlist.hh"

namespace ista {

class StaFunc;
class StaArc;

/**
 * @brief The tag class of propagation.
 *
 */
class StaPropagationTag {
 public:
  enum class TagType : unsigned {
    kProp = 0,
    kFalse = 1,
    kMinMax = 2,
    kMulticycle = 3
  };
  StaPropagationTag() = default;
  ~StaPropagationTag() = default;
  StaPropagationTag(const StaPropagationTag& orig);
  StaPropagationTag& operator=(const StaPropagationTag& orig);

  StaPropagationTag* copy() { return new StaPropagationTag(*this); }

  void set_is_prop(unsigned is_prop) { _is_prop = is_prop; }
  [[nodiscard]] unsigned is_prop() const { return _is_prop; }

  void set_is_through_point(unsigned is_through_point) {
    _is_through_point = is_through_point;
  }
  [[nodiscard]] unsigned is_through_point() const { return _is_through_point; }

  void set_is_searched(unsigned is_searched) { _is_searched = is_searched; }
  [[nodiscard]] unsigned is_searched() const { return _is_searched; }

  void set_is_collected(unsigned is_collected) { _is_collected = is_collected; }
  [[nodiscard]] unsigned is_collected() const { return _is_collected; }

  void set_is_false_path(unsigned is_false_path) {
    _is_false_path = is_false_path;
  }
  [[nodiscard]] unsigned is_false_path() const { return _is_false_path; }

  void set_is_min_max_delay(unsigned is_min_max_delay) {
    _is_min_max_delay = is_min_max_delay;
  }
  [[nodiscard]] unsigned is_min_max_delay() const { return _is_min_max_delay; }

  void set_is_multicycle_path(unsigned is_multicycle_path) {
    _is_multicycle_path = is_multicycle_path;
  }
  [[nodiscard]] unsigned is_multicycle_path() const {
    return _is_multicycle_path;
  }
  void setTag(TagType tag_type, bool is_set);
  [[nodiscard]] bool isTagSet(TagType tag_type) const;

 private:
  unsigned _is_through_point : 1 = 0;
  unsigned _is_searched : 1 = 0;
  unsigned _is_collected : 1 = 0;
  unsigned _is_prop : 1 = 1;  // default set all vertex prop.
  unsigned _is_false_path : 1 = 0;
  unsigned _is_min_max_delay : 1 = 0;
  unsigned _is_multicycle_path : 1 = 0;
  unsigned _reserved : 25 = 0;
};

/**
 * @brief The static timing analysis DAG vetex, which map to the pin.
 *
 */
class StaVertex {
 public:
  enum Color : unsigned { kWhite = 0, kGray = 1, kBlack = 2 };
  enum RiseFall : int { kRise = 0, kFall = 1 };
  explicit StaVertex(DesignObject* obj);
  ~StaVertex() = default;

  void addSrcArc(StaArc* src_arc) { _src_arcs.push_back(src_arc); }
  void addSnkArc(StaArc* snk_arc) { _snk_arcs.push_back(snk_arc); }
  void removeSrcArc(StaArc* src_arc) {
    LOG_FATAL_IF(!std::erase_if(
        _src_arcs, [src_arc](StaArc* arc) { return arc == src_arc; }));
  }
  void removeSnkArc(StaArc* snk_arc) {
    LOG_FATAL_IF(!std::erase_if(
        _snk_arcs, [snk_arc](StaArc* arc) { return arc == snk_arc; }));
  }

  std::vector<StaArc*>& get_src_arcs() { return _src_arcs; }
  std::vector<StaArc*>& get_snk_arcs() { return _snk_arcs; }
  StaArc* getSetupHoldArc(AnalysisMode analysis_mode);
  StaArc* getCheckArc(AnalysisMode analysis_mode);
  std::vector<StaArc*> getSrcCheckArcs(AnalysisMode analysis_mode);

  void clearSrcArcs() { _src_arcs.clear(); }
  void clearSnkArcs() { _snk_arcs.clear(); }
  void clearSrcNetArcs();
  void clearSnkNetArcs();

  DesignObject* get_design_obj() const { return _obj; }

  std::string getName() { return _obj ? _obj->getFullName() : "Nil"; }
  LibCell* getOwnCell();
  const char* getOwnCellName();
  const char* getOwnCellOrPortName();
  std::string getNameWithCellName();
  const char* getOwnInstanceOrPortName();

  void addData(StaSlewData* slew_data);
  void addData(StaClockData* clock_data);
  void addData(StaPathDelayData* delay_data);

  void resetSlewBucket() { _slew_bucket.freeData(); }
  void resetClockBucket() { _clock_bucket.freeData(); }
  void resetPathDelayBucket() { _path_delay_bucket.freeData(); }
  unsigned isResetVertexBucket() {
    return (_slew_bucket.isFreeData() && _clock_bucket.isFreeData() &&
            _path_delay_bucket.isFreeData());
  }
  void resetVertexArcData();

  void initSlewData();
  void initPathDelayData();

  StaDataBucket& getSlewBucket() { return _slew_bucket; }
  StaDataBucket& getClockBucket() { return _clock_bucket; }
  StaDataBucket& getDataBucket() { return _path_delay_bucket; }

  std::vector<StaArc*> getSrcArc(StaVertex* src_vertex);
  std::vector<StaArc*> getSnkArc(StaVertex* snk_vertex);

  unsigned is_clock() const { return _is_clock; }
  void set_is_clock() { _is_clock = 1; }

  unsigned is_clock_gate_clock() const { return _is_clock_gate_clock; }
  void set_is_clock_gate_clock() { _is_clock_gate_clock = 1; }

  unsigned is_port() const { return _is_port; }
  void set_is_port() { _is_port = 1; }

  unsigned is_start() const { return _is_start; }
  void set_is_start() { _is_start = 1; }
  void reset_is_start() { _is_start = 0; }

  unsigned is_end() const { return _is_end; }
  void set_is_end() { _is_end = 1; }

  unsigned is_clock_gate_end() const { return _is_clock_gate_end; }
  void set_is_clock_gate_end() { _is_clock_gate_end = 1; }

  unsigned is_const() const { return _is_const; }
  void set_is_const() { _is_const = 1; }

  unsigned is_const_vdd() const { return _is_const_vdd; }
  void set_is_const_vdd() { _is_const_vdd = 1; }

  unsigned is_const_gnd() const { return _is_const_gnd; }
  void set_is_const_gnd() { _is_const_gnd = 1; }

  unsigned is_slew_prop() const { return _is_slew_prop; }
  void set_is_slew_prop() { _is_slew_prop = 1; }
  void reset_is_slew_prop() { _is_slew_prop = 0; }

  unsigned is_delay_prop() const { return _is_delay_prop; }
  void set_is_delay_prop() { _is_delay_prop = 1; }
  void reset_is_delay_prop() { _is_delay_prop = 0; }

  unsigned is_bwd() const { return _is_bwd; }
  void set_is_bwd() { _is_bwd = 1; }
  void reset_is_bwd() { _is_bwd = 0; }

  unsigned is_fwd() const { return _is_fwd; }
  void set_is_fwd() { _is_fwd = 1; }
  void reset_is_fwd() { _is_fwd = 0; }

  unsigned is_crosstalk_prop() const { return _is_crosstalk_prop; }
  void set_is_crosstalk_prop() { _is_crosstalk_prop = 1; }
  void reset_is_crosstalk_prop() { _is_crosstalk_prop = 0; }

  void set_level(unsigned level);
  unsigned get_level() const { return _level; }
  void resetLevel() { _level = 0; }
  unsigned isSetLevel() { return _level != 0; }

  void set_is_sdc_clock_pin() { _is_sdc_clock_pin = 1; }
  unsigned is_sdc_clock_pin() const { return _is_sdc_clock_pin; }
  void reset_is_sdc_clock_pin() { _is_sdc_clock_pin = 0; }

  void set_is_ideal_clock_latency() { _is_ideal_clock_latency = 1; }
  unsigned is_ideal_clock_latency() const { return _is_ideal_clock_latency; }
  void reset_is_ideal_clock_latency() { _is_ideal_clock_latency = 0; }

  void set_is_bidirection() { _is_bidirection = 1; }
  unsigned is_bidirection() { return _is_bidirection; }
  void reset_is_bidirection() { _is_bidirection = 0; }

  void set_is_assistant() { _is_assistant = 1; }
  unsigned is_assistant() const { return _is_assistant; }
  void reset_is_assistant() { _is_assistant = 0; }

  void set_is_foward_find() { _is_foward_find = 1; }
  unsigned is_foward_find() const { return _is_foward_find; }

  void set_is_fwd_reset() { _is_fwd_reset = 1; }
  unsigned is_fwd_reset() const { return _is_fwd_reset; }

  void set_is_bwd_reset() { _is_bwd_reset = 1; }
  unsigned is_bwd_reset() const { return _is_bwd_reset; }

  void addFanoutEndVertex(StaVertex* fanout_end_vertex) {
    LOG_FATAL_IF(!fanout_end_vertex) << "insert end vertex:nullptr.";
    _fanout_end_vertexes.insert(fanout_end_vertex);
  }
  void addFanoutEndVertex(const BTreeSet<StaVertex*>& fanout_end_vertex_set) {
    std::copy(
        fanout_end_vertex_set.begin(), fanout_end_vertex_set.end(),
        std::inserter(_fanout_end_vertexes, _fanout_end_vertexes.begin()));
  }
  auto& get_fanout_end_vertexes() { return _fanout_end_vertexes; }

  void set_is_backward_find() { _is_backward_find = 1; }
  unsigned is_backward_find() const { return _is_backward_find; }
  void addFaninStartVertex(StaVertex* fanin_start_vertex) {
    LOG_FATAL_IF(!fanin_start_vertex) << "insert start vertex:nullptr.";
    _fanin_start_vertexes.insert(fanin_start_vertex);
  }
  void addFaninStartVertex(const BTreeSet<StaVertex*>& fanin_start_vertex_set) {
    std::copy(
        fanin_start_vertex_set.begin(), fanin_start_vertex_set.end(),
        std::inserter(_fanin_start_vertexes, _fanin_start_vertexes.begin()));
  }
  auto& get_fanin_start_vertexes() { return _fanin_start_vertexes; }

  void set_prop_tag(StaPropagationTag&& prop_tag) {
    _prop_tag = std::move(prop_tag);
  }
  auto& get_prop_tag() { return _prop_tag; }

  void setWhite() { _color = kWhite; }
  void setGray() { _color = kGray; }
  void setBlack() { _color = kBlack; }

  [[nodiscard]] unsigned isWhite() const { return _color == kWhite; }
  [[nodiscard]] unsigned isGray() const { return _color == kGray; }
  [[nodiscard]] unsigned isBlack() const { return _color == kBlack; }
  void resetColor() { _color = kWhite; }

  void setMaxRiseCap(double cap) { _max_cap[kRise] = cap; }
  auto& getMaxRiseCap() { return _max_cap[kRise]; }
  void setMaxFallCap(double cap) { _max_cap[kFall] = cap; }
  auto& getMaxFallCap() { return _max_cap[kFall]; }

  void setMaxRiseSlew(double slew) { _max_slew[kRise] = slew; }
  auto& getMaxRiseSlew() { return _max_slew[kRise]; }
  void setMaxFallSlew(double slew) { _max_slew[kFall] = slew; }
  auto& getMaxFallSlew() { return _max_slew[kFall]; }

  void setMaxFanout(double max_fanout) { _max_fanout = max_fanout; }
  auto& getMaxFanout() { return _max_fanout; }

  std::vector<StaData*> getClockData(AnalysisMode analysis_mode,
                                     TransType trans_type);

  StaClock* getPropClock() {
    auto clock_data = dynamic_cast<StaClockData*>(_clock_bucket.frontData());
    return clock_data ? clock_data->get_prop_clock() : nullptr;
  }
  StaClock* isHavePropClock() {
    if (!_clock_bucket.frontData()) {
      return nullptr;
    } else {
      return dynamic_cast<StaClockData*>(_clock_bucket.frontData())
          ->get_prop_clock();
    }
  }
  StaClock* getPropClock(AnalysisMode analysis_mode, TransType trans_type);
  std::unordered_set<StaClock*> getPropagatedClock(AnalysisMode analysis_mode,
                                                   TransType trans_type,
                                                   bool is_data_path);
  bool isPropClock(const char* clock_name, AnalysisMode analysis_mode,
                   TransType trans_type);

  unsigned isRisingTriggered();
  unsigned isFallingTriggered();

  std::mutex& get_fwd_mutex() { return _fwd_mutex; }
  std::mutex& get_bwd_mutex() { return _bwd_mutex; }

  std::optional<int64_t> getClockArriveTime(
      AnalysisMode analysis_mode, TransType trans_type,
      std::optional<std::string> clock_name = std::nullopt);

  std::optional<int64_t> getArriveTime(AnalysisMode analysis_mode,
                                       TransType trans_type);
  std::optional<double> getArriveTimeNs(AnalysisMode analysis_mode,
                                        TransType trans_type) {
    auto arrive_time = getArriveTime(analysis_mode, trans_type);
    if (arrive_time) {
      return FS_TO_NS(arrive_time.value());
    }

    return std::nullopt;
  }
  std::optional<int> getReqTime(AnalysisMode analysis_mode,
                                TransType trans_type);
  std::optional<double> getReqTimeNs(AnalysisMode analysis_mode,
                                     TransType trans_type) {
    auto req = getReqTime(analysis_mode, trans_type);
    if (req) {
      return FS_TO_NS(*req);
    } else {
      return std::nullopt;
    }
  }
  std::optional<int64_t> getSlack(AnalysisMode analysis_mode,
                                  TransType trans_type);
  std::optional<double> getSlackNs(AnalysisMode analysis_mode,
                                   TransType trans_type) {
    auto slack = getSlack(analysis_mode, trans_type);
    if (slack) {
      return FS_TO_NS(*slack);
    } else {
      return std::nullopt;
    }
  }
  std::optional<double> getWorstSlackNs(AnalysisMode analysis_mode) {
    auto rise_slack = getSlackNs(analysis_mode, TransType::kRise);
    if (rise_slack) {
      auto fall_slack = getSlackNs(analysis_mode, TransType::kFall);
      if (rise_slack && fall_slack) {
        return (*rise_slack < *fall_slack) ? rise_slack : fall_slack;
      }
    }

    return std::nullopt;
  }

  std::optional<double> getTNSNs(AnalysisMode analysis_mode);

  std::optional<int> getSlew(AnalysisMode analysis_mode, TransType trans_type);
  std::optional<double> getSlewNs(AnalysisMode analysis_mode,
                                  TransType trans_type) {
    auto slew = getSlew(analysis_mode, trans_type);
    if (slew) {
      return FS_TO_NS(*slew);
    } else {
      return std::nullopt;
    }
  }
  std::optional<double> getWorstSlewNs(AnalysisMode analysis_mode) {
    auto rise_slew = getSlewNs(analysis_mode, TransType::kRise);
    if (rise_slew) {
      auto fall_slew = getSlewNs(analysis_mode, TransType::kFall);
      if (rise_slew && fall_slew) {
        if (analysis_mode == AnalysisMode::kMax) {
          return (*rise_slew > *fall_slew) ? rise_slew : fall_slew;
        } else {
          return (*rise_slew < *fall_slew) ? rise_slew : fall_slew;
        }
      }
    }

    return std::nullopt;
  }

  double getLoad(AnalysisMode analysis_mode, TransType trans_type);
  double getNetSlewImpulse(AnalysisMode analysis_mode, TransType trans_type);
  double getNetLoadDelay(AnalysisMode analysis_mode, TransType trans_type);
  double getNetLoad();
  double getResistance(AnalysisMode analysis_mode, TransType trans_type);
  int getNetworkLatency(AnalysisMode analysis_mode, TransType trans_type);

  StaSlewData* getSlewData(AnalysisMode analysis_mode, TransType trans_type,
                           StaData* src_slew_data);

  StaPathDelayData* getPathDelayData(AnalysisMode analysis_mode,
                                     TransType trans_type,
                                     StaData* src_delay_data);
  void getPathDepth(std::priority_queue<int, std::vector<int>,
                                        std::greater<int>>& depth_min_queue,
                    int depth = 0);

  unsigned GetWorstPathDepth(AnalysisMode analysis_mode);

  unsigned exec(StaFunc& func);

  void dump();

 private:
  DesignObject* _obj;                     //!< The mapped design object.
  unsigned _is_clock : 1 = 0;             //!< The vertex is clock pin.
  unsigned _is_clock_gate_clock : 1 = 0;  //!< The vertex is the clock pin of
                                          //!< the clock gate cell.
  unsigned _is_port : 1 = 0;              //!< The vertex is design port.
  unsigned _is_start : 1 = 0;             //!<  The vertex is start node.
  unsigned _is_end : 1 = 0;               //!< The vertex is end node.
  unsigned _is_clock_gate_end : 1 = 0;  //!< The vertex is the enable pin of the
                                        //!< clock gate cell.
  unsigned _is_const : 1 = 0;           //!< The vertex is const.
  unsigned _is_const_vdd : 1 = 0;       //!< The vertex is const one.
  unsigned _is_const_gnd : 1 = 0;       //!< The vertex is const zero.
  unsigned _color : 2 = 0;              //!< The vertex color.
  unsigned _is_slew_prop : 1 = 0;       //!< The vertex is slew propagated.
  unsigned _is_delay_prop : 1 = 0;      //!< The vertex is delay propagated.
  unsigned _is_bwd : 1 = 0;  //!< The vetex is req time backward propagated.
  unsigned _is_fwd : 1 = 0;  //!< The vertex is arrive time forward propagated.
  unsigned _is_crosstalk_prop : 1 = 0;  // The vertex is crosstalk propagated.

  unsigned _level : 10 = 0;            //!< The vertex level, start from 1;
  unsigned _is_sdc_clock_pin : 1 = 0;  //!< The create_clock or
                                       //!< create_generate_clock constrain pin.
  unsigned _is_ideal_clock_latency : 1 = 0;  //!< The ideal clock latency set.
  unsigned _is_bidirection : 1 = 0;          //!< The vertex is inout pin.
  unsigned _is_assistant : 1 = 0;  // for the inout node, split two node, input
                                   // main, output assistant.
  unsigned _is_foward_find : 1 =
      0;  //!< The vertex forward propagate to find end.
  unsigned _is_backward_find : 1 =
      0;  //!< The vetex backward propagate to find start.
  unsigned _is_fwd_reset : 1 = 0;  //!< The vertex is reset by fwd.
  unsigned _is_bwd_reset : 1 = 0;  //!< The vertex is reset by bwd.

  std::vector<StaArc*> _src_arcs;  //!< The timing arc sourced from the vertex.
  std::vector<StaArc*> _snk_arcs;  //!< The timing arc sinked to the vertex.
  StaDataBucket _slew_bucket;      //!< The slew data bucket.
  StaDataBucket _clock_bucket;     //!< The clock path data bucket.
  StaDataBucket _path_delay_bucket;  //!< The data path data bucket.
  std::mutex _fwd_mutex;             //!< The fwd mutex for mutiple thread.
  std::mutex _bwd_mutex;             //!< The bwd mutex for mutiple thread.

  std::array<std::optional<double>, TRANS_SPLIT> _max_cap;
  std::array<std::optional<double>, TRANS_SPLIT> _max_slew;
  std::optional<double> _max_fanout;

  StaPropagationTag _prop_tag;  //!< The propagation tag.

  BTreeSet<StaVertex*>
      _fanout_end_vertexes;  //<! The endpoint vertexes of the timing path.
  BTreeSet<StaVertex*>
      _fanin_start_vertexes;  //<! The start vertexes of the timing path.

  FORBIDDEN_COPY(StaVertex);
};

/**
 * @brief Traverse the src arc of the vertex, usage:
 * StaVertex* vertex;
 * FOREACH_SRC_ARC(vertex, src_arc)
 * {
 *    do_something_for_arc();
 * }
 */
#define FOREACH_SRC_ARC(vertex, src_arc) \
  for (auto* src_arc : vertex->get_src_arcs())

/**
 * @brief Traverse the snk arc of the vertex, usage:
 * StaVertex* vertex;
 * FOREACH_SNK_ARC(vertex, snk_arc)
 * {
 *    do_something_for_arc();
 * }
 */
#define FOREACH_SNK_ARC(vertex, snk_arc) \
  for (auto* snk_arc : vertex->get_snk_arcs())

/**
 * @brief Traverse the slew bucket data of the vertex, usage:
 * StaVertex* vertex;
 * StaData* slew_data;
 * FOREACH_SLEW_DATA(vertex, delay_data)
 * {
 *    do_something_for_slew_data();
 * }
 */
#define FOREACH_SLEW_DATA(vertex, slew_data)                \
  for (StaDataBucketIterator iter(vertex->getSlewBucket()); \
       iter.hasNext() ? slew_data = iter.next().get(), true : false;)

/**
 * @brief Traverse the clock bucket data of the vertex, usage:
 * StaVertex* vertex;
 * StaData* clock_data;
 * FOREACH_CLOCK_DATA(vertex, clock_data)
 * {
 *    do_something_for_clock_data();
 * }
 */
#define FOREACH_CLOCK_DATA(vertex, clock_data)               \
  for (StaDataBucketIterator iter(vertex->getClockBucket()); \
       iter.hasNext() ? clock_data = iter.next().get(), true : false;)

/**
 * @brief Traverse the data bucket data of the vertex, usage:
 * StaVertex* vertex;
 * StaData* delay_data;
 * FOREACH_DELAY_DATA(vertex, delay_data)
 * {
 *    do_something_for_delay_data();
 * }
 */
#define FOREACH_DELAY_DATA(vertex, delay_data)              \
  for (StaDataBucketIterator iter(vertex->getDataBucket()); \
       iter.hasNext() ? delay_data = iter.next().get(), true : false;)

}  // namespace ista