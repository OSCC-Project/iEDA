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
 * @file PwrVertex.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The power vertex of power graph, which may be pin or port.
 * @version 0.1
 * @date 2023-01-18
 */
#pragma once

#include <map>
#include <mutex>

#include "PwrData.hh"
#include "PwrFunc.hh"
#include "netlist/Pin.hh"
#include "sta/StaClock.hh"
#include "sta/StaVertex.hh"

namespace ipower {
class PwrArc;
class PwrSeqVertex;

/**
 * @brief Seq vertex sort by name.
 *
 */
struct PwrSeqVertexComp {
  bool operator()(const PwrSeqVertex* const& lhs,
                  const PwrSeqVertex* const& rhs) const;
};

/**
 * @brief The power vertex of power graph.
 *
 */
class PwrVertex {
 public:
  explicit PwrVertex(StaVertex* sta_vertex) : _sta_vertex(sta_vertex) {}
  ~PwrVertex() = default;

  using PwrSeqVertexSet = BTreeSet<PwrSeqVertex*, PwrSeqVertexComp>;

  auto* get_sta_vertex() { return _sta_vertex; }

  auto* getDesignObj() { return _sta_vertex->get_design_obj(); }
  ista::Instance* getOwnInstance() {
    if (auto* the_pin = dynamic_cast<ista::Pin*>(_sta_vertex->get_design_obj());
        the_pin) {
      return the_pin->get_own_instance();
    }

    return nullptr;
  }

  ista::Net* getConnectNet() {
    auto* design_obj = _sta_vertex->get_design_obj();
    return design_obj->get_net();
  }

  std::string getName() const { return _sta_vertex->getName(); }

  std::optional<double> getDriveVoltage();

  unsigned isPin() { return _sta_vertex->get_design_obj()->isPin(); }
  unsigned isMacroPin() {
    return isPin() && (_sta_vertex->getOwnCell()->isMacroCell());
  }
  unsigned isSeqPin() {
    return isPin() && (_sta_vertex->getOwnCell()->isSequentialCell());
  }
  unsigned isSeqClockPin() { return isPin() && (_sta_vertex->is_clock()); }

  LibPort* getLibertyPort() {
    return isPin() ? dynamic_cast<ista::Pin*>(getDesignObj())->get_cell_port()
                   : nullptr;
  }
  unsigned isSeqDataIn() {
    auto lib_port = getLibertyPort();
    return lib_port ? lib_port->isSeqDataIn() : false;
  }

  [[nodiscard]] unsigned is_seq_visited() const { return _is_seq_visited; }
  void set_is_seq_visited() { _is_seq_visited = 1; }

  [[nodiscard]] unsigned is_const() const { return _is_const; }
  void set_is_const() { _is_const = 1; }

  [[nodiscard]] unsigned is_const_propagated() const {
    return _is_const_propagated;
  }
  void set_is_const_propagated() { _is_const_propagated = 1; }

  [[nodiscard]] unsigned is_toggle_sp_propagated() const {
    return _is_toggle_sp_propagated;
  }
  void set_is_toggle_sp_propagated() { _is_toggle_sp_propagated = 1; }

  [[nodiscard]] unsigned is_const_vdd() const { return _is_const_vdd; }
  void set_is_const_vdd() { _is_const_vdd = 1; }

  [[nodiscard]] unsigned is_const_gnd() const { return _is_const_gnd; }
  void set_is_const_gnd() { _is_const_gnd = 1; }

  [[nodiscard]] unsigned is_input_port() const { return _is_input_port; }
  void set_is_input_port() { _is_input_port = 1; }

  [[nodiscard]] unsigned is_output_port() const { return _is_output_port; }
  void set_is_output_port() { _is_output_port = 1; }

  [[nodiscard]] unsigned is_clock_network() const { return _is_clock_network; }
  void set_is_clock_network() { _is_clock_network = 1; }

  void set_internal_power(double internal_power) {
    _internal_power = internal_power;
  }
  [[nodiscard]] double getInternalPower() const { return _internal_power.value_or(0.0); }
  [[nodiscard]] auto get_internal_power() const { return _internal_power; }

  void addSrcArc(PwrArc* src_arc) { _src_arcs.emplace_back(src_arc); }
  void addSnkArc(PwrArc* snk_arc) { _snk_arcs.emplace_back(snk_arc); }
  auto& get_src_arcs() { return _src_arcs; }
  auto& get_snk_arcs() { return _snk_arcs; }

  void addData(PwrToggleData* toggle_data,
               std::optional<PwrClock*> the_fastest_clock = std::nullopt);

  void addData(PwrSPData* sp_data) {
    if (sp_data) {
      _sp_bucket.addData(sp_data, 0);
    }
  }

  void addData(double toggle, double sp, PwrDataSource data_source,
               std::optional<PwrClock*> the_fastest_clock = std::nullopt);

  PwrDataBucket& getToggleBucket() { return _toggle_bucket; }
  PwrDataBucket& getSPBucket() { return _sp_bucket; }

  double getToggleData(std::optional<PwrDataSource> data_source);
  double getSPData(std::optional<PwrDataSource> data_source);

  void addFanoutSeqVertex(PwrSeqVertex* fanout_seq_vertex, unsigned level) {
    LOG_FATAL_IF(!fanout_seq_vertex) << "insert nullptr.";
    _fanout_seq_vertexes.insert(fanout_seq_vertex);
    if (_fanout_seq_vertex_to_level.contains(fanout_seq_vertex)) {
      auto current_level = _fanout_seq_vertex_to_level[fanout_seq_vertex];
      _fanout_seq_vertex_to_level[fanout_seq_vertex] =
          (level > current_level) ? level : current_level;
    } else {
      _fanout_seq_vertex_to_level[fanout_seq_vertex] = level;
    }
  }

  void addFanoutSeqVertex(const PwrSeqVertexSet& seq_vertex_set) {
    std::ranges::copy(
        seq_vertex_set,
        std::inserter(_fanout_seq_vertexes, _fanout_seq_vertexes.begin()));
  }
  auto& get_fanout_seq_vertexes() { return _fanout_seq_vertexes; }
  auto& get_fanout_seq_vertex_to_level() { return _fanout_seq_vertex_to_level; }

  std::optional<PwrSeqVertex*> getFanoutMinSeqLevel();
  void set_own_seq_vertex(PwrSeqVertex* own_seq_vertex) {
    _own_seq_vertex = own_seq_vertex;
  }
  auto* get_own_seq_vertex() { return _own_seq_vertex; }

  auto& get_mutex() { return _mutex; }

  std::unordered_set<StaClock*> getOwnClockDomain();
  std::optional<StaClock*> getOwnFastestClockDomain();

  bool isHaveConstSrcVertex();

  unsigned exec(PwrFunc& the_power_func) { return the_power_func(this); }

  void dumpVertexInfo();

 private:
  unsigned _is_seq_visited : 1 =
      0;  //!< The vertex is visited when building the seq graph.
  unsigned _is_const_propagated : 1 =
      0;  //!< The vertex is visited when propagate const.
  unsigned _is_toggle_sp_propagated : 1 =
      0;  //!< The vertex is visited when propagate toggle sp.
  unsigned _is_const : 1 = 0;          //!< The vertex is const.
  unsigned _is_const_vdd : 1 = 0;      //!< The vertex is const one.
  unsigned _is_const_gnd : 1 = 0;      //!< The vertex is const zero.
  unsigned _is_input_port : 1 = 0;     //!< The vertex is input port.
  unsigned _is_output_port : 1 = 0;    //!< The vertex is output port.
  unsigned _is_clock_network : 1 = 0;  //!< The vertex is clock nerwork.
  unsigned _reserved : 23 = 0;         //!< reserved.

  std::optional<double> _internal_power; //!< The pin internal power.

  StaVertex* _sta_vertex;          //!< The mapped sta vertex.
  std::vector<PwrArc*> _src_arcs;  //!< The power arc sourced from the vertex.
  std::vector<PwrArc*> _snk_arcs;  //!< The power arc sinked to the vertex.
  PwrDataBucket _toggle_bucket;    //! The toggle bucket.
  PwrDataBucket _sp_bucket;        //!< The sp bucket.

  PwrSeqVertexSet _fanout_seq_vertexes;  //!< The fan out seq vertexes if the
                                         //!< vertex is belong to combine logic,
                                         //!< sorted by object name.
  std::map<PwrSeqVertex*, unsigned>
      _fanout_seq_vertex_to_level;  //!< The fanout seq vertex level to the
                                    //!< vertex from end vertex, end vertex is
                                    //!< level zero.
  PwrSeqVertex* _own_seq_vertex =
      nullptr;  //!< The own seq vertex, may be the vertex is belong a
                //!< sequential instance.
  std::mutex _mutex;
};

/**
 * @brief Traverse the src arc of the power vertex, usage:
 * PwrVertex* pwr_vertex;
 * FOREACH_SRC_PWR_ARC(pwr_vertex, src_arc)
 * {
 *    do_something_for_arc();
 * }
 */
#define FOREACH_SRC_PWR_ARC(pwr_vertex, src_arc) \
  for (auto* src_arc : pwr_vertex->get_src_arcs())

/**
 * @brief Traverse the snk arc of the vertex, usage:
 * PwrVertex* pwr_vertex;
 * FOREACH_SNK_PWR_ARC(pwr_vertex, snk_arc)
 * {
 *    do_something_for_arc();
 * }
 */
#define FOREACH_SNK_PWR_ARC(pwr_vertex, snk_arc) \
  for (auto* snk_arc : pwr_vertex->get_snk_arcs())

}  // namespace ipower