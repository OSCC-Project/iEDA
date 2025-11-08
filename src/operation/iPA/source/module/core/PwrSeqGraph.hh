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
 * @file PwrSeqGraph.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief the sequential graph for labeling propagation levels.
 * @version 0.1
 * @date 2023-02-27
 */
#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <ranges>
#include <vector>
#include <shared_mutex>

#include "BTreeSet.hh"
#include "PwrVertex.hh"
#include "include/PwrConfig.hh"
#include "netlist/Instance.hh"
#include "netlist/Pin.hh"

namespace ipower {

class PwrSeqArc;
class PwrSeqVertex;

/**
 * @brief Seq arc sort compare.
 *
 */
struct PwrSeqArcComp {
  bool operator()(const PwrSeqArc* const& lhs,
                  const PwrSeqArc* const& rhs) const;
};

/**
 * @brief The sequential vertex including levelize message.
 *
 */
class PwrSeqVertex {
 public:
  PwrSeqVertex(BTreeSet<PwrVertex*>&& data_in_vertexes,
               BTreeSet<PwrVertex*>&& data_out_vertexes)
      : _seq_in_vertexes(std::move(data_in_vertexes)),
        _seq_out_vertexes(std::move(data_out_vertexes)) {
    _obj_name = get_own_seq_inst()->get_name();

    // Set the own seq vertex for all data in vertexes and data out vertexes.
    auto set_own_seq_vertex = [this](PwrVertex* the_vertex) {
      the_vertex->set_own_seq_vertex(this);
    };
    std::ranges::for_each(data_in_vertexes, set_own_seq_vertex);
    std::ranges::for_each(data_out_vertexes, set_own_seq_vertex);
  }
  // Build special seq vertex for port.
  PwrSeqVertex(SeqPortType port_type, PwrVertex* port_vertex) {
    _obj_name = port_vertex->getDesignObj()->get_name();

    if (port_type == SeqPortType::kInput) {
      set_is_input_port();
      addOutVertex(port_vertex);
    } else {
      set_is_output_port();
      addInVertex(port_vertex);
    }
  }

  ~PwrSeqVertex() = default;

  using PwrSeqArcSet = std::set<PwrSeqArc*, PwrSeqArcComp>;

  auto& get_obj_name() const { return _obj_name; }

  [[nodiscard]] unsigned isInputPort() const { return _is_input_port; }
  void set_is_input_port() { _is_input_port = 1; }

  [[nodiscard]] unsigned isOutputPort() const { return _is_output_port; }
  void set_is_output_port() { _is_output_port = 1; }

  [[nodiscard]] unsigned isConst() const { return _is_const; }
  void set_is_const() { _is_const = 1; }

  [[nodiscard]] unsigned is_const_vdd() const { return _is_const_vdd; }
  void set_is_const_vdd() { _is_const_vdd = 1; }

  [[nodiscard]] unsigned is_const_gnd() const { return _is_const_gnd; }
  void set_is_const_gnd() { _is_const_gnd = 1; }

  [[nodiscard]] unsigned isLevelSet() const { return _is_level_set; }
  void set_is_level_set() { _is_level_set = 1; }

  [[nodiscard]] unsigned get_level() const { return _level; }
  void set_level(unsigned level) { _level = level; }

  void setGrey() { _tricolor_mark = TricolorMark::kGrey; }
  void setBlack() { _tricolor_mark = TricolorMark::kBlack; }
  TricolorMark get_tricolor_mark() { return _tricolor_mark; }

  void addInVertex(PwrVertex* vertex) {
    vertex->set_own_seq_vertex(this);
    _seq_in_vertexes.insert(vertex);
  }
  void addOutVertex(PwrVertex* vertex) {
    vertex->set_own_seq_vertex(this);
    _seq_out_vertexes.insert(vertex);
  }

  auto& get_seq_in_vertexes() { return _seq_in_vertexes; }
  auto& get_seq_out_vertexes() { return _seq_out_vertexes; }
  unsigned getSeqOutMaxLevel() {
    unsigned max_level = 0;
    for (auto* seq_out_pwr_vertex : _seq_out_vertexes) {
      unsigned sta_level = seq_out_pwr_vertex->get_sta_vertex()->get_level();
      if (sta_level > max_level) {
        max_level = sta_level;
      }
    }
    return max_level;
  }
  auto getOneConnectNet() {
    // first seq in net
    for (auto* seq_in_vertex : _seq_in_vertexes) {
      if (auto* net = seq_in_vertex->getConnectNet(); net) {
        return net;
      }
    }
     // second seq out net
    for (auto* seq_out_vertex : _seq_out_vertexes) {
      if (auto* net = seq_out_vertex->getConnectNet(); net) {
        return net;
      }
    }

    LOG_FATAL << "No connect net found!";
    return static_cast<ista::Net*>(nullptr);
  }
  auto getDataInVertexes() {
    BTreeSet<PwrVertex*> data_in_vertexes;
    for (auto* seq_in_vertex :
         _seq_in_vertexes | std::ranges::views::filter([](auto* seq_in_vertex) {
           return seq_in_vertex->isSeqDataIn();
         })) {
      data_in_vertexes.insert(seq_in_vertex);
    }
    return data_in_vertexes;
  }
  std::pair<std::optional<double>, unsigned>
  getDataInVertexWorstSlackAndDepth();

  void addSrcArc(PwrSeqArc* src_arc);
  void addSnkArc(PwrSeqArc* snk_arc);
  auto& get_src_arcs() { return _src_arcs; }
  auto& get_snk_arcs() { return _snk_arcs; }
  ista::Instance* get_own_seq_inst() {
    return !_seq_in_vertexes.empty()
               ? (*_seq_in_vertexes.begin())->getOwnInstance()
               : nullptr;
  }
  bool isMacro() {
    if (auto* own_seq_inst = get_own_seq_inst(); own_seq_inst) {
      return own_seq_inst->get_inst_cell()->isMacroCell();
    }
    return false;
  }
  unsigned exec(PwrFunc& the_power_func) { return the_power_func(this); }

 private:
  unsigned _is_input_port : 1 = 0;   //!< The seq vertex is input port.
  unsigned _is_output_port : 1 = 0;  //!< The seq vertex is output port.
  unsigned _is_const : 1 = 0;        //!< The connected input is VDD or GND.
  unsigned _is_const_vdd : 1 = 0;    //!< The vertex is const one.
  unsigned _is_const_gnd : 1 = 0;    //!< The vertex is const zero.
  unsigned _is_level_set : 1 = 0;    //!< The level is set.
  unsigned _level : 18 = 0;  //!< The level to which the current vertex belongs.
  unsigned _reserved : 8 = 0;  //!< reserved.

  std::string_view
      _obj_name;  //!< The seq vertex own instance name or port name.
  BTreeSet<PwrVertex*> _seq_in_vertexes;   //!< The datain vertexes.
  BTreeSet<PwrVertex*> _seq_out_vertexes;  //!< The dataout vertexes.
  PwrSeqArcSet _src_arcs;  //!< The sequential arc sourced from the vertex.
  PwrSeqArcSet _snk_arcs;  //!< The sequential arc sinked to the vertex.
  TricolorMark _tricolor_mark =
      TricolorMark::kWhite;  //!< The tricolor mark for check pipeline loop.

  FORBIDDEN_COPY(PwrSeqVertex);
};

/**
 * @brief Represent the connection relationship between sequential vertexes.
 *
 */
class PwrSeqArc {
 public:
  PwrSeqArc(PwrSeqVertex* src, PwrSeqVertex* snk) : _src(src), _snk(snk) {}
  virtual ~PwrSeqArc() = default;

  [[nodiscard]] unsigned isPipelineLoop() const { return _is_pipeline_loop; }
  void set_is_pipeline_loop() { _is_pipeline_loop = 1; }

  [[nodiscard]] unsigned get_combine_depth() const { return _combine_depth; }
  void set_combine_depth(unsigned combine_depth) { _combine_depth = combine_depth; }

  [[nodiscard]] auto* get_src() const { return _src; }
  [[nodiscard]] auto* get_snk() const { return _snk; }

  bool isSelfLoop() { return _src == _snk; }

 private:
  unsigned _is_pipeline_loop : 1 =
      0;  //!<  The arc is belong to a pipe line loop.
  unsigned _combine_depth : 10 = 0;  //!< The seq arc combine depth.
  unsigned _reserved : 21 = 0;       //!< reserved.

  PwrSeqVertex* _src;  //!< The arc src vertex.
  PwrSeqVertex* _snk;  //!< The arc snk vertex.
  FORBIDDEN_COPY(PwrSeqArc);
};

/**
 * @brief The macro of foreach seq src arc, usage:
 * PwrSeqVertex* vertex;
 * PwrSeqArc* arc;
 * FOREACH_SRC_SEQ_ARC(vertex, arc)
 * {
 *    do_something_for_arc();
 * }
 *
 */
#define FOREACH_SRC_SEQ_ARC(vertex, arc)                  \
  if (auto& arcs = vertex->get_src_arcs(); !arcs.empty()) \
    for (auto p = arcs.begin(); p != arcs.end() ? arc = *p, true : false; ++p)

/**
 * @brief The macro of foreach seq snk arc, usage:
 * PwrSeqVertex* vertex;
 * PwrSeqArc* arc;
 * FOREACH_SNK_SEQ_ARC(vertex, arc)
 * {
 *    do_something_for_arc();
 * }
 *
 */
#define FOREACH_SNK_SEQ_ARC(vertex, arc)                  \
  if (auto& arcs = vertex->get_snk_arcs(); !arcs.empty()) \
    for (auto p = arcs.begin(); p != arcs.end() ? arc = *p, true : false; ++p)

/**
 * @brief The sequential vertexes levelization database.
 *
 */
class PwrSeqGraph {
 public:
  PwrSeqGraph() = default;
  ~PwrSeqGraph() = default;

  PwrSeqGraph(PwrSeqGraph&& other) noexcept = default;
  PwrSeqGraph& operator=(PwrSeqGraph&& other) noexcept = default;

  void addPwrSeqVertex(PwrSeqVertex* vertex);
  auto& get_vertexes() { return _vertexes; }

  void addPwrSeqArc(PwrSeqArc* arc);
  auto& get_arcs() { return _arcs; }

  PwrSeqArc* findSeqArc(PwrSeqVertex* src_vertex, PwrSeqVertex* snk_vertex);

  void addInputPortVerex(PwrSeqVertex* vertex) {
    _input_port_vertexes.emplace_back(vertex);
  }
  auto& get_input_port_vertexes() { return _input_port_vertexes; }

  void addOutputPortVerex(PwrSeqVertex* vertex) {
    _output_port_vertexes.emplace_back(vertex);
  }
  auto& get_output_port_vertexes() { return _output_port_vertexes; }

  unsigned getSeqVertexNum() {
    return _vertexes.size() - _input_port_vertexes.size() -
           _output_port_vertexes.size();
  }
  unsigned getMacroSeqVertexNum() {
    unsigned num = 0;
    for (auto& vertex : _vertexes) {
      if (vertex->isMacro()) {
        ++num;
      }
    }
    return num;
  }
  unsigned getSeqArcNum() { return _arcs.size(); }
  unsigned getInputPortNum() { return _input_port_vertexes.size(); }
  unsigned getOutputPortNum() { return _output_port_vertexes.size(); }

  std::pair<std::size_t, std::size_t> getSeqVertexMaxFanoutAndMaxFain();
  // add seq vertex of the current level.
  void addLevelSeqVertex(unsigned level, PwrSeqVertex* vertex) {
    _level_to_seq_vertex[level].emplace_back(vertex);
  }
  auto& get_level_to_seq_vertex() { return _level_to_seq_vertex; }

  void printSeqLevelInfo(std::ostream& out);

  // get the Maximum depth at levelizing.
  [[nodiscard]] unsigned get_level_depth() const {
    return _level_to_seq_vertex.size();
  }

  void insertInstToVertex(Instance* seq_inst, PwrSeqVertex* seq_vertex);
  PwrSeqVertex* getSeqVertex(DesignObject* seq_obj) {
    return _obj_to_vertex.contains(seq_obj) ? _obj_to_vertex[seq_obj] : nullptr;
  }

  void insertPortToVertex(PwrVertex* pwr_port, PwrSeqVertex* seq_vertex) {
    _port_to_vertex[pwr_port] = seq_vertex;
    _obj_to_vertex[pwr_port->get_sta_vertex()->get_design_obj()] = seq_vertex;
  }
  PwrSeqVertex* getPortSeqVertex(PwrVertex* pwr_port) {
    return _port_to_vertex.contains(pwr_port) ? _port_to_vertex[pwr_port]
                                              : nullptr;
  }

  unsigned exec(std::function<unsigned(PwrSeqGraph*)> the_power_func) {
    return the_power_func(this);
  }

 private:
  std::map<unsigned, std::vector<PwrSeqVertex*>>
      _level_to_seq_vertex;  //!<  sequential vertexes contained in each
                             //!<  level. k:level, v:seq vertex
  std::vector<std::unique_ptr<PwrSeqVertex>>
      _vertexes;                                  //!< All sequential vertexes.
  std::vector<std::unique_ptr<PwrSeqArc>> _arcs;  //!< All sequential arcs.
  std::vector<PwrSeqVertex*> _input_port_vertexes;   //!< Input port vertexes.
  std::vector<PwrSeqVertex*> _output_port_vertexes;  // 1< Output port vertexes.
  std::map<DesignObject*, PwrSeqVertex*>
      _obj_to_vertex;  //!< The seq belong inst to graph vertex.
  std::map<PwrVertex*, PwrSeqVertex*>
      _port_to_vertex;  //!< The seq belong power port to graph vertex.

};

/**
 * @brief The macro of foreach seq vertex, usage:
 * PwrSeqGraph* graph;
 * PwrSeqVertex* vertex;
 * FOREACH_SEQ_VERTEX(graph, vertex)
 * {
 *    do_something_for_vertex();
 * }
 *
 */
#define FOREACH_SEQ_VERTEX(graph, vertex)                        \
  if (auto& vertexes = (graph)->get_vertexes(); !vertexes.empty()) \
    for (auto p = vertexes.begin();                              \
         p != vertexes.end() ? vertex = p->get(), true : false; ++p)

/**
 * @brief The macro of foreach seq arc, usage:
 * PwrSeqGraph* graph;
 * PwrSeqArc* arc;
 * FOREACH_SEQ_ARC(graph, arc)
 * {
 *    do_something_for_arc();
 * }
 *
 */
#define FOREACH_SEQ_ARC(graph, arc)                  \
  if (auto& arcs = (graph)->get_arcs(); !arcs.empty()) \
    for (auto p = arcs.begin();                      \
         p != arcs.end() ? arc = p->get(), true : false; ++p)
}  // namespace ipower
