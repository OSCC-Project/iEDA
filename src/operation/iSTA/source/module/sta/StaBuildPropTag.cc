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
 * @file StaBuildTag.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The impelmention of propagation tag for specify path by -from -through
 * -to.
 * @version 0.1
 * @date 2022-04-21
 */
#include "StaBuildPropTag.hh"

#include <algorithm>
#include <iterator>

#include "Sta.hh"
#include "StaGraph.hh"
#include "sta/StaVertex.hh"

namespace ista {

/**
 * @brief acquire the intersection of subgraph.
 *
 * @param other
 * @return StaSubGraph
 */
StaSubGraph StaSubGraph::intersectSubGraph(StaSubGraph& other) {
  StaSubGraph intersect_graph;
  std::set<StaArc*> intersect_arcs;

  std::set_intersection(_prop_arcs.begin(), _prop_arcs.end(),
                        other._prop_arcs.begin(), other._prop_arcs.end(),
                        std::inserter(intersect_arcs, intersect_arcs.begin()));
  intersect_graph.set_prop_arcs(std::move(intersect_arcs));
  return intersect_graph;
}

/**
 * @brief mark the sub graph vertex prop tag.
 *
 */
void StaSubGraph::markSubGraphPropTag(StaPropagationTag::TagType tag_type) {
  for (auto* prop_arc : _prop_arcs) {
    prop_arc->get_src()->get_prop_tag().setTag(tag_type, true);
    prop_arc->get_snk()->get_prop_tag().setTag(tag_type, true);
  }
}

/**
 * @brief build the tag of propagation for the vertex.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaBuildPropTag::operator()(StaVertex* the_vertex) {
  // propagate path.
  if (isPropPath()) {
    if (the_vertex->is_start()) {
      if (!the_vertex->get_prop_tag().isTagSet(_tag_type)) {
        the_vertex->setWhite();
      }
      return 1;
    }

    FOREACH_SNK_ARC(the_vertex, snk_arc) {
      if (!snk_arc->isDelayArc()) {
        continue;
      }

      auto* src_vertex = snk_arc->get_src();

      if (src_vertex->is_const()) {
        continue;
      }

      if (src_vertex->isBlack()) {
        continue;
      }

      if (src_vertex->isGray() && the_vertex->isGray()) {
        continue;
      }

      // set src vertex accord the vertex.
      if (the_vertex->isBlack()) {
        src_vertex->setBlack();
      } else {
        src_vertex->setGray();
      }

      // The vertex is through point, src vertex set black.
      if (src_vertex->get_prop_tag().is_through_point() &&
          src_vertex->get_prop_tag().isTagSet(_tag_type)) {
        src_vertex->setBlack();
      }

      if (!src_vertex->exec(*this)) {
        return 0;
      }
    }

  } else {
    // collect path.
    FOREACH_SRC_ARC(the_vertex, src_arc) {
      if (!src_arc->isDelayArc()) {
        continue;
      }

      auto* snk_vertex = src_arc->get_snk();
      if (snk_vertex->is_const()) {
        continue;
      }

      if (snk_vertex->isWhite()) {
        continue;
      }

      if (the_vertex->isBlack() && !snk_vertex->isBlack()) {
        if (!the_vertex->get_prop_tag().is_through_point() ||
            !the_vertex->get_prop_tag().isTagSet(_tag_type)) {
          continue;
        }
      }

      auto& curr_sub_graph = getCurrSubGraph();
      curr_sub_graph.addPropArc(src_arc);

      if (snk_vertex->get_prop_tag().is_collected()) {
        continue;
      }

      if (!snk_vertex->exec(*this)) {
        return 0;
      }
    }

    the_vertex->get_prop_tag().set_is_collected(true);
  }

  return 1;
}

/**
 * @brief set from through to vertex tag.
 *
 * @param the_graph
 * @param prop_vec
 * @param tag_type
 */
void StaBuildPropTag::setFromThroughToTag(StaGraph* the_graph,
                                          std::vector<std::string>& prop_vec,
                                          PointType tag_type) {
  if (tag_type == PointType::kStart) {
    // first set all vertex not set.
    StaVertex* start_vertex;
    FOREACH_START_VERTEX(the_graph, start_vertex) {
      start_vertex->get_prop_tag().setTag(_tag_type, false);
    }
  }

  if (tag_type == PointType::kEnd) {
    // first set all vertex not set.
    StaVertex* end_vertex;
    FOREACH_END_VERTEX(the_graph, end_vertex) {
      end_vertex->get_prop_tag().setTag(_tag_type, false);
    }
  }

  // set prop vec node tag.
  for (auto& prop_node : prop_vec) {
    auto objs = the_graph->get_nl()->findObj(prop_node.c_str(), false, false);
    for (auto* obj : objs) {
      auto the_vertex = the_graph->findVertex(obj);
      if (the_vertex) {
        if (obj->isInout() && (tag_type == PointType::kEnd) &&
            !(*the_vertex)->is_end()) {
          (*the_vertex) = the_graph->getAssistant((*the_vertex));
        }
        (*the_vertex)->get_prop_tag().setTag(_tag_type, true);
        if (tag_type == PointType::kThrough) {
          (*the_vertex)->get_prop_tag().set_is_through_point(true);
        }
      }
    }
  }
}

/**
 * @brief set from to tag.
 *
 * @param the_graph
 * @param from_vec
 * @param to_vec
 */
void StaBuildPropTag::setFromToTag(StaGraph* the_graph,
                                   std::vector<std::string>& from_vec,
                                   std::vector<std::string>& to_vec) {
  if (!from_vec.empty()) {
    setFromThroughToTag(the_graph, from_vec, PointType::kStart);
  }

  if (!to_vec.empty()) {
    setFromThroughToTag(the_graph, to_vec, PointType::kEnd);
  }
}

/**
 * @brief reset vertex tag data.
 *
 * @param the_graph
 */
void StaBuildPropTag::resetVertexTagData(StaGraph* the_graph) {
  StaVertex* the_vertex;
  FOREACH_VERTEX(the_graph, the_vertex) {
    if (!the_vertex->is_start() && !the_vertex->is_end()) {
      the_vertex->get_prop_tag().setTag(_tag_type, false);
    }

    the_vertex->resetColor();
    the_vertex->get_prop_tag().set_is_searched(false);
    the_vertex->get_prop_tag().set_is_collected(false);
  }
}

/**
 * @brief for the multi through, we need to build the final intersection sub
 * graph.
 *
 * @return StaSubGraph
 */
StaSubGraph StaBuildPropTag::buildFinalSubGraph() {
  auto& sub_graphs = get_sub_graphs();
  auto& first_sub_graph = sub_graphs.front();
  StaSubGraph final_sub_graph = first_sub_graph;
  for (auto& sub_graph : sub_graphs) {
    if (&first_sub_graph == &sub_graph) {
      continue;
    }
    final_sub_graph = first_sub_graph.intersectSubGraph(sub_graph);
  }

  return final_sub_graph;
}

/**
 * @brief accord from through to, build tag graph.
 *
 * @param the_graph
 * @param froms
 * @param tos
 * @param throughs_list
 * @return unsigned
 */
unsigned StaBuildPropTag::buildTagGraph(
    StaGraph* the_graph, std::vector<std::string> froms,
    std::vector<std::string> tos,
    std::vector<std::vector<std::string>> throughs_list) {
  if (!throughs_list.empty()) {
    for (auto& prop_throughs : throughs_list) {
      // init the graph vertex tag.
      resetVertexTagData(the_graph);
      setFromThroughToTag(the_graph, prop_throughs, PointType::kThrough);
      setFromToTag(the_graph, froms, tos);

      // propagate the graph to color the path.
      StaVertex* end_vertex;
      FOREACH_END_VERTEX(the_graph, end_vertex) {
        if (end_vertex->get_prop_tag().isTagSet(_tag_type)) {
          setPropPath();
          end_vertex->setGray();
          if (!end_vertex->exec(*this)) {
            return 0;
          }
        }
      }

      LOG_INFO << "collect propagation path";
      // collect the path of color vertex.
      reservedSubGraph();
      StaVertex* start_vertex;
      FOREACH_START_VERTEX(the_graph, start_vertex) {
        if (start_vertex->get_prop_tag().isTagSet(_tag_type) &&
            start_vertex->isBlack()) {
          setCollectPath();
          if (!start_vertex->exec(*this)) {
            return 0;
          }
        }
      }
    }

    // build the final sub graph.
    auto final_sub_graph = buildFinalSubGraph();
    final_sub_graph.markSubGraphPropTag(_tag_type);

  } else {
    setFromToTag(the_graph, froms, tos);
  }

  return 1;
}

/**
 * @brief build tag of propagation for the graph.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaBuildPropTag::operator()(StaGraph* the_graph) {
  LOG_INFO << "build propagation tag start";
  auto* ista = getSta();
  auto& report_spec = ista->get_report_spec();
  if (!report_spec) {
    return 1;
  }

  auto& prop_froms = report_spec->get_prop_froms();
  auto& prop_tos = report_spec->get_prop_tos();
  auto& prop_throughs_list = report_spec->get_prop_throughs();

  buildTagGraph(the_graph, prop_froms, prop_tos, prop_throughs_list);

  LOG_INFO << "build propagation tag end";

  return 1;
}
}  // namespace ista
