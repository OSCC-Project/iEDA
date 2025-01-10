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
 * @file StaDataPropagation.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of data propagation, by which propgate the arrvie time.
 * from the start vertex.
 * @version 0.1
 * @date 2021-03-10
 */
#pragma once

#include "StaFunc.hh"

namespace ista {

/**
 * @brief The forward propagation for calculating the arrive time.
 *
 */
class StaFwdPropagation : public StaFunc {
 public:
  unsigned operator()(StaVertex* the_vertex) override;
  unsigned operator()(StaArc* the_arc) override;
  unsigned operator()(StaGraph* the_graph) override;

 protected:
  unsigned createClockVertexStartData(StaVertex* the_vertex);
  unsigned createPortVertexStartData(StaVertex* the_vertex);
  unsigned createStartData(StaVertex* the_vertex);
};

/**
 * @brief The backward propagation for calculating the require time.
 *
 */
class StaBwdPropagation : public StaFunc {
 public:
  unsigned operator()(StaVertex* the_vertex) override;
  unsigned operator()(StaArc* the_arc) override;
  unsigned operator()(StaGraph* the_graph) override;

 private:
  unsigned createEndData(StaVertex* the_vertex);
};
/**
 * @brief The data propagation backward from the
 * end vertex to start vertex.
 *
 */
class StaDataPropagation : public StaFunc {
 public:
  enum class PropType { kFwdProp, kBwdProp, kIncrFwdProp };
  explicit StaDataPropagation(PropType prop_type) : _prop_type(prop_type) {}
  ~StaDataPropagation() override = default;
  unsigned operator()(StaGraph* the_graph) override;

 private:
  PropType _prop_type;
};

}  // namespace ista
