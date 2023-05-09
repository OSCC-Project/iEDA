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
 * @file StaBuildExceptionTag.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief Build exception tag of multicycle path, false path, max/min delay.
 * @version 0.1
 * @date 2022-07-20
 */
#pragma once

#include "StaBuildPropTag.hh"
#include "sta/StaFunc.hh"
#include "sta/StaVertex.hh"

namespace ista {

/**
 * @brief Build the exception tag of sdc exception.
 *
 */
class StaBuildExceptionTag : public StaBuildPropTag {
 public:
  explicit StaBuildExceptionTag(StaPropagationTag::TagType tag_type)
      : StaBuildPropTag(tag_type) {}
  ~StaBuildExceptionTag() override = default;

  unsigned operator()(StaVertex* the_vertex) override;
  unsigned operator()(StaGraph* the_graph) override;

  void set_sdc_exception(SdcException* sdc_exception) {
    _sdc_exception = sdc_exception;
  }
  auto* get_sdc_exception() { return _sdc_exception; }

 private:
  SdcException* _sdc_exception;
};

}  // namespace ista