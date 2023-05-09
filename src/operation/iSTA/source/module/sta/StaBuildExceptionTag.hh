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