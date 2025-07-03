#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "log/Log.hh"
#include "ops/annotate_toggle_sp/AnnotateData.hh"
#include "vcd/VcdParserRustC.hh"

namespace ipower {

/**
 * @brief The vcd wrapper class of the rust vcd parser.
 *
 */
class RustVcdParserWrapper {
 public:
  unsigned readVcdFile(const char* vcd_file_path);
  unsigned buildAnnotateDB(const char* top_instance_name);
  unsigned calcScopeToggleAndSp(const char* top_instance_name);

  // std::vector<RustSignalTC> countTC();
  // std::vector<RustSignalDuration> countDuration();
  void printAnnotateDB(std::ostream& out) { _annotate_db.printAnnotateDB(out); }
  auto* get_annotate_db() { return &_annotate_db; }

 private:
  RustVCDFile* _vcd_file;
  void* _vcd_file_ptr;
  RustVCDScope* _top_instance_scope;

  // std::vector<RustSignalTC> _signal_tc_vec;

  std::optional<int64_t> _begin_time;  //!< simulation begin time.
  std::optional<int64_t> _end_time;    //!< simulation end time.
  AnnotateDB _annotate_db;  //!< The annotate database for store waveform data.
};

}  // namespace ipower