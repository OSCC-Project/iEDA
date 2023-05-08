#ifndef SRC_PLATFORM_EVALUATOR_SOURCE_WRAPPER_DBWRAPPER_HPP_
#define SRC_PLATFORM_EVALUATOR_SOURCE_WRAPPER_DBWRAPPER_HPP_

#include "Config.hpp"
#include "EvalDB.hpp"
#include "EvalLog.hpp"

namespace eval {

using namespace idb;

class DBWrapper
{
 public:
  DBWrapper() = delete;
  explicit DBWrapper(Config* config);
  DBWrapper(const DBWrapper&) = delete;
  DBWrapper(DBWrapper&&) = delete;
  ~DBWrapper();

  DBWrapper& operator=(const DBWrapper&) = delete;
  DBWrapper& operator=(DBWrapper&&) = delete;

  const Layout* get_layout() const { return _eval_db->_layout; }
  Design* get_design() const { return _eval_db->_design; }
  EvalDB* get_eval_db() const { return _eval_db; }

 private:
  DBConfig _db_config;
  CongConfig _cong_config;
  DRCConfig _drc_config;
  GDSWrapperConfig _gds_wrapper_config;
  PowerConfig _power_config;
  TimingConfig _timing_config;
  WLConfig _wl_config;

  EvalDB* _eval_db;

  void initIDB();
  void wrapIDBData();
  void wrapLayout(IdbLayout* idb_layout);
  void wrapDesign(IdbDesign* idb_design);
  // wrap different netlists
  void wrapWLNetlists(IdbDesign* idb_design);
  void wrapCongNetlists(IdbDesign* idb_design);
  void wrapGDSNetlist(IdbBuilder* idb_builder);

  // wrap different pins
  WLPin* wrapWLPin(IdbPin* idb_pin);
  CongPin* wrapCongPin(IdbPin* idb_pin);
  GDSPin* wrapGDSPin(IdbPin* idb_pin);
  // wrap instances
  void wrapInstances(IdbDesign* idb_design);

  bool isCoreOverlap(IdbInstance* idb_inst);
  std::string fixSlash(std::string raw_str);
};

}  // namespace eval

#endif  // SRC_PLATFORM_EVALUATOR_SOURCE_WRAPPER_DBWRAPPER_HPP_
