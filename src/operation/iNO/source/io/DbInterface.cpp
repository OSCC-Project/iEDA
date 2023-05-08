#include "DbInterface.h"

namespace ino {
DbInterface *DbInterface::_db_interface = nullptr;

DbInterface *DbInterface::get_db_interface(NoConfig *config, IdbBuilder *idb,
                                           TimingEngine *timing) {
  static std::mutex mt;
  if (_db_interface == nullptr) {
    std::lock_guard<std::mutex> lock(mt);
    if (_db_interface == nullptr) {
      _db_interface = new DbInterface(config);
      _db_interface->_timing_engine = timing;
      _db_interface->_idb = idb;
      _db_interface->initData();
    }
  }
  return _db_interface;
}

void DbInterface::destroyDbInterface() {
  if (_db_interface != nullptr) {
    delete _db_interface;
    _db_interface = nullptr;
  }
}

void DbInterface::initData() {
  // log report
  string report_path = _config->get_report_file();
  _reporter = new Reporter(report_path);
  _reporter->reportTime(true);
}

} // namespace ino
