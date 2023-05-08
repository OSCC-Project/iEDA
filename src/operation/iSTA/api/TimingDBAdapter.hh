
/**
 * @file TimingDBAdapter.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief TimingDBAdapter is a adapter convert the idb to timing netlist.
 * @version 0.1
 * @date 2021-09-04
 */
#pragma once
#include "DisallowCopyAssign.hh"
#include "HashMap.hh"
#include "sta/Sta.hh"

namespace ista {

/**
 * @brief The timing db adapter.
 *
 */
class TimingDBAdapter {
 public:
  explicit TimingDBAdapter(Sta* ista);
  virtual ~TimingDBAdapter() = default;

  virtual bool isPlaced(DesignObject* pin_or_port) {
    LOG_FATAL << "The function is not implemented.";
    return true;
  }
  virtual double dbuToMeters(int distance) const {
    LOG_FATAL << "The function is not implemented.";
    return 0.0;
  }

  virtual void location(DesignObject* pin_or_port,
                        // Return values.
                        double& x, double& y, bool& exists) {
    LOG_FATAL << "The function is not implemented.";
  }

  virtual unsigned convertDBToTimingNetlist() {
    LOG_FATAL << "The function is not implemented.";
    return 1;
  }
  Netlist* getNetlist() { return _ista->get_netlist(); }

 protected:
  Sta* _ista = nullptr;

 private:
  DISALLOW_COPY_AND_ASSIGN(TimingDBAdapter);
};

}  // namespace ista
