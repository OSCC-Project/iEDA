/**
 * @file DesignObject.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The base class for Pin, Port, Net, Instance, Cell.
 * @version 0.1
 * @date 2021-02-03
 */

#pragma once

#include <string>
#include <utility>

#include "Type.hh"
#include "log/Log.hh"

namespace ista {

class Instance;
class Net;

/**
 * @brief The base class for design such as Pin, Port, Net, Instance, Cell.
 *
 */
class DesignObject {
 public:
  explicit DesignObject(const char* name);
  virtual ~DesignObject() = default;

  DesignObject(DesignObject&& other) noexcept;
  DesignObject& operator=(DesignObject&& rhs) noexcept;

  virtual unsigned isNetlist() { return 0; }

  virtual unsigned isPin() { return 0; }
  virtual unsigned isNet() { return 0; }

  virtual unsigned isInstance() { return 0; }

  virtual unsigned isPort() { return 0; }

  virtual unsigned isInput() {
    LOG_FATAL << "The object is not defined.";
    return 0;
  }

  virtual unsigned isOutput() {
    LOG_FATAL << "The object is not defined.";
    return 0;
  }

  virtual unsigned isInout() {
    LOG_FATAL << "The object is not defined.";
    return 0;
  }

  const char* get_name() const { return _name.c_str(); }
  void set_name(const char* name) { _name = name; }

  virtual std::string getFullName() {
    LOG_FATAL << "The object do not have fullname.";
    return nullptr;
  }

  virtual double cap() {
    LOG_FATAL << "The object do not has cap.";
    return 0.0;
  }
  virtual double cap(AnalysisMode mode, TransType trans_type) {
    LOG_FATAL << "The object do not has cap.";
    return 0.0;
  }

  virtual unsigned isConst() {
    LOG_FATAL << "The object can not be judged const.";
    return 0;
  }

  virtual Net* get_net() {
    LOG_FATAL << "The func is not defined.";
    return nullptr;
  }

  virtual void set_net(Net* /*net*/) {
    LOG_FATAL << "The func is not defined.";
  }

  virtual unsigned isPortBus() { return 0; }
  virtual unsigned isPinBus() { return 0; }

  virtual Instance* get_own_instance() {
    LOG_FATAL << "The func is not defined.";
    return nullptr;
  }

 private:
  std::string _name;
};

}  // namespace ista
