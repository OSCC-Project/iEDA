#pragma once

namespace icts {

class DesignObject {
 public:
  DesignObject() : _is_newly(true) {}
  DesignObject(const DesignObject &) = default;
  ~DesignObject() = default;

  void set_is_newly(bool newly) { _is_newly = newly; }

  bool is_newly() const { return _is_newly; }

 protected:
  bool _is_newly;
};

}  // namespace icts