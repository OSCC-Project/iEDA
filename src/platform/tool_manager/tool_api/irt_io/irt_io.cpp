#include "irt_io.h"

namespace iplf {

  // public

  RTIO& RTIO::getInst() {
    if (_rt_io_instance == nullptr) {
      _rt_io_instance = new RTIO();
    }
    return *_rt_io_instance;
  }

  void RTIO::delInst() {
    if (_rt_io_instance != nullptr) {
      delete _rt_io_instance;
      _rt_io_instance = nullptr;
    }
  }

  // private

  RTIO* RTIO::_rt_io_instance = nullptr;

}  // namespace iplf
