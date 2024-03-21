#include "MPAPI.hh"

#include "data_manager/parser/IDBParserEngine.hh"

// std::vector<std::pair<int64_t, int64_t>> imp::SAPlaceSeqPairInt64(
//     int max_iters, int num_actions, double cool_rate, const std::vector<int64_t>* pin_x_off, const std::vector<int64_t>* pin_y_off,
//     const std::vector<int64_t>* initial_lx, const std::vector<int64_t>* initial_ly, const std::vector<int64_t>* dx,
//     const std::vector<int64_t>* dy, const std::vector<int64_t>* halo_x, const std::vector<int64_t>* halo_y,
//     const std::vector<size_t>* pin2vertex, const std::vector<size_t>* net_span, int64_t region_lx, int64_t region_ly, int64_t region_dx,
//     int64_t region_dy, size_t num_moveable, bool pack_left, bool pack_bottom)
// {
//   return std::vector<std::pair<int64_t, int64_t>>();
// }

namespace imp {

MPAPI& MPAPI::getInst()
{
  if (!_s_imp_api_instance) {
    _s_imp_api_instance = new MPAPI();
  }

  return *_s_imp_api_instance;
}

void MPAPI::destoryInst()
{
  if (_s_imp_api_instance) {
    delete _s_imp_api_instance;
    _s_imp_api_instance = nullptr;
  }
}

MPAPI* MPAPI::_s_imp_api_instance = nullptr;

void MPAPI::initAPI(std::string mp_json_path, idb::IdbBuilder* idb_builder)
{
  _mp = std::make_shared<imp::MP>(new imp::IDBParser(idb_builder));
}
}  // namespace imp