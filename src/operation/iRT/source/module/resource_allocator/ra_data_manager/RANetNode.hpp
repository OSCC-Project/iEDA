#pragma once

namespace irt {

class RANetNode
{
 public:
  RANetNode() = default;
  RANetNode(const irt_int ra_net_idx, const irt_int result_idx)
  {
    _ra_net_idx = ra_net_idx;
    _result_idx = result_idx;
  }
  ~RANetNode() = default;
  // getter
  irt_int get_ra_net_idx() const { return _ra_net_idx; }
  irt_int get_result_idx() const { return _result_idx; }
  // setter
  // function
 private:
  irt_int _ra_net_idx = -1;
  irt_int _result_idx = -1;
};

}  // namespace irt
