#pragma once

namespace irt {

class RAGCellNode
{
 public:
  RAGCellNode() = default;
  RAGCellNode(const irt_int gcell_idx, const irt_int result_idx)
  {
    _gcell_idx = gcell_idx;
    _result_idx = result_idx;
  }
  ~RAGCellNode() = default;
  // getter
  irt_int get_gcell_idx() const { return _gcell_idx; }
  irt_int get_result_idx() const { return _result_idx; }

  // setter

  // function

 private:
  irt_int _gcell_idx = -1;
  irt_int _result_idx = -1;
};

}  // namespace irt
