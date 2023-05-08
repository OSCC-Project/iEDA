/**
 * @file ReportTable.cc
 * @author simin tao (taosm@pcl.ac.cn).
 * @brief The wrapper of report table, https://github.com/seleznevae/libfort.git
 * @version 0.1
 * @date 2022-08-03
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "ReportTable.hh"

namespace ieda {
/**
 * @brief Write one row in table.
 *
 * @tparam T The type of str, should be char* or std::string.
 * @tparam Ts
 * @param str
 * @param strings
 * @return unsigned 1 if success, 0 else fail.
 */
template <typename T, typename... Ts>
unsigned ReportTable::writeRow(const T& str, const Ts&... strings)
{
  return write_ln(str, strings...) ? 1 : 0;
}

/**
 * @brief Write one row in table from the iterator of container.
 *
 * @tparam InputIt
 * @param first
 * @param last
 * @return unsigned
 */
template <typename InputIt>
unsigned ReportTable::writeRowFromIterator(InputIt first, InputIt last)
{
  return range_write_ln(first, last) ? 1 : 0;
}

void ReportTable::initHeader()
{
  if (_header_list.size() <= 0) {
    return;
  }
  /// header info
  (*this) << TABLE_HEAD;

  for (int i = 0; i < static_cast<int>(_header_list.size()); i++) {
    (*this)[0][i] = _header_list[i].c_str();
  }

  (*this) << TABLE_ENDLINE;
}

}  // namespace ieda