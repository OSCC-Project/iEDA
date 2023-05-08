#ifndef IDB_LOOKUP_TABLE_H
#define IDB_LOOKUP_TABLE_H

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace idb {
  template<class T1, class T2>
  class IdbLookup1DTable {
   public:
    IdbLookup1DTable() { }
    IdbLookup1DTable(const IdbLookup1DTable &table)
        : _rows(table._rows), _values(table._values), _row_name(table._row_name) { }
    IdbLookup1DTable(IdbLookup1DTable &&table)
        : _rows(std::move(table._rows)), _values(std::move(table._values)), _row_name(table._row_name) { }
    ~IdbLookup1DTable() { }

    IdbLookup1DTable &operator=(const IdbLookup1DTable &table) {
      _rows     = table._rows;
      _values   = table._values;
      _row_name = table._row_name;
      return *this;
    }
    IdbLookup1DTable &operator=(IdbLookup1DTable &&table) {
      _rows     = std::move(table._rows);
      _values   = std::move(table._values);
      _row_name = table._row_name;
      return *this;
    }

    // getter
    std::vector<T1> &get_rows() { return _rows; }
    const std::string &get_row_name() const { return _row_name; }
    // setter
    void set_row_name(std::string &name) { _row_name = name; }
    // others
    size_t getRowPosition(const T1 &row) const;
    T2 findValue(const T1 &row) const;
    T2 findMinValue() const { return _values.front(); }
    T2 findMaxValue() const { return _values.back(); }

   private:
    std::vector<T1> _rows;
    std::vector<T2> _values;
    std::string _row_name;
  };
  template<class T1, class T2>
  inline size_t IdbLookup1DTable<T1, T2>::getRowPosition(const T1 &row) const {
    size_t position;
    if (row >= _rows.front() && row <= _rows.back()) {
      auto pos = std::lower_bound(_rows.begin(), _rows.end(), row);
      if (pos != _rows.begin()) {
        --pos;
      }
      position = pos - _rows.begin();
    } else if (row < _rows.front()) {
      position = 0;
    } else {
      position = _rows.size() - 1;
    }
    return position;
  }

  template<class T1, class T2>
  inline T2 IdbLookup1DTable<T1, T2>::findValue(const T1 &row) const {
    size_t position = getRowPosition(row);
    T2 value        = _values[position];
    return value;
  }
  /*****************************************************************************************************************************/
  // 2D Table
  template<class T1, class T2, class T3>
  class Idb2DLookupTable {
   public:
    Idb2DLookupTable() { }
    Idb2DLookupTable(const std::string &row_name, const std::string &column_name, const std::vector<T1> &row_list,
                     const std::vector<T2> &column_list, const std::vector<std::vector<T3>> &value_table)
        : _row_name(row_name),
          _column_name(column_name),
          _row_list(row_list),
          _column_list(column_list),
          _value_table(value_table) { }
    Idb2DLookupTable(const Idb2DLookupTable &table)
        : _row_name(table._row_name),
          _column_name(table._column_name),
          _row_list(table._row_list),
          _column_list(table._column_list),
          _value_table(table._value_table) { }
    Idb2DLookupTable(Idb2DLookupTable &&table)
        : _row_name(table._row_name),
          _column_name(table._column_name),
          _row_list(std::move(table._row_list)),
          _column_list(std::move(table._column_list)),
          _value_table(std::move(table._value_table)) { }
    ~Idb2DLookupTable() { }
    // operator
    Idb2DLookupTable &operator=(const Idb2DLookupTable &table) {
      _row_name    = table._row_name;
      _column_name = table._column_name;
      _row_list    = table._row_list;
      _column_list = table._column_list;
      _value_table = table._value_table;
      return *this;
    }
    Idb2DLookupTable &operator=(Idb2DLookupTable &&table) {
      _row_name    = table._row_name;
      _column_name = table._column_name;
      _row_list    = std::move(table._row_list);
      _column_list = std::move(table._column_list);
      _value_table = std::move(table._value_table);
      return *this;
    }
    // getter
    const std::string &get_row_name() const { return _row_name; }
    const std::string &get_column_name() const { return _column_name; }
    std::vector<T1> &get_row_list() const { return _row_list; }
    std::vector<T2> &get_column_list() const { return _column_list; }

    // setter
    void set_row_name(std::string &row_name) { _row_name = row_name; }
    void set_column_name(std::string &column_name) { _column_name = column_name; }
    // others
    size_t getRowPosition(const T1 &rowId) const;
    size_t getColumnPosition(const T2 &columnId) const;
    T3 findValue(const T1 &rowId, const T2 &columnId) const;
    T3 findMin() const { return _value_table.front().front(); }
    T3 findMax() const { return _value_table.back().back(); }
    // debug
    void printTable();

   private:
    std::string _row_name;
    std::string _column_name;
    std::vector<T1> _row_list;
    std::vector<T2> _column_list;
    std::vector<std::vector<T3>> _value_table;
  };
  template<class T1, class T2, class T3>
  inline size_t Idb2DLookupTable<T1, T2, T3>::getRowPosition(const T1 &rowId) const {
    auto pos        = --(std::lower_bound(_row_list.begin(), _row_list.end(), rowId));
    size_t position = std::max(0, (int)std::distance(_row_list.begin(), pos));
    return position;
  }
  template<class T1, class T2, class T3>
  inline size_t Idb2DLookupTable<T1, T2, T3>::getColumnPosition(const T2 &columnId) const {
    auto pos        = --(std::lower_bound(_column_list.begin(), _column_list.end(), columnId));
    size_t position = std::max(0, (int)std::distance(_column_list.begin(), pos));
    return position;
  }
  template<class T1, class T2, class T3>
  inline T3 Idb2DLookupTable<T1, T2, T3>::findValue(const T1 &rowId, const T2 &columnId) const {
    size_t rowPosition    = getRowPosition(rowId);
    size_t columnPosition = getColumnPosition(columnId);
    T3 value              = _value_table[rowPosition][columnPosition];
    return value;
  }
  template<class T1, class T2, class T3>
  inline void Idb2DLookupTable<T1, T2, T3>::printTable() {
    std::cout << "RowName :" << _row_name << std::endl;
    for (auto &r : _row_list) {
      std::cout << r << " ";
    }
    std::cout << std::endl;
    std::cout << "ColName :" << _column_name << std::endl;
    for (auto &c : _column_list) {
      std::cout << c << " ";
    }
    std::cout << std::endl;
    std::cout << "values :" << std::endl;
    for (auto &values : _value_table) {
      for (auto &value : values) {
        std::cout << value << " ";
      }
      std::cout << std::endl;
    }
  }

}  // namespace idb

#endif
