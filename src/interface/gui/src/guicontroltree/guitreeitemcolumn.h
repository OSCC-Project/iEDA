// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file guitreeitemcolumn.h
 * @author Wang Jun (wen8365@gmail.com)
 * @brief Custom column in tree item
 * @version 0.1
 * @date 2021-07-02
 * 
 *
 * 
 */

#ifndef GUICONTROLTREECOLUMN_H
#define GUICONTROLTREECOLUMN_H

#include <QMap>
#include <QString>

struct ColumnInfo {
  enum ColumnType {
    LABEL,
    CHECK,
  };

  /**
   * @brief The column id, e.g. Label, Visibility, Selectivity.
   */
  QString name;
  /**
   * @brief The column number
   */
  int col;
  /**
   * @brief The column type, e.g. text, checkbox.
   */
  ColumnType type;
  /**
   * @brief The column number of the master. If a column is the master of other
   * columns, when it unchecked, it's slaves will be disabled.
   */
  int master;
};

class GuiTreeItemColumn {
 public:
  GuiTreeItemColumn() = default;
  virtual ~GuiTreeItemColumn() = default;
  void loadConfig();
  /**
   * @brief Get the column number of label
   * @return
   */
  int labelCol() { return _label_col; };
  int colCount() { return _columns.size(); };
  ColumnInfo getColumn(int col) { return _columns.value(col); };
  QList<ColumnInfo> get_columns() { return _columns.values(); };

 private:
  QMap<int, ColumnInfo> _columns;
  int _label_col;
};

#endif  // GUICONTROLTREECOLUMN_H
