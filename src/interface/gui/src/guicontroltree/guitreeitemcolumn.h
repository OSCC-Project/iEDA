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
