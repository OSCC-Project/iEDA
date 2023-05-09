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
 * @file guitreeitemcolumn.cpp
 * @author Wang Jun (wen8365@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-07-02
 * 
 *
 * 
 */

#include "guitreeitemcolumn.h"

#include <QJsonArray>

#include "guijsonparser.h"

void GuiTreeItemColumn::loadConfig() {
  QJsonArray json_arr =
      //   GuiJSONParser::parseJsonArrary(":/conf/ItemColumn.json");
      GuiJSONParser::parseJsonArrary("./iEDA/src/iGUI/res/conf/ItemColumn.json");
  QJsonArray::iterator iterator;
  for (iterator = json_arr.begin(); iterator != json_arr.end(); ++iterator) {
    QJsonObject obj = (*iterator).toObject();
    ColumnInfo column;

    QJsonValue value = obj.value("name");
    if (value.type() == QJsonValue::String) {
      column.name = value.toString().trimmed();
    }

    value = obj.value("col");
    if (value.type() == QJsonValue::Double) {
      column.col = value.toInt();
    }

    value = obj.value("type");
    if (value.type() == QJsonValue::Double) {
      column.type = (ColumnInfo::ColumnType)value.toInt();
    }

    value = obj.value("master");
    if (value.type() == QJsonValue::Double) {
      column.master = value.toInt();
    }

    if (column.type == ColumnInfo::LABEL) {
      _label_col = column.col;
    }

    _columns.insert(column.col, column);
  }
}
