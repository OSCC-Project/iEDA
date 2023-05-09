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
 * @file guijsonparser.h
 * @author Wang Jun (wen8365@gmail.com)
 * @brief Util for parsing json configuration files
 * @version 0.1
 * @date 2021-07-02
 * 
 *
 * 
 */

#ifndef GUIJSONPARSE_H
#define GUIJSONPARSE_H

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>

class GuiJSONParser {
 public:
  /**
   * @brief parse the given JSON file into a QJsonArray.
   * @param file_path
   * @return the parse result, true if pasrse success, false if otherwise.
   */
  static QJsonArray parseJsonArrary(QString file_path);

  /**
   * @brief parse the given JSON file into a QJsonObject.
   * @param file_path
   * @return the parse result, true if pasrse success, false if otherwise.
   */
  static QJsonObject parseJsonObject(QString file_path);

 private:
  static bool parse(QString file_path, QJsonDocument &doc);
};

#endif  // GUIJSONPARSE_H
