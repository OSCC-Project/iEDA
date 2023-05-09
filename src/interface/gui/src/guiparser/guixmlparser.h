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
 * @file guixmlparser.h
 * @author Wang Jun (wen8365@gmail.com)
 * @brief Util for parsing xml configuration files
 * @version 0.1
 * @date 2021-07-02
 * 
 *
 * 
 */

#ifndef GUIXMLPARSER_H
#define GUIXMLPARSER_H

#include <QString>
#include <QXmlDefaultHandler>

class GuiXMLParser {
 public:
  /**
   * @brief use the given handler to parse the given XML file.
   * @param file_path
   * @param handler handler use for handling content and error.
   * @return the parse result, true if pasrse success, false if otherwise.
   */
  static bool parse(QString file_path, QXmlDefaultHandler* handler);
};

#endif  // GUIXMLPARSER_H
