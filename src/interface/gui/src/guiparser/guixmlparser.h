/**
 * @file guixmlparser.h
 * @author Wang Jun (wen8365@gmail.com)
 * @brief Util for parsing xml configuration files
 * @version 0.1
 * @date 2021-07-02
 * 
 * @copyright Copyright (c) 2021
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
