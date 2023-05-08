/**
 * @file guijsonparser.h
 * @author Wang Jun (wen8365@gmail.com)
 * @brief Util for parsing json configuration files
 * @version 0.1
 * @date 2021-07-02
 * 
 * @copyright Copyright (c) 2021
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
