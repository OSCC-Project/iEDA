#include "guijsonparser.h"

#include <QDebug>
#include <QFile>

QJsonArray GuiJSONParser::parseJsonArrary(QString file_path) {
  QJsonArray json_arr;
  QJsonDocument doc;
  bool ok = parse(file_path, doc);
  if (!ok) {
    qDebug() << "JSON parse abort! File path is " << file_path;
    return json_arr;
  }
  if (!doc.isArray()) {
    qDebug() << "No json array detectd!";
    return json_arr;
  }
  return doc.array();
}

QJsonObject GuiJSONParser::parseJsonObject(QString file_path) {
  QJsonObject json_obj;
  QJsonDocument doc;
  bool ok = parse(file_path, doc);
  if (!ok) {
    qDebug() << "JSON parse abort!";
    return json_obj;
  }
  if (!doc.isObject()) {
    qDebug() << "No json object detectd!";
    return json_obj;
  }
  return doc.object();
}

bool GuiJSONParser::parse(QString file_path, QJsonDocument &doc) {
  QFile *json_file = new QFile(file_path);
  if (!json_file->exists() ||
      !json_file->open(QIODevice::ReadOnly | QIODevice::Text)) {
    return false;
  }
  QByteArray json_str = json_file->readAll();
  QJsonParseError error;
  doc = QJsonDocument::fromJson(json_str, &error);
  if (doc.isNull()) {
    qDebug() << "JSON [" << file_path
             << "] parse error: " << error.errorString();
    return false;
  }
  if (json_file->isOpen()) {
    json_file->close();
  }

  return true;
}
