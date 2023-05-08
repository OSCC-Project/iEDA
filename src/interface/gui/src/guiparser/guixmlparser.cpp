#include "guixmlparser.h"

#include <QDebug>
#include <QFile>
#include <QXmlSimpleReader>

bool GuiXMLParser::parse(QString file_path, QXmlDefaultHandler* handler) {
  QFile* file = new QFile(file_path);
  if (!file->open(QIODevice::ReadOnly | QIODevice::Text)) return false;

  QXmlSimpleReader xml_reader;
  QXmlInputSource* source = new QXmlInputSource(file);

  xml_reader.setContentHandler(handler);
  xml_reader.setErrorHandler(handler);

  bool ok = xml_reader.parse(source);
  if (!ok) {
    qDebug() << "Parsing xml [" << file_path
             << "] failed: " << handler->errorString();
    return false;
  }
  if (file->isOpen()) file->close();

  return true;
}
