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
