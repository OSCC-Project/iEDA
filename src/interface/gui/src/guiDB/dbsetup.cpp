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
#include "dbsetup.h"

#include "guigraphicsscene.h"

// DbSetup::DbSetup(const std::map<std::string, std::list<std::string>>& filemap, GuiGraphicsScene* scene) {
//   _fileMap = filemap;
//   _scene   = scene;
// }

DbSetup::DbSetup(GuiGraphicsScene* scene) { _scene = scene; }

DbSetup::DbSetup(const std::vector<std::string>& lef_paths, const std::string& def_path, GuiGraphicsScene* scene) {
  _lef_paths = lef_paths;
  _def_path  = def_path;
  _scene     = scene;
}

void DbSetup::addItem(QGraphicsItem* item) { _scene->addItem(item); }

void DbSetup::fitView(double width, double height) {
  if (width == 0 || height == 0) {
    return;
  }
  _scene->fitView(width, height);
  _scene->update();
}
