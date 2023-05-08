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
