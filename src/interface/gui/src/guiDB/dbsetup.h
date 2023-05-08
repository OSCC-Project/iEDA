#ifndef DBSETUP_H
#define DBSETUP_H

#include <list>
#include <map>
#include <string>
#include <vector>

class GuiGraphicsScene;
class QGraphicsItem;

enum class DbSetupType {
  kNone,
  kChip,
  kFloorplan,
  kGlobalPlace,
  kDetailPlace,
  kGlobalRouting,
  kDetailRouting,
  kClockTree,
  kMax
};

class DbSetup {
 public:
  //   DbSetup(const std::map<std::string, std::list<std::string>>& filemap, GuiGraphicsScene* scene);
  DbSetup(GuiGraphicsScene* scene);
  DbSetup(const std::vector<std::string>& lef_paths, const std::string& def_path, GuiGraphicsScene* scene);
  ~DbSetup() = default;

  virtual void initDB(DbSetupType type = DbSetupType::kChip) { }
  virtual void createChip(DbSetupType type = DbSetupType::kChip) { }
  virtual int32_t getLayerCount()             = 0;
  virtual std::vector<std::string> getLayer() = 0;
  virtual void fitView(double width = 0, double height = 0);
  virtual void update(std::string node_name, std::string parent_name) = 0;

  /// getter
  std::vector<std::string> get_lef_paths() { return _lef_paths; }
  std::string get_def_path() { return _def_path; }
  DbSetupType get_type() { return _type; }

  bool is_floorplan() { return _type == DbSetupType::kFloorplan ? true : false; }
  ////setter
  void set_type(DbSetupType type) { _type = type; }

  void addItem(QGraphicsItem* item);

  /// operator

 protected:
  GuiGraphicsScene* _scene;
  //   std::map<std::string, std::list<std::string>> _fileMap;
  std::vector<std::string> _lef_paths;
  std::string _def_path;
  DbSetupType _type;
};

static DbSetupType translate_type(std::string type) {
  if (type == "floorplan") {
    return DbSetupType::kFloorplan;
  } else if (type == "global_place") {
    return DbSetupType::kGlobalPlace;
  } else if (type == "clock_tree") {
    return DbSetupType::kClockTree;
  } else {
    return DbSetupType::kChip;
  }
}

#endif  // DBSETUP_H
