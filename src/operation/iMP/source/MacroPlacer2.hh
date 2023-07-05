#ifndef IMP_MACROPLACER
#define IMP_MACROPLACER
#include <string>
namespace imp {
class Option;
class DataManager;
class Summary;

class MacroPlacer
{
 public:
  MacroPlacer(DataManager* dm, Option* opt);
  MacroPlacer(const std::string& idb_json, const std::string& opt_json);
  MacroPlacer();
  ~MacroPlacer();

  void setDataManager(DataManager* dm);
  void setDataManager(const std::string& idb_json);
  // open functions
  void runMacroPlacer();

 private:
  //   void setConfig(Option* opt);
  void updateDensity();
  void addHalo();
  void deleteHalo();
  void writeSummary();
  void plot();

 private:
  DataManager* _dm;
  Option* _opt;
};
}  // namespace imp
#endif