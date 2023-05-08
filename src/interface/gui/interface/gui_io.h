#pragma once
#include <string>
#include <vector>

#include "DrcViolationSpot.h"
#include "builder.h"
#include "file_cts.h"
#include "mainwindow.h"

#define guiInst (igui::GuiIO::getInstance())
using namespace idb;

namespace igui {
  class GuiIO {
   public:
    static GuiIO* getInstance() {
      if (!_instance) {
        _instance = new GuiIO;
      }
      return _instance;
    }
    /// getter
    MainWindow* get_main_window() { return _gui_win; }
    /// win operation
    void init();
    void startGUI(IdbBuilder* builder = nullptr, std::string type = "");
    int exec();
    void showWindow();
    void hideWindow();

    /// data operation-
    void readDB(std::vector<std::string> lef_paths, std::string def_path);
    void readDB(IdbBuilder* _builder);
    void readDrcDb(std::map<std::string, std::vector<idrc::DrcViolationSpot*>>& drc_db, int max_num = -1);
    void readClockTreeDb(std::vector<iplf::CtsTreeNodeMap*>& node_list);

    /// iFP
    bool autoRunFloorplan();

    /// iPL
    bool autoRunPlacer();
    void updateInstanceInFastMode(std::vector<iplf::FileInstance>& file_inst_list);
    bool guiUpdateInstanceInFastMode(std::string directory = "", bool b_reset = false);

    /// iCTS
    bool autoRunCTS();

    /// iRouter
    bool autoRunRouter();

    /// iDRC
    bool autoRunDrcCheckDef();

    /// timer
    void timerStart(int ms = 1000);
    void timerStop();

   private:
    static GuiIO* _instance;

    MainWindow* _gui_win = nullptr;
    QApplication* _app   = nullptr;

    GuiIO() { }
    ~GuiIO() = default;
  };

}  // namespace igui
