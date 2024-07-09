/**
 * @file main.cpp
 * @brief The main function of iPNP. Refer to iSTA/main.cc. 
 * function: Launch the tcl console and interact with the user.
 * @version 0.1
 * @date 2024-06-20
 */

#include <iostream>
#include <string>
#include "iPNP.hh"

// int registerCommands() {
//   registerTclCmd(CmdSetDesignWorkSpace, "set_design_workspace");
//   registerTclCmd(CmdReadVerilog, "read_netlist");
//   registerTclCmd(CmdReadLefDef, "read_lef_def");
//   registerTclCmd(CmdReadLiberty, "read_liberty");
//   registerTclCmd(CmdLinkDesign, "link_design");
//   registerTclCmd(CmdReadSpef, "read_spef");
//   registerTclCmd(CmdReadSdc, "read_sdc");
//   registerTclCmd(CmdReportTiming, "report_timing");
//   registerTclCmd(CmdReportConstraint, "report_constraint");
//   registerTclCmd(CmdDefToVerilog, "def_to_verilog");
//   registerTclCmd(CmdVerilogToDef, "verilog_to_def");

//   return EXIT_SUCCESS;
// }

int main(int argc, char** argv) {
  using namespace ipnp;
  std::string config_file = "config file path";
  iPNP pnp_object = iPNP(config_file);
  pnp_object.run();

  // Log::init(argv);

  // // for debug
  // // Log::setVerboseLogLevel("Arnoldi*", 1);

  // // for (int i = 0; i < argc; ++i) {
  // //   std::cout << "argv[" << i << "] = " << argv[i] << std::endl;
  // // }

  // std::string hello_info =
  //     "\033[49;32m***************************\n"
  //     "  _  _____  _____   ___  \n"
  //     " (_)/  ___||_   _| / _ \\ \n"
  //     "  _ \\ `--.   | |  / /_\\ \\\n"
  //     " | | `--. \\  | |  |  _  |\n"
  //     " | |/\\__/ /  | |  | | | |\n"
  //     " |_|\\____/   \\_/  \\_| |_/\n"
  //     "***************************\n"
  //     "WELCOME TO iSTA TCL-shell interface. \e[0m";

  // // get an UserShell (singleton) instance
  // auto shell = ieda::UserShell::getShell();

  // // set call back for register commands
  // shell->set_init_func(registerCommands);

  // // if no args, run interactively
  // cxxopts::Options options("iSTA", "iSTA command line help.");
  // options.add_options()("v,version", "Print Git Version")(
  //     "h,help", "iSTA command usage.")("script", "Tcl script file",
  //                                      cxxopts::value<std::string>());

  // try {
  //   options.parse_positional("script");

  //   auto argv_parse_result = options.parse(argc, argv);

  //   if (argv_parse_result.count("help")) {
  //     options.custom_help("script [OPTIONS...]");
  //     std::cout << std::endl;
  //     std::cout << options.help() << std::endl;
  //     return 0;
  //   }

  //   shell->displayHello(hello_info);

  //   if (argv_parse_result.count("version")) {
  //     std::cout << "\033[49;32mGit Version: " << GIT_VERSION << "\033[0m"
  //               << std::endl;
  //   }

  //   if (argc == 1) {
  //     shell->displayHelp();
  //     shell->userMain(argc, argv);
  //   } else if (argc == 2) {
  //     shell->userMain(argv[1]);
  //   } else {
  //     if (argv_parse_result.count("script")) {
  //       // discard the first arg from main()
  //       // pass the rest of the args to Tcl interpreter
  //       auto tcl_argc = argc - 1;
  //       auto tcl_argv = argv + 1;
  //       shell->userMain(tcl_argc, tcl_argv);
  //     }
  //   }
  // } catch (const cxxopts::exceptions::exception& e) {
  //   std::cerr << "Error parsing options: " << e.what() << std::endl;
  //   return 1;
  // }

  // Log::end();

  return 0;
}
