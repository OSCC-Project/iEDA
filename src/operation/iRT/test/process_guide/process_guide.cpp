#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Guide.hpp"
#include "Net.hpp"
#include "Util.hpp"

void readGuide(const std::string& guide_file_path, std::vector<Net>& net_list)
{
  std::ifstream* guide_file_stream = getInputFileStream(guide_file_path);

  std::string new_line;
  bool is_new_net = true;

  std::string net_name, layer_name;
  while (getline(*guide_file_stream, new_line)) {
    int lb_x, lb_y, rt_x, rt_y;
    if (is_new_net) {
      net_name = new_line;
      is_new_net = false;
    } else if (new_line == "(") {
      std::vector<Guide> guide_list;
      while (getline(*guide_file_stream, new_line)) {
        if (new_line == ")") {
          is_new_net = true;

          Net net;
          net.set_net_name(net_name);
          net.set_guide_list(guide_list);
          net_list.push_back(net);

          static int net_num = 0;
          net_num++;
          if (net_num % 10000 == 0) {
            std::cout << "read " << net_num << " nets" << std::endl;
          }
          break;
        }
        std::istringstream str(new_line);
        str >> lb_x >> lb_y >> rt_x >> rt_y >> layer_name;

        Guide guide;
        guide.set_lb_x(lb_x);
        guide.set_lb_y(lb_y);
        guide.set_rt_x(rt_x);
        guide.set_rt_y(rt_y);
        guide.set_layer_name(layer_name);
        guide_list.push_back(guide);
      }
    }
  }
  closeFileStream(guide_file_stream);
  std::cout << "read end" << std::endl;
}

void writeGuide(std::vector<Net>& net_list, const std::string& guide_file_path)
{
  std::ofstream* guide_file_stream = getOutputFileStream(guide_file_path);

  for (size_t i = 0; i < net_list.size(); i++) {
    Net& net = net_list[i];
    (*guide_file_stream) << net.get_net_name() << "\n(\n";
    for (Guide& guide : net.get_guide_list()) {
      (*guide_file_stream) << guide.get_lb_x() << " " << guide.get_lb_y() << " " << guide.get_rt_x() << " " << guide.get_rt_y() << " "
                           << guide.get_layer_name() << "\n";
    }
    (*guide_file_stream) << ")\n";
    if ((i + 1) % 10000 == 0) {
      std::cout << "wrote " << (i + 1) << " nets" << std::endl;
    }
  }
  closeFileStream(guide_file_stream);
  std::cout << "write end" << std::endl;
}

int main()
{
  const std::string& guide_file_path_read = "";
  const std::string& guide_file_path_write = "";

  std::vector<Net> net_list;
  readGuide(guide_file_path_read, net_list);
  writeGuide(net_list, guide_file_path_write);

  std::cout << "////////////////////////////////////////////" << std::endl;
  std::cout << "guide_file_path_read: " << guide_file_path_read << std::endl;
  std::cout << "guide_file_path_write: " << guide_file_path_write << std::endl;
  std::cout << "************ file_diff_lines_num: " << countDifferentLines(guide_file_path_read, guide_file_path_write) << " ************"
            << std::endl;
  std::cout << "////////////////////////////////////////////" << std::endl;

  return 0;
}