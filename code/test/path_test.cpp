#include<iostream>
#include<string>
#include "path.h"
#include "utils/logging.h"


using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]); 
  std::string path = "C:\\Users\\Administrator\\..\\..\\Desktop\\text\\data.22.txt";
  // std::string path = "/Users/Administrator/Desktop/text/data.22.txt";
  // std::string path = "data.22.txt";
  LOG(INFO)<<"path_append_slash:"<<path_append_slash(path,'/');
  LOG(INFO)<<"path_remove_slash:"<<path_remove_slash(path_append_slash(path,'/'));
  LOG(INFO)<<"path_has_slash:"<<path_has_slash(path);
  LOG(INFO)<<"path_convert_slashes:"<<path_convert_slashes(path,'/');
  LOG(INFO)<<"path_remove_filename:"<<path_remove_filename(path);
  LOG(INFO)<<"path_remove_extension:"<<path_remove_extension(path);
  LOG(INFO)<<"path_combine:"<<path_combine(path,std::string("a.txt"));
  LOG(INFO)<<"extension_name:"<<extension_name(path);
  LOG(INFO)<<"path_get_filename:"<<path_get_filename(path);
  LOG(INFO)<<"path_resolve:"<<path_resolve(path);
}
