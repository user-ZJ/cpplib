#include<iostream>
#include<string>


namespace BASE_NAMESPACE{


/**
 * 获取文件扩展名
 */
std::string extension_name(const std::string& path){
  return path.substr(path.find_last_of('.')+1);
}

};
