#include<iostream>
#include<string>


namespace BASE_NAMESPACE{

/**
 * 获取路径中的文件名
 * with_suffix:是否带后缀
 */
std::string basename(const std::string& path,bool with_suffix){
  std::string::size_type iPos = path.find_last_of('\\')+1;
  std::string filename = path.substr(iPos,path.length()-iPos);
  if(!with_suffix){
     return filename.substr(0,filename.rfind("."));
  }else{
    return filename;
  }
}

/**
 * 获取文件后缀名
 */
std::string suffixname(const std::string& path){
  return path.substr(path.find_last_of('.')+1);
}

};
