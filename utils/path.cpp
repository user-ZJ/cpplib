#include<iostream>
#include<string>

std::string basename(const std::string& path,bool with_suffix){
  std::string::size_type iPos = path.find_last_of('\\')+1;
  std::string filename = path.substr(iPos,path.length()-iPos);
  if(with_suffix){
     return filename.substr(0,filename.rfind("."));
  }else{
    return filename;
  }
}

std::string suffixname(const std::string& path){
  return path.substr(path.find_last_of('.')+1);
}


