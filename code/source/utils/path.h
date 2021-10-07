/*
 * @Author: zack 
 * @Date: 2021-10-05 10:31:33 
 * @Last Modified by:   zack 
 * @Last Modified time: 2021-10-05 10:31:33 
 */
#pragma once

namespace BASE_NAMESPACE{

#include<string>


std::string basename(const std::string& path,bool with_suffix);
std::string suffixname(const std::string& path);

};