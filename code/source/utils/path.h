/*
 * @Author: zack
 * @Date: 2021-10-05 10:31:33
 * @Last Modified by: zack
 * @Last Modified time: 2021-10-15 18:24:32
 *
 * 参考https://github.com/zlib-ng/minizip-ng/blob/master/doc/mz_os.md实现
 */
#pragma once

#include <cstring>
#include <sys/stat.h>

namespace BASE_NAMESPACE {

#define PATH_SLASH_UNIX ('/')
#if defined(_WIN32)
#define PATH_SLASH_PLATFORM ('\\')
#else
#define PATH_SLASH_PLATFORM (PATH_SLASH_UNIX)
#endif

#include <string>

//添加路径分隔符
std::string path_append_slash(const std::string &path, const char slash) {
  if (path[path.size() - 1] != '\\' && path[path.size() - 1] != '/')
    return path + slash;
  return path;
}

//删除路径末尾的分隔符
std::string path_remove_slash(const std::string &path) {
  int sub_len = path.size();
  while (sub_len >= 0) {
    if (path[sub_len - 1] == '\\' || path[sub_len - 1] == '/')
      sub_len--;
    else
      break;
  }
  return path.substr(0, sub_len);
}

// 在路径末尾是否存在分隔符
bool path_has_slash(const std::string &path) {
  if (path[path.size() - 1] == '\\' || path[path.size() - 1] == '/')
    return true;
  return false;
}

// 将windows分隔符转换为linux分隔符，或linux路径分隔符转换为linux路径分隔符
// 如果路径中同时存在linux和windows分隔符，则转换为同一种格式
std::string path_convert_slashes(const std::string &path, const char slash) {
  std::string res = path;
  for (int i = 0; i < path.size(); i++) {
    if (res[i] == '\\' || res[i] == '/')
      res[i] = slash;
  }
  return res;
}

// 解析路径，删除路径中的 . 和 ..
std::string path_resolve(const std::string &path) {
  char output[path.size() + 1];
//   char input[path.size() + 1];
//   std::strcpy(input, path.c_str());
  const char *source = path.c_str();
  const char *check = output;
  char *target = output;

  int max_output = path.size() + 1;

  while (*source != 0 && max_output > 1) {
    check = source;
    if ((*check == '\\') || (*check == '/'))
      check += 1;

    if ((source == path) || (target == output) || (check != source)) {
      /* Skip double paths */
      if ((*check == '\\') || (*check == '/')) {
        source += 1;
        continue;
      }
      if (*check == '.') {
        check += 1;

        /* Remove . if at end of string and not at the beginning */
        if ((*check == 0) && (source != path && target != output)) {
          /* Copy last slash */
          *target = *source;
          target += 1;
          max_output -= 1;
          source += (check - source);
          continue;
        }
        /* Remove . if not at end of string */
        else if ((*check == '\\') || (*check == '/')) {
          source += (check - source);
          /* Skip slash if at beginning of string */
          if (target == output && *source != 0)
            source += 1;
          continue;
        }
        /* Go to parent directory .. */
        else if (*check == '.') {
          check += 1;
          if ((*check == 0) || (*check == '\\' || *check == '/')) {
            source += (check - source);

            /* Search backwards for previous slash */
            if (target != output) {
              target -= 1;
              do {
                if ((*target == '\\') || (*target == '/'))
                  break;

                target -= 1;
                max_output += 1;
              } while (target > output);
            }

            if ((target == output) && (*source != 0))
              source += 1;
            if ((*target == '\\' || *target == '/') && (*source == 0))
              target += 1;

            *target = 0;
            continue;
          }
        }
      }
    }

    *target = *source;

    source += 1;
    target += 1;
    max_output -= 1;
  }

  *target = 0;
  return std::string(output);
}

// 删除路径中的文件名
std::string path_remove_filename(const std::string &path) {
  int sub_len = path.size();
  while (sub_len >= 0) {
    if (path[sub_len - 1] == '\\' || path[sub_len - 1] == '/'){
        sub_len--;
        break;
    }
    sub_len--;
  }
  return path.substr(0, sub_len);
}

// 删除路径中文件扩展名
std::string path_remove_extension(const std::string &path) {
  int sub_len = path.size();
  while (sub_len >= 0) {
    if (path[sub_len - 1] == '\\' || path[sub_len - 1] == '/')
      break;
    if (path[sub_len - 1] == '.'){
        sub_len--;
        break;
    }
    sub_len--;
  }
  return path.substr(0, sub_len);
}

// 路径拼接
std::string path_combine(const std::string &path, const std::string &join) {
  if (path.size() == 0)
    return join;
  std::string result = path_append_slash(path, PATH_SLASH_PLATFORM);
  return result + join;
}

// 获取路径中的文件名
std::string path_get_filename(const std::string &path) {
  int pos = path.size() - 1;
  while (pos >= 0) {
    if (path[pos] == '\\' || path[pos] == '/')
      break;
    pos--;
  }
  return path.substr(pos + 1);
}

std::string extension_name(const std::string &path){
  return path.substr(path.find_last_of('.')+1);
}


///////////////////   directory  /////////////////////////////

// 创建目录
int32_t os_make_dir(const char *path) {                                                                                                                                                                                              
    int32_t err = 0;
    err = mkdir(path, 0755);
    if (err != 0 && errno != EEXIST)
        return -104;
    return 0;
}

// 创建目录
int32_t dir_make(const char *path){
    int32_t err = 0;
    int16_t len = 0;
    char *current_dir = NULL;
    char *match = NULL;
    char hold = 0;


    len = (int16_t)strlen(path);
    if (len <= 0)
        return 0;

    current_dir = (char *)malloc((uint16_t)len + 1);
    if (current_dir == NULL)
        return -4;

    strcpy(current_dir, path);
    path_remove_slash(current_dir);

    err = os_make_dir(current_dir);
    if (err != 0) {
        match = current_dir + 1;
        while (1) {
            while (*match != 0 && *match != '\\' && *match != '/')
                match += 1;
            hold = *match;
            *match = 0;

            err = os_make_dir(current_dir);
            if (err != 0)
                break;
            if (hold == 0)
                break;

            *match = hold;
            match += 1;
        }
    }

    free(current_dir);
    return err;
}

// 判断是否是目录
bool is_dir(const char *path) {                                                                                                                                                                                                
    struct stat path_stat;
    memset(&path_stat, 0, sizeof(path_stat));
    stat(path, &path_stat);
    if (S_ISDIR(path_stat.st_mode))
        return true;

    return false;
}

// 是否是链接
bool is_symlink(const char *path) {                                                                                                                                                                                            
    struct stat path_stat;

    memset(&path_stat, 0, sizeof(path_stat));
    lstat(path, &path_stat);
    if (S_ISLNK(path_stat.st_mode))
        return true;
    return false;
}


///////////////////   directory  /////////////////////////////


}; // namespace BASE_NAMESPACE