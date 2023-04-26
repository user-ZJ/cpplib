#include "utils/file-util.h"
#include "utils/flags.h"
#include "utils/logging.h"
#include "utils/string-util.h"
#include <cassert>
#include <complex>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

using namespace BASE_NAMESPACE;
using namespace std;

void permute(const vector<int> &nums, vector<vector<int>> &res, vector<int> &path, unordered_set<int> &used,
             std::vector<int> &last, std::vector<std::pair<int, int>> &limits) {
  if (path.size() == nums.size()) {
    res.push_back(path);
    return;
  }
  for (int i = 0; i < nums.size(); i++) {
    if (used.count(i) || (i > 0 && nums[i] == nums[i - 1] && !used.count(i - 1))) continue;
    // 添加限制条件
    if (last[nums[i] - 1] <= path.size()) continue;
    bool bk = false;
    for (const auto &l : limits) {
      if (nums[i] == l.first and std::find(path.begin(), path.end(), l.second) != path.end()) {
        bk = true;
        break;
      }
    }
    if (bk) continue;
    used.insert(i);
    path.push_back(nums[i]);
    permute(nums, res, path, used,last, limits);
    path.pop_back();
    used.erase(i);
  }
}

vector<vector<int>> permute(vector<int> &nums, std::vector<int> &last, std::vector<std::pair<int, int>> &limits) {
  vector<vector<int>> res;
  vector<int> path;
  unordered_set<int> used;
  permute(nums, res, path, used, last, limits);
  return res;
}
// std::vector<std::vector<int>> permute(std::vector<int> &nums) {
//   std::vector<std::vector<int>> res;
//   std::sort(nums.begin(), nums.end());  // 排序，方便去重
//   permute(nums, res, 0);
//   return res;
// }

int main(int argc, char *argv[]) {
  google::EnableLogCleaner(30);  // keep your logs for 30 days
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);  //初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  google::InstallFailureSignalHandler();
  std::vector<int> last{4, 5, 2, 5, 4};
  std::vector<std::pair<int, int>> limits = {{1, 2}, {3, 2}, {5, 1}, {3, 4}, {3, 1}};
  std::vector<int> p{1, 2, 3, 4, 5};

  // 计算所有可能输出路径
  auto res = permute(p,last,limits);
  std::vector<int> mincheck(p.size(),p.size());
  for (int i = 0; i < res.size(); i++) {
    LOG(INFO) << printCollection(res[i]);
    for(int j=0;j<res[i].size();j++)
      mincheck[res[i][j]-1] = std::min(mincheck[res[i][j]-1],j+1);
  }
  LOG(INFO) << res.size();
  LOG(INFO)<<printCollection(mincheck);

  return 0;
}