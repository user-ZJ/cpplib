#include<iostream>
#include<string>
#include<vector>
#include "utils/logging.h"
#include "utils/flags.h"
#include "utils/string-util.h"
#include <omp.h>


using namespace BASE_NAMESPACE;

int counter = 0;
#pragma omp threadprivate(counter)


void fun()
{
    #pragma omp parallel
    {
        int count;
        #pragma omp single copyprivate(counter)
        {
            counter = 50;
        }
        count = ++counter;
        printf("ThreadId: %d, count = %d\n", omp_get_thread_num(), count);
    }
}

int main(int argc, char *argv[]) {
  FLAGS_log_dir = ".";
  google::EnableLogCleaner(30);  // keep your logs for 30 days
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);  //初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  fun();
  return 0;
}
