#include "Poco/Observer.h"
#include "Poco/Runnable.h"
#include "Poco/Task.h"
#include "Poco/TaskManager.h"
#include "Poco/TaskNotification.h"
#include "Poco/ThreadPool.h"
#include "Poco/Process.h"
#include "Poco/PipeStream.h"
#include "Poco/StreamCopier.h"
#include <fstream>
#include <iostream>

using Poco::Observer;
using Poco::Process;
using Poco::ProcessHandle;

class SampleTask : public Poco::Task {
 public:
  SampleTask(const std::string &name) : Task(name) {}
  void runTask() {
    for (int i = 0; i < 10; ++i) {
      setProgress(float(i) / 10);  // report progress
      sleep(100);
    }
  }
};

class ProgressHandler {
 public:
  void onProgress(Poco::TaskProgressNotification *pNf) {
    std::cout << pNf->task()->name() << " progress: " << pNf->progress() << std::endl;
    pNf->release();
  }
  void onFinished(Poco::TaskFinishedNotification *pNf) {
    std::cout << pNf->task()->name() << " finished." << std::endl;
    pNf->release();
  }
};

class HelloRunnable : public Poco::Runnable {
  virtual void run() {
    std::cout << "Hello, world!" << std::endl;
  }
};

int main(int argc, char **argv) {
  Poco::TaskManager tm;
  ProgressHandler pm;
  //   tm.addObserver(Observer<ProgressHandler, Poco::TaskProgressNotification>(pm, &ProgressHandler::onProgress));
  //   tm.addObserver(Observer<ProgressHandler, Poco::TaskFinishedNotification>(pm, &ProgressHandler::onFinished));
  //   tm.start(new SampleTask("Task 1"));  // tm takes ownership
  //   tm.start(new SampleTask("Task 2"));
  //   tm.joinAll();
  HelloRunnable runnable;
  Poco::ThreadPool::defaultPool().start(runnable);
  Poco::ThreadPool::defaultPool().joinAll();
  std::string cmd("/bin/ps");
  std::vector<std::string> args;
  args.push_back("-ax");
  Poco::Pipe outPipe;
  ProcessHandle ph = Process::launch(cmd, args, 0, &outPipe, 0);
  Poco::PipeInputStream istr(outPipe);
  std::ofstream ostr("processes.txt");
  Poco::StreamCopier::copyStream(istr, ostr);
  return 0;
}