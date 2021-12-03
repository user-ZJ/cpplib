#pragma once
#ifndef GLOG_CUSTOM_PREFIX_SUPPORT
#define GLOG_CUSTOM_PREFIX_SUPPORT
#endif
#include "glog/logging.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

// using namespace google;

namespace BASE_NAMESPACE {

// google::InitGoogleLogging(argv[0]);
// google::InitGoogleLogging(argv[0],&CustomPrefix);


// void CustomPrefix(std::ostream &s, const LogMessageInfo &l, void *) {
//   s << l.severity[0] << std::setw(4) << 1900 + l.time.year() << std::setw(2)
//     << 1 + l.time.month() << std::setw(2) << l.time.day() << ' ' << std::setw(2)
//     << l.time.hour() << ':' << std::setw(2) << l.time.min() << ':'
//     << std::setw(2) << l.time.sec() << "." << std::setw(6) << l.time.usec()
//     << ' ' << std::setfill(' ') << std::setw(5) << l.thread_id
//     << std::setfill('0') << ' ' << l.filename << ':' << l.line_number << "]";
// }


// class LoggingWrapper {
//  public:
//   enum class LogSeverity : int {
//     INFO = 0,
//     WARN = 1,
//     ERROR = 2,
//     FATAL = 3,
//   };
//   LoggingWrapper(LogSeverity severity)
//       : severity_(severity), should_log_(true) {}
//   LoggingWrapper(LogSeverity severity, bool log)
//       : severity_(severity), should_log_(log) {}
//   std::stringstream& Stream() { return stream_; }
//   ~LoggingWrapper() {
//     if (should_log_) {
//       switch (severity_) {
//         case LogSeverity::INFO:
//         case LogSeverity::WARN:
//           std::cout << stream_.str() << std::endl;
//           break;
//         case LogSeverity::ERROR:
//           std::cerr << stream_.str() << std::endl;
//           break;
//         case LogSeverity::FATAL:
//           std::cerr << stream_.str() << std::endl;
//           std::flush(std::cerr);
//           std::abort();
//           break;
//       }
//     }
//   }

//  private:
//   std::stringstream stream_;
//   LogSeverity severity_;
//   bool should_log_;
// };

// #define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 :
// __FILE__)
// // #define LOG std::cout<<__FILENAME__<<":"<<__LINE__<<" "
// //#define LOG std::cout<<__DATE__<<" "<<__TIME__<<"
// "<<__FILENAME__<<":"<<__LINE__<<" " #define LOG(severity)
// LoggingWrapper(LoggingWrapper::LogSeverity::severity).Stream()<<__FILENAME__<<":"<<__LINE__<<"
// "

// #define MAY_LOG(severity, should_log)
// LoggingWrapper(LoggingWrapper::LogSeverity::severity,
// (should_log)).Stream()<<__FILENAME__<<":"<<__LINE__<<" "

}; // namespace BASE_NAMESPACE
