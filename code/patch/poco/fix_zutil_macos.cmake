# Poco 1.13.0 内置的 zlib 在现代 macOS SDK 下会命中 TARGET_OS_MAC 分支，
# 把 fdopen 定义成 NULL 而与 <stdio.h> 冲突；zlib 1.2.12 起上游改为只判断 MACOS。
# 在 Poco 源码根目录下以 cmake -P 执行，可重复执行。
set(zutil Foundation/src/zutil.h)
file(READ ${zutil} content)
string(REPLACE "#if defined(MACOS) || defined(TARGET_OS_MAC)" "#if defined(MACOS)" content "${content}")
file(WRITE ${zutil} "${content}")
