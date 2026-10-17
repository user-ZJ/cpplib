# 在 openfst 1.6.5 源码根目录下以 cmake -P 执行，可重复执行。
#
# 1. configure 把 -std=c++11 追加在用户 CXXFLAGS 之后，无法从外部覆盖；glog 0.7 头文件要求 C++14。
#    只改生成好的 configure：改动 configure.ac 会触发 automake 重新生成，要求本机装有 aclocal。
# 2. bi-table.h 的 VectorHashBiTable 拷贝构造访问了不存在的成员 s_，clang 17 起会报错；
#    改为与上游 1.8.4 相同的写法。

function(replace_in_file file from to)
  file(READ ${file} content)
  string(REPLACE "${from}" "${to}" content "${content}")
  file(WRITE ${file} "${content}")
endfunction()

replace_in_file(configure "-std=c++11" "-std=c++14")
replace_in_file(src/include/fst/bi-table.h "selector_(new S(table.s_))" "selector_(new S(*table.selector_))")
