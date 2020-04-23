

```cmake
cmake_minimum_required (VERSION 2.6)   该语句一般都可以放置在CMakeLists.txt的开头，用于说明CMake最低版本要求

add_executable()是标注生成的执行文件名和使用的source 文件名，如：add_executable(Tutorial tutorial.cpp)

PROJECT(name)  该指令一般置于CMakeLists.txt的开头，定义了工程的名称。但项目最终编译生成的可执行文件并不一定是这个项目名称，而是由另一条命令确定的
执行了该条指令之后，将会自动创建两个变量：
PROJECT_BINARY_DIR = <projectname>_BINARY_DIR：二进制文件保存路径；
PROJECT_SOURCE_DIR = <projectname>_SOURCE_DIR：源代码路径

SET(VAR [VALUE] [CACHE TYPEDOCSTRING [FORCE]]) 例如：SET(CMAKE_INSTALL_PREFIX /usr/local)显式的将CMAKE_INSTALL_PREFIX的值定义为/usr/local，如此在外部构建情况下执行make install命令时，make会将生成的可执行文件拷贝到/usr/local/bin目录下

ADD_SUBDIRECTORY(source_dir [binary_dir] [EXCLUDE_FROM_ALL]) source_dir：源文件路径；[binary_dir]：中间二进制与目标二进制存放路径；[EXECLUDE_FROM_ALL]：将这个目录从编译过程中排除；这个指令用于向当前工程添加存放源文件的子目录，并可以指定中间二进制和目标二进制存放的位置。EXCLUDE_FROM_ALL 参数的含义是将这个目录从编译过程中排除。比如，工程有时候存在example，可能就需要工程构建完成后，再进入example目录单独进行构建

INCLUDE_DIRECTORIES([AFTER|BEFORE] [SYSTEM] dir1 dir2 ...) 
[AFTER|BEFORE]：追加标志，指定控制追加或置前；
dir1, ..., dir n：
添加的一系列头文件搜索路径；向工程添加多个特定的头文件搜索路径，路径之间用空格分隔
INCLUDE_DIRECTORIES(/usr/include/thrift)

ADD_EXECUTABLE(exename srcname)
exename：可执行文件名
srcname：生成该可执行文件的源文件
该命令给出源文件名称，并指出需要编译出的可执行文件名
SET(SRC_LIST main.cc
        rpc/CRNode.cpp 
        rpc/Schd_types.cpp 
        task/TaskExecutor.cpp
        task/TaskMoniter.cpp
        util/Const.cpp 
        util/Globals.cc
        )
ADD_EXECUTABLE(CRNode ${SRC_LIST})

ADD_LIBRARY(libname [SHARED|STATIC|MODULE] [EXCLUDE_FROM_ALL] source1 source2 ... sourceN)
libname：库文件名称；
[SHARED|STATIC|MODULE]：生成库文件类型（共享库/静态库）MODULE自适应，根据需要编译成动态库或静态库
[EXCLUDE_FROM_ALL]：表示该库不会被默认构建
source1, ..., sourceN：生成库所依赖的源文件
ADD_LIBRARY(hello SHARED ${LIBHELLO_SRC})
ADD_LIBRARY默认构建一个静态库

EXECUTABLE_OUTPUT_PATH为生成可执行文件路径
LIBRARY_OUTPUT_PATH为生成库文件路径
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

LINK_DIRECTORIES(directory1 directory2 ...) 该指令用于添加外部库的搜索路径
TARGET_LINK_LIBRARIES(target library1 <debug | optimized> library2 ..)  
target：目标文件；
library1, ..., libraryN：链接外部库文件
指定链接目标文件时需要链接的外部库
target_link_libraries(zipapp archive)默认链接静态库

MESSAGE([SEND_ERROR | STATUS | FATAL_ERROR] “message to display” …)
SEND_ERROR：产生错误，生成过程被跳过；
STATUS：输出前缀为 -- 的信息
FATAL_ERROR：立即终止所有cmake过程

SET_TARGET_PROPERTIES(target1 target2 ... PROPERTIES prop1 value1 prop2 value2 ...)
该指令为一个目标设置属性，语法是列出所有用户想要变更的文件，然后提供想要设置的值。用户可以使用任何想用的属性与对应的值，并在随后的代码中调用GET_TARGET_PROPERTY命令取出属性的值。
PREFIX覆盖了默认的目标名前缀（如lib）
SUFFIX覆盖了默认的目标名后缀（如.so）
IMPORT_PREFIX, IMPORT_PREFIX与PREFIX, SUFFIX是等价的属性，但针对的是DLL导入库（即共享库目标）。
OUTPUT_NAME用来设置目标的真实名称
LINK_FLAGS为一个目标的链接阶段添加额外标志 	
COMPILE_FLAGS 设置附加的编译器标志，在构建目标内的源文件时被用到
LINKER_LANGUAGE 改变链接可执行文件或共享库的工具。默认值是设置与库中文件相匹配的语言，CXX与C是该属性的公共值
VERSION指定构建的版本号，
SOVERSION指定构建的API版本号

AUX_SOURCE_DIRECTORY 查找某个路径下的所有源文件，并将源文件列表存储到一个变量中
AUX_SOURCE_DIRECTORY(< dir > < variable >)
AUX_SOURCE_DIRECTORY(. SRC_LIST)

INSTALL
INSTALL命令可以按照对象的不同分为三种类型：目标文件、非目标文件、目录
目标文件：INSTALL(TARGETS targets...
    [[ARCHIVE|LIBRARY|RUNTIME]
    [DESTINATION < dir >]
    [PERMISSIONS permissions...]
    [CONFIGURATIONS
    [Debug|Release|...]]
    [COMPONENT < component >]
    [OPTIONAL]
    ] [...])
TARGETS targets：targets即为我们通过ADD_EXECUTABLE或ADD_LIBRARY定义的目标文件，可能是可执行二进制，动态库，静态库；
ARCHIVE|LIBRARY|RUNTIME 静态库，动态库，二进制文件
DESTINATION < dir >：dir即为定义的安装路径。安装路径可以是绝对/相对路径，若如果路径以/开头，则是绝对路径，且绝对路径的情况下，CMAKE_INSTALL_PREFIX就无效了。
如果希望使用CMAKE_INSTALL_PREFIX定义安装路径，就需要使用相对路径，这时候安装后的路径就是${CMAKE_INSTALL_PREFIX}/<dir>
非目标文件：
INSTALL(PROGRAMS files... DESTINATION < dir >
    [PERMISSIONS permissions...]
    [CONFIGURATIONS [Debug|Release|...]]
    [COMPONENT < component >]
    [RENAME < name >] [OPTIONAL])
使用方法基本和上述目标文件指令的INSTALL相同，唯一别的不同是，安装非目标文件之后的权限为OWNER_EXECUTE, GOUP_EXECUTE, WORLD_EXECUTE，即755权限目录的安装
目录：
INSTALL(DIRECTORY dirs... DESTINATION < dir >
    [FILE_PERMISSIONS permissions...]
    [DIRECTORY_PERMISSIONS permissions...]
    [USE_SOURCE_PERMISSIONS]
    [CONFIGURATIONS [Debug|Release|...]]
    [COMPONENT < component >]
    [[PATTERN < pattern > | REGEX < regex >]
    [EXCLUDE] [PERMISSIONS permissions...]] [...])
DIRECTORY dirs：dirs是所在源文件目录的相对路径。但必须注意：abc与abc/有很大区别：
若是abc，则该目录将被安装为目标路径的abc；
若是abc/，则代表将该目录内容安装到目标路径，但不包括该目录本身。
INSTALL(DIRECTORY icons scripts/ DESTINATION share/myproj
    PATTERN "CVS" EXCLUDE
    PATTERN "scripts/*" PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
将icons目录安装到< prefix >/share/myproj；将scripts/中的内容安装到< prefix >/share/myproj；不包含目录名为CVS的目录；对于scripts/*文件指定权限为OWNER_EXECUTE, OWNER_WRITE, OWNER_READ, GROUP_EXECUT, GROUP_READ；

```



## CMake的预定义变量

- PROJECT_SOURCE_DIR：工程根目录；

- PROJECT_BINARY_DIR：运行cmake命令的目录

- CMAKE_INCLUDE_PATH：环境变量，非cmake变量；

- CMAKE_LIBRARY_PATH：环境变量；

  CMAKE_CURRENT_SOURCE_DIR：当前处理的CMakeLists.txt文件所在路径；

  CMAKE_CURRENT_BINARY_DIR：target编译目录；使用ADD_SURDIRECTORY指令可以更改该变量的值；SET(EXECUTABLE_OUTPUT_PATH < dir >) 指令不会对该变量有影响，但改变了最终目标文件的存储路径

CMAKE_CURRENT_LIST_FILE：输出调用该变量的CMakeLists.txt的完整路径；
CMAKE_CURRENT_LIST_LINE：输出该变量所在的行；

CMAKE_MODULE_PATH：定义自己的cmake模块所在路径；

EXECUTABLE_OUTPUT_PATH：重新定义目标二进制可执行文件的存放位置；

LIBRARY_OUTPUT_PATH：重新定义目标链接库文件的存放位置；

PROJECT_NAME：返回由PROJECT指令定义的项目名称；

CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS：用来控制IF...ELSE...语句的书写方式

  

### 系统信息预定义变量

 CMAKE_MAJOR_VERSION cmake主版本号,如2.8.6中的2

CMAKE_MINOR_VERSION cmake次版本号,如2.8.6中的8

CMAKE_PATCH_VERSION cmake补丁等级,如2.8.6中的6

CMAKE_SYSTEM 系统名称,例如Linux-2.6.22

CMAKE_SYSTEM_NAME 不包含版本的系统名,如Linux

CMAKE_SYSTEM_VERSION 系统版本,如2.6.22

CMAKE_SYSTEM_PROCESSOR 处理器名称,如i686

UNIX 在所有的类UNIX平台为TRUE,包括OS X和cygwin

WIN32 在所有的win32平台为TRUE,包括cygwin

### 开关选项

BUILD_SHARED_LIBS 控制默认的库编译方式。

- 注：如果未进行设置,使用ADD_LIBRARY时又没有指定库类型,默认编译生成的库都是静态库。

CMAKE_C_FLAGS 设置C编译选项

CMAKE_CXX_FLAGS 设置C++编译选项

\#定义预编译宏：TEST

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTEST" )

set(CMAKE_Cxx_FLAGS "${CMAKE_Cxx_FLAGS} -DTEST" ) 

add_compile_definitions(TEST HAVE_CUDA)

## 示例

```cmake
add_library(archive archive.cpp zip.cpp lzma.cpp)
add_executable(zipapp zipapp.cpp)
target_link_libraries(zipapp archive)
```



## 参考

https://www.jianshu.com/p/9d246e4071d4

https://www.cnblogs.com/lx17746071609/p/11436242.html