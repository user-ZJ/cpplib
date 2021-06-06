# 调试方法

调试工具gdb，DDD（gdb的GUI工具）



## gdb

Ctrl+P和Ctrl+N组合键可以浏览以前的命令，并在需要时编辑这些命令。只需要单击ENTER键即可重复发出上一个命令（在重复执行next命令一次一行地单步调试代码时，这种方法非常有用）

define命令允许用户定义缩写和宏



**gdb的TUI模式**

GDB以名为TUI(Terminal User Interface,终端用户界面)的模式提供了基于文本交互和图形用户交互的折中方法。在这一模式中，GDB将终端屏幕划分为类似于DDD的源文本窗口和控制台的多个子窗口；可以在类似于源文本窗口的子窗口中跟踪程序执行的进展过程，同时在类似于控制台的子窗口中发出GDB命令。

在调用GDB时在命令行上加上-tui选项，或者处于非TUI模式时在GDB中使用Ctrl+X+A组合键。如果当前处于TUI模式，则离开TUI模式。



**调试死循环问题**

1. gdb启动调试程序
2. run开始执行程序
3. 使用ctrl+c挂起，gdb会显示停止的位置
4. continue命令继续运行
5. 使用ctrl+c挂起，gdb会显示停止的位置
6. 重复4，5步，即可确定死循环代码的区间

## tips

gdb调试过程中不需要退出gdb，重新编译代码后，回到gdb窗口再次运行run，gdb后重新加载新编译的程序；这样有几点好处：

1. 不需要重新输入命令行参数
2. gdb保留了设置的断点，不需要再次设置

### run(r)

运行调试程序

run xxx  xxx为命令行参数

run    如果调试程序已经在运行，再次使用run命令重新运行调试，如果重用老的命令行参数，则不需要在加命令行参数，gdb会询问是否重新运行。

### break(b)

设置断点

* break line_number  在文件的第xxx行设置断点
* break filename:line_number  再文件filename的line_number处设置断点，例如break source/bed.c:35
* break function,如break main  在main()函数的第一行设置断点
* break filename:function  在文件filename中的函数function()的入口处设置断点
* break 30 if num_y==1  break和condition的组合命令，在第30行设置断点，只有当num_y==1时断点才生效
* break +offset或break -offset，在当前选中栈帧中正在执行的源代码前面或后面设置断点偏移行数
* break *address  在虚拟内存地址处设置断点，这对于程序没有调试信息的部分是必需的

#### tbreak 临时断点

临时断点是首次到达后就会被自动删除的断点，使用tbreak设置，用法和break用法一致

#### 条件断点

break break-args if(condition)

条件中断适用于对于索引变量特定值处处理问题的**循环结构**

break main if argc>1   如果用户想程序中键入了一些命令行参数，则在main处中断

break if  (i==70000)    当循环中i==70000时中断

使用cond命令使普通断点转变为条件断点

cond 3 i==3   给断点3加上i==3的条件

删除条件但保持该断点：

cond 3

#### 断点命令列表

让GDB每次到达某个断点时自动执行一组命令，从而达到自动查看某些变量的目的

```shell
commands breakpoint-number
...
commands
...
end
```

breakpoint-number是要将命令添加到其上的断点的标识符。

commands是用新行分割的任何有效GDB命令列表

```shell
commands 1
silent
printf "fibonacci was passed %d.\n",n
continue
end
```

可以动态的修改给定断点的命令集，或者简单的通过重新定义一个空集来取消该命令集合

```shell
commands 1
end
```





### condition（cond）

conditon 1 num_y==1  使得1号断点只有在num_y==1才会暂停程序执行；使用info break查询断点编号



### watch

监视点，一种特殊类型的断点，当某个表达式的值改变了，就暂停执行

```shell
watch i
watch (i|j>12) && i>24 && strlen(name)>6
```

一旦监视点的变量不存在与当前**作用域，**GDB就会自动删除监视点



###  clear

删除断点

* clear 30 删除第30行的断点
* clear function
* clear filename:function
* clear linenumber
* clear filename:linenumber



### delete

根据断点编号删除断点

* delete 1 3 4 删除断点、监视点、捕获点 1,3,4

* delete 删除所有断点

### disable/enable

禁用/启用断点：保留断点以便以后使用，暂时又不希望gdb停止执行，可以禁用他们。

* disable breakpoint-list
* enable breakpoint-list
* disable 禁用所有断点
* enable 启用所有断点
* enable once breakpoint-list 在断点下次引起GDB暂停执行后被禁用。与tbreak类似，但tbreak是删除断点。

### next(n)

单步调试，遇到函数进入

### step(s)

单步调试，遇到函数进入

### continue(c)

* continue 恢复运行，知道遇到下一个断点。
* continue 3 恢复运行，并忽略接下来3个断点

### finish(fin)

恢复执行，直到恰好在当前栈帧完成之后为止，也就是说，这意味着如果不是在main()函数中，finish命令会导致GDB恢复执行，直到恰好在函数返回之后为止。

对于递归函数，finish只会将你带到递归的上一层。

### until（u）

* until 执行循环（while/for）的其余部分，让GDB在循环后面的第一行代码处暂停
* until 17 执行到17行
* until swap 执行到swap()函数入后暂停
* until swapflaw.c:17 执行到swap.cc文件的17行
* until swapflaw.c:swap 执行到swap.cc文件的swap()函数的入后处。

### print(p)

* 查看对应变量

  print i

* 查看动态数字解决方案,print *pointer@number_of_elements

```cpp
int *x;
int main(){
    x = (int *)malloc(25*sizeof(int));
    x[3] = 12;
    return 0;
}
```

```shell
break main
n
n
p *x@25
```

* 以不同数据格式显示

  p/x y 以十六进制显示变量

  p/c  y 以字符显示

  p/s  y 以字符串显示

  p/f  y  以浮点显示

### display(disp)

用法和print类似，只不过GDB在执行每次有暂停（断点，next，step等命令）时输出指定条目

* dis disp 1 禁用某个显示项
* enable disp 1  启用某个显示项
* undisp 1 删除显示项
* info disp  显示显示项

### call

调用程序中的函数

```shell
commands 2
printf "************current tree**********\n"
call printtree(root)
end
```

### ptype

快速浏览类或结构体的结构

### backtrace(bt)

打印当前调试环境中所有栈帧的信息

backtrace [-full] [n]

- n：一个整数值，当为正整数时，表示打印最里层的 n 个栈帧的信息；n 为负整数时，那么表示打印最外层 n 个栈帧的信息；
- -full：打印栈帧信息的同时，打印出局部变量的值。

### frame(f)

查看栈帧

* frame id	通过栈帧的编号指定。0 为当前被调用函数对应的栈帧号，最大编号的栈帧对应的函数通常就是 main() 主函数
* frame addr	借助栈帧的地址指定。栈帧地址可以通过 info frame 命令（后续会讲）打印出的信息中看到
* frame func	通过函数的函数名指定。注意，如果是类似递归函数，其对应多个栈帧的话，通过此方法指定的是编号最小的那个栈帧
* up n    在当前栈帧编号（假设为 m）的基础上，选定 m+n 为编号的栈帧作为新的当前栈帧
* down n    当前栈帧编号（假设为 m）的基础上，选定 m-n 为编号的栈帧作为新的当前栈帧

#### info frame

该命令会依次打印出当前栈帧的如下信息：

- 当前栈帧的编号，以及栈帧的地址；
- 当前栈帧对应函数的存储地址，以及该函数被调用时的代码存储的地址
- 当前函数的调用者，对应的栈帧的地址；
- 编写此栈帧所用的编程语言；
- 函数参数的存储地址以及值；
- 函数中局部变量的存储地址；
- 栈帧中存储的寄存器变量，例如指令寄存器（64位环境中用 rip 表示，32为环境中用 eip 表示）、堆栈基指针寄存器（64位环境用 rbp 表示，32位环境用 ebp 表示）等。

#### info args

查看当前函数各个参数的值

#### info locals

查看当前函数中各局部变量的值

### info(i)

* info breakpoints 查询断点编号，获取断点的清单以及它们的属性。

属性有：

1. 标识符（Num）：断点的唯一标识符

2. 类型（Type）：这个字段指出该断点是断点、监视点还是捕获点

3. 部署（Disp）：

   保持（keep），下次到达断点后不改变断点

   删除（del），下次到达断点后删除该断点，tbreak创建的断点

   禁用（dis），下次到达断点后禁用该断点，enable once设置的断点

4. 启用状态（Enb）：这个字段说明断点当前是启用还是禁用

5. 地址（Address）：断点在内存中的位置

6. 位置（What）：断点位于源代码中的行。

* info locals 查看当前栈帧中所有局部变量的值列表

* info args    查看当前函数各个参数的值

* info disp 查看display的id

* info frame

  依次打印出当前栈帧的如下信息：

  1. 当前栈帧的编号，以及栈帧的地址；

  2. 当前栈帧对应函数的存储地址，以及该函数被调用时的代码存储的地址

  3. 当前函数的调用者，对应的栈帧的地址；

  4. 编写此栈帧所用的编程语言；

  5. 函数参数的存储地址以及值；

  6. 函数中局部变量的存储地址；

  7. 栈帧中存储的寄存器变量，例如指令寄存器（64位环境中用 rip 表示，32为环境中用 eip 表示）、堆栈基指针寄存器（64位环境用 rbp 表示，32位环境用 ebp 表示）等。

### list

list查看源代码，同时还会切换GDB的焦点

list function 查看function的源代码

## 多线程调试

info threads  给出当前所有线程的信息，前面*表示当前所在的线程

thread 3  切换到线程3

break  88  thread 3   给线程3在第88行设置断点

break  88  thread 3 if x==y  给线程3在第88行设置条件断点



## strace和ltrace

strace ./a.out

strace跟踪系统调用

ltrace跟踪库调用

当没有源码，且没有明确报错信息时尝试使用调试。





# gdb调试C++程序

参数列表

| 命令           | 命令缩写 | 命令说明                                                     |
| -------------- | -------- | ------------------------------------------------------------ |
| list           | l        | 显示多行源代码                                               |
| break          | b        | 设置断点,程序运行到断点的位置会停下来                        |
| info           | i        | 描述程序的状态                                               |
| run            | r        | 开始运行程序                                                 |
| display        | disp     | 跟踪查看某个变量,每次停下来都显示它的值                      |
| step           | s        | 执行下一条语句,如果该语句为函数调用,则**进入函数**执行其中的第一条语句 |
| next           | n        | 执行下一条语句,如果该语句为函数调用,**不会进入函数内部执行**(即不会一步步地调试函数内部语句) |
| print          | p        | 打印内部变量值                                               |
| continue       | c        | 继续程序的运行,直到遇到下一个断点                            |
| set var name=v |          | 设置变量的值                                                 |
| start          | st       | 开始执行程序,在main函数的第一条语句前面停下来                |
| file           |          | 装入需要调试的程序                                           |
| kill           | k        | 终止正在调试的程序                                           |
| watch          |          | 监视变量值的变化                                             |
| backtrace      | bt       | 查看函数调用信息(堆栈)                                       |
| frame          | f        | 查看栈帧                                                     |
| quit           | q        | 退出GDB环境                                                  |

## gdb调试示例程序

```cpp
#include <stdio.h>
void debug(char *str)
{
    printf("debug info :%s\n",str );
}
main(int argc,char *argv[]){
    int i,j;
    j=0;
    printf("param1=%s\n", argv[1]);
    for(i=0;i<10;i++){
        j+=5;
        printf("now a=%d\n", j);
    }
}
```

```shell
gcc -g -o test test.c
```

**使用gdb调试，编译需要-g选项编译**

在Linux下，我们可以使用“strip”命令来去掉ELF文件中的调试信息：

## 启动gdb调试

启动gdb调试有两种方式

1. gdb 执行程序
2. 先输入gdb进入gdb程序，再输入file 执行程序

```shell
# 方法1
gdb test
# 方法2
gdb
file test
```

## 运行gdb调试程序

### run/r

使用run或者r命令开始程序的执行,也可以使用 run parameter将参数传递给该程序

```shell
gdb test
(gdb) r mytest
Starting program: /root/test mytest
param1=mytest
now a=5
now a=10
now a=15
now a=20
```

### list

list命令显示多行源代码,从上次的位置开始显示,默认情况下,一次显示10行,第一次使用时,从代码其实位置显示

list n显示已第n行未中心的10行代码

list functionname显示以functionname的函数为中心的10行代码

```python
(gdb) list main
1       #include <stdio.h>
2       void debug(char *str)
3       {
4           printf("debug info :%s\n",str );
5       }
6       main(int argc,char *argv[]){
7           int i,j;
8           j=0;
9           printf("param1=%s\n", argv[1]);
10          for(i=0;i<10;i++){
```

### break

break location:在location位置设置断点,改位置可以为某一行,某函数名或者其它结构的地址
GDB会在执行该位置的代码之前停下来

```shell
break 10 #在第10行打断点
```

delete breakpoints 断点号                        删除断点

disable/enable n               表示使得编号为n的断点暂时失效或有效

### info

查看信息

info breakpoints   **查看断点相关的信息**

### display

查看参数的值    

如：display j

```shell
(gdb) display j
1: j = 0
(gdb) n
11              j+=5;
1: j = 0
(gdb) display j
2: j = 0
(gdb) n
12              printf("now a=%d\n", j);
2: j = 5
1: j = 5
(gdb) n
now a=5
10          for(i=0;i<10;i++){
2: j = 5
1: j = 5
```

### step和next

step执行下一条语句,如果该语句为函数调用,则**进入函数**执行其中的第一条语句

next执行下一条语句,如果该语句为函数调用,**不会进入函数内部执行**

### watch

watch可设置观察点(watchpoint)。使用观察点可以使得当某表达式的值发生变化时,程序暂停执行

```shell
(gdb) b main
Breakpoint 1 at 0x40053b: file test.c, line 8.
(gdb) r
Starting program: /root/test

Breakpoint 1, main (argc=1, argv=0x7ffdcf012448) at test.c:8
8           j=0;
(gdb) watch j
Hardware watchpoint 2: j
(gdb) c
Continuing.
param1=(null)
Hardware watchpoint 2: j

Old value = 0
New value = 5
main (argc=1, argv=0x7ffdcf012448) at test.c:12
12              printf("now a=%d\n", j);
(gdb) c
Continuing.
now a=5
Hardware watchpoint 2: j

Old value = 5
New value = 10
main (argc=1, argv=0x7ffdcf012448) at test.c:12
12              printf("now a=%d\n", j);
```

### print

```shell
(gdb) b 12
Breakpoint 1 at 0x40056c: file test.c, line 12.
(gdb) r
Starting program: /root/test
param1=(null)

Breakpoint 1, main (argc=1, argv=0x7ffc77f89fc8) at test.c:12
12              printf("now a=%d\n", j);
(gdb) p j
$1 = 5
(gdb) c
Continuing.
now a=5

Breakpoint 1, main (argc=1, argv=0x7ffc77f89fc8) at test.c:12
12              printf("now a=%d\n", j);
(gdb) p i,j
$2 = 10
(gdb) p j
$3 = 10
(gdb) p i
$4 = 1
```

### backtrace

可使用frame 查看堆栈中某一帧的信息



# gdb调试coredump文件

core文件会包含了程序运行时的内存，寄存器状态，堆栈指针，内存管理信息还有各种函数调用堆栈信息等，我们可以理解为是程序工作当前状态存储生成的一个文件，许多的程序出错的时候都会产生一个core文件，通过工具分析这个文件，我们可以定位到程序异常退出的时候对应的堆栈调用等信息，找出问题所在并进行及时解决

我们通过修改kernel的参数，可以指定内核所生成的coredump文件的文件名。例如，使用下面的命令使kernel生成名字为*core.filename.pid*格式的core dump文件：

echo “/data/coredump/core.%e.%p” >/proc/sys/kernel/core_pattern

这样配置后，产生的core文件中将带有崩溃的程序名、以及它的进程ID。上面的%e和%p会被替换成程序文件名以及进程ID。

需要说明的是，在内核中还有一个与coredump相关的设置，就是/proc/sys/kernel/core_uses_pid。如果这个文件的内容被配置成1，那么即使core_pattern中没有设置%p，最后生成的core dump文件名仍会加上进程ID

**如何判断一个文件是coredump文件？**

在类unix系统下，coredump文件本身主要的格式也是ELF格式，因此，我们可以通过readelf命令进行判断。

readelf -h core 可以看到ELF文件头的Type字段的类型是：CORE (Core file)

file core   也可以看到core file

## 生成core文件

ulimit -c unlimited

echo /data/coredump/core.%e.%p> /proc/sys/kernel/core_pattern 将使程序崩溃时生成的coredump文件位于/data/coredump/目录下

## coredump产生的几种可能情况

1. 内存访问越界
2. 多线程程序使用了线程不安全的函数
3. 多线程读写的数据未加锁保护
4. 非法指针
5. 堆栈溢出

## 利用gdb进行coredump的定位

 gdb  程序名(包含路径)   core*(core文件名和路径）

**查看堆栈使用bt或者where命令**

没有调试信息的情况下，打开coredump堆栈，并不会直接显示core的代码行。

此时，frame addr(帧数)或者简写如上，f 1 跳转到core堆栈的第1帧。因为第0帧是libc的代码，已经不是我们自己代码了。

disassemble打开该帧函数的反汇编代码。箭头位置表示coredump时该函数调用所在的位置

shell echo free@plt |c++filt 去掉函数的名词修饰

## coredump示例1

```c
#include "stdio.h"
#include "stdlib.h"

void dumpCrash()
{
    char *pStr = "test_content";
    free(pStr);
}
int main()
{
    dumpCrash();
    return 0;
}
```

```shell
gcc -o dumptest1 -g dumptest1.c  #-g添加调试信息
```

```shell
gdb /root/dumptest1 core
(gdb) bt
#0  0x00007f5321c701d8 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
#1  0x00007f5321c715e0 in __GI_abort () at abort.c:89
#2  0x00007f5321cabbcc in __libc_message (do_abort=do_abort@entry=2, fmt=fmt@entry=0x7f5321da5a10 "*** Error in `%s': %s: 0x%s ***\n")
    at ../sysdeps/posix/libc_fatal.c:175
#3  0x00007f5321cb1763 in malloc_printerr (action=<optimized out>, str=0x7f5321da5a38 "munmap_chunk(): invalid pointer", ptr=<optimized out>,
    ar_ptr=<optimized out>) at malloc.c:5007
#4  0x0000000000400523 in dumpCrash () at dumptest1.c:7
#5  0x0000000000400534 in main () at dumptest1.c:11
(gdb) where
#0  0x00007f5321c701d8 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
#1  0x00007f5321c715e0 in __GI_abort () at abort.c:89
#2  0x00007f5321cabbcc in __libc_message (do_abort=do_abort@entry=2, fmt=fmt@entry=0x7f5321da5a10 "*** Error in `%s': %s: 0x%s ***\n")
    at ../sysdeps/posix/libc_fatal.c:175
#3  0x00007f5321cb1763 in malloc_printerr (action=<optimized out>, str=0x7f5321da5a38 "munmap_chunk(): invalid pointer", ptr=<optimized out>,
    ar_ptr=<optimized out>) at malloc.c:5007
#4  0x0000000000400523 in dumpCrash () at dumptest1.c:7
#5  0x0000000000400534 in main () at dumptest1.c:11
```

```shell
gcc -o dumptest1 dumptest1.c  #没有-g
# 没有调试信息的情况下，打开coredump堆栈，并不会直接显示core的代码行
```

```shell
gdb /root/dumptest1 core
(gdb) bt
#0  0x00007f2bca3121d8 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
#1  0x00007f2bca3135e0 in __GI_abort () at abort.c:89
#2  0x00007f2bca34dbcc in __libc_message (do_abort=do_abort@entry=2, fmt=fmt@entry=0x7f2bca447a10 "*** Error in `%s': %s: 0x%s ***\n")
    at ../sysdeps/posix/libc_fatal.c:175
#3  0x00007f2bca353763 in malloc_printerr (action=<optimized out>, str=0x7f2bca447a38 "munmap_chunk(): invalid pointer", ptr=<optimized out>,
    ar_ptr=<optimized out>) at malloc.c:5007
#4  0x0000000000400523 in dumpCrash ()
#5  0x0000000000400534 in main ()
(gdb) f 4  # f 4 跳转到core堆栈的第1帧
#4  0x0000000000400523 in dumpCrash ()
# disassemble打开该帧函数的反汇编代码
(gdb) disassemble
Dump of assembler code for function dumpCrash:
   0x0000000000400507 <+0>:     push   %rbp
   0x0000000000400508 <+1>:     mov    %rsp,%rbp
   0x000000000040050b <+4>:     sub    $0x10,%rsp
   0x000000000040050f <+8>:     movq   $0x4005c4,-0x8(%rbp)
   0x0000000000400517 <+16>:    mov    -0x8(%rbp),%rax
   0x000000000040051b <+20>:    mov    %rax,%rdi
   0x000000000040051e <+23>:    callq  0x400400 <free@plt>
=> 0x0000000000400523 <+28>:    nop
   0x0000000000400524 <+29>:    leaveq
   0x0000000000400525 <+30>:    retq
End of assembler dump.
(gdb) shell echo free@plt |c++filt
free@plt
```

上面的free使用去掉名词修饰效果和之前还是一样的。但是我们可以推测到这里是在调用free函数

## coredump示例2--寻找this指针和虚指针

```c++
#include "stdio.h"
#include <iostream>
#include "stdlib.h"
using namespace std;
class base
{
public:
    base();
    virtual void test();
private:
    char *basePStr;
};
class dumpTest : public base
{
public:
    void test();
private:
    char *childPStr;
};
base::base()
{
    basePStr = "test_info";
}
void base::test()
{
    cout<<basePStr<<endl;
}
void dumpTest::test()
{
    cout<<"dumpTest"<<endl;
    delete childPStr;
}
void dumpCrash()
{
    char *pStr = "test_content";
    free(pStr);
}
int main()
{
    dumpTest dump;
    dump.test();
    return 0;
}
```

```shell
g++ -o dumptest2 dumptest2.cpp
```

```shell
gdb /root/dumptest2 core
(gdb) bt
#0  0x00007f5520246f64 in __GI___libc_free (mem=0x7ffed1ccae20) at malloc.c:2968
#1  0x00000000004009cb in dumpTest::test() ()
#2  0x0000000000400a0d in main ()
(gdb) f 1
#1  0x00000000004009cb in dumpTest::test() ()
(gdb) info frame
Stack level 1, frame at 0x7ffed1ccad20:
 rip = 0x4009cb in dumpTest::test(); saved rip 0x400a0d
 called by frame at 0x7ffed1ccad50, caller of frame at 0x7ffed1ccad00
 Arglist at 0x7ffed1ccad10, args:
 Locals at 0x7ffed1ccad10, Previous frame's sp is 0x7ffed1ccad20
 Saved registers:
  rbp at 0x7ffed1ccad10, rip at 0x7ffed1ccad18
```

Previous frame's sp is 0x7ffed1ccad20表示前一帧的栈寄存器地址是0x7ffed1ccad20

它的前一帧也就是main函数里调用dump.test()的位置，那我们在这个地址上应该可以找到dump的this指针和它的虚指针，以及虚指针指向的虚函数表

```shell 
(gdb) x 0x7ffed1ccad20
0x7ffed1ccad20: 0x00400b48
(gdb) x 0x00400b48
0x400b48 <_ZTV8dumpTest+16>:    0x0040098e
(gdb) shell echo _ZTV8dumpTest|c++filt
vtable for dumpTest
(gdb) x 0x0040098e
0x40098e <_ZN8dumpTest4testEv>: 0xe5894855
(gdb) shell echo _ZN8dumpTest4testEv|c++filt
dumpTest::test()
(gdb) x 0x0040098e-4
0x40098a <_ZN4base4testEv+46>:  0x90c3c990
(gdb) shell echo _ZN4base4testEv|c++filt
base::test()
```

0x7ffed1ccad20地址指向的是前一帧保存dump信息的位置，0x00400b48就表示dump的this指针，通过x 0x00400b48看到_ZTV8dumpTest+16的内容。

shell echo_ZTV8dumpTest|c++filt 可以看到“vtable for dumpTest”的内容。这个就表示dumpTest的虚函数表

通过x 0x0040098e可以看到，存储的内容就是dumpTest::test()

如上，在实际问题中，C++程序的很多coredump问题都是和指针相关的，很多segmentfault都是由于指针被误删或者访问空指针、或者越界等造成的，而这些都一般意味着正在访问的对象的this指针可能已经被破坏了，此时，我们通过去寻找函数对应的对象的this指针、虚指针能验证我们的推测。之后再结合代码寻找问题所在。

## coredump示例3--所有线程堆栈

```cpp
#include <iostream>
#include <pthread.h>
#include <unistd.h>
using namespace std;
#define NUM_THREADS 5 //线程数
int count = 0;

void* say_hello( void *args )
{
    while(1)
    {
        sleep(1);
        cout<<"hello..."<<endl;
        if(NUM_THREADS ==  count)
        {
            char *pStr = "";
            delete pStr;
        }
    }
} //函数返回的是函数指针，便于后面作为参数
int main()
{
    pthread_t tids[NUM_THREADS]; //线程id
    for( int i = 0; i < NUM_THREADS; ++i )
    {
        count = i+1;
        int ret = pthread_create( &tids[i], NULL, say_hello,NULL); //参数：创建的线程id，线程参数，线程运行函数的起始地址，运行函数的参数
        if( ret != 0 ) //创建线程成功返回0
        {
            cout << "pthread_create error:error_code=" << ret << endl;
        }
    }
    pthread_exit( NULL ); //等待各个线程退出后，进程才结束，否则进程强制结束，线程处于未终止的状态
}
```

```shell
g++ -o dumptest3 dumptest3.cpp -lpthread
```

info threads查看所有线程正在运行的指令信息

thread apply all bt打开所有线程的堆栈信息

threadapply thread ID bt 查看指定线程堆栈信息

thread thread ID	进入指定线程栈空间

# objdump

-a, --archive-headers：显示archive头信息

-f, --file-headers：显示elf文件头信息

-p, --private-headers：显示对象格式的特定文件头内容

**-h： 显示各个段的头信息**

-x, --all-headers：显示所有头信息

**-d, --disassemble：显示可执行段的汇编器内容**

-D, --disassemble-all：显示所有段的汇编器内容

-S, --source：混合源代码与反汇编

**-s, --full-contents：显示所有段的内容**

-g, --debugging：显示elf文件的调试信息

-e, --debugging-tags：使用ctags风格显示调试信息

-G, --stabs：显示STABS信息

-W：显示DWARF信息

**-t, --syms：显示符号表**

-T, --dynamic-syms：显示动态符号表

-r, --reloc：显示文件中的重定位条目

-R, --dynamic-reloc：在文件中显示动态重定位条目

-v, --version：显示objdump版本

-i, --info：列出支持的对象格式和体系结构

# readelf

-a --all               Equivalent to: -h -l -S -s -r -d -V -A -I

-h --file-header       显示elf文件头信息

**-l --program-headers   显示 program headers**
     --segments          --program-headers的别名

-S --section-headers   显示sections' header
     --sections          --section-headers的别名

-g --section-groups    显示 section groups

-t --section-details   显示 section details

-e --headers           Equivalent to: -h -l -S

**-s --syms              显示符号表**
     --symbols           An alias for --syms

--dyn-syms             显示动态符号表

-n --notes             Display the core notes (if present)

**-r --relocs            显示重定位信息(如果有)**

-u --unwind            显示展开信息 (if present)

-d --dynamic           显示动态段 (if present)

-V --version-info      显示版本段 (if present)

-A --arch-specific     显示特定于体系结构的信息（如果有）

-c --archive-index     在档案中显示符号/文件索引

**-D --use-dynamic       显示符号时使用动态部分信息**

-x --hex-dump=<number|name>    将<number | name>节的内容作为字节转储

-p --string-dump=<number|name>    将<number | name>节的内容作为字符串转储

-R --relocated-dump=<number|name>    转储<number | name>节的内容作为重定位字节

-z --decompress        转储节之前解压缩节

--dwarf-depth=N        不要显示深度N或更大的DIE

--dwarf-start=N        以相同深度或更深显示以N开头的DIE

-I --histogram         显示存储段列表长度的直方图

-W --wide              允许输出宽度超过80个字符



readelf -h xxx.o  查看elf文件头



nm

c++filt 的工具可以用来解析C++被修饰过的名称



兼容C语言和C++语言定义两套头文件

```cpp
#ifdef __cplusplus
extern "C" {
#endif
	void *memset (void *, int, size_t);
#ifdef __cplusplus
}
#endif
```



gcc -E hello.c -o hello.i  //预编译

gcc -S hello.c -o hello.s  //编译

gcc -c hello.c -o hello.o //汇编

符号修饰标准、变量内存布局、函数调用方式等这些跟可执行代码二进制兼容性相关的内容称为ABI（Application Binary Interface）

**-fPIC**：GCC产生地址无关代码，Position-independent Code

-fPIE：地址无关方式编译的可执行文件，Position-Independent Executable





# 其他

动态链接器会按照下列顺序依次装载或查找共享对象（目标文件）：

由环境变量LD_LIBRARY_PATH指定的路径。

由路径缓存文件/etc/ld.so.cache指定的路径。

默认共享库目录，先/usr/lib，然后/lib。

LD_LIBRARY_PATH也会影响GCC编译时查找库的路径，它里面包含的目录相当于链接时GCC的“-L”参数

“**-rpath**”选项（或者GCC的-Wl,-rpath），这种方法可以指定链接产生的目标程序的共享库查找路径。

LD_PRELOAD，这个文件中我们可以指定预先装载的一些共享库甚或是目标文件。在LD_PRELOAD里面指定的文件会在动态链接器按照固定规则搜索共享库之前装载，它比LD_LIBRARY_PATH里面所指定的目录中的共享库还要优先。无论程序是否依赖于它们，LD_PRELOAD里面指定的共享库或目标文件都会被装载。

系统配置文件中有一个文件是/etc/ld.so.preload，它的作用与LD_PRELOAD一样。这个文件里面记录的共享库或目标文件的效果跟
LD_PRELOAD里面指定的一样，也会被提前装载

**-export-dynamic**

有一种情况是，当程序使用dlopen()动态加载某个共享模块，而该共享模块须反向引用主模块的符号时，有可能主模块的某些符号因为在链接时没有被其他共享模块引用而没有被放到动态符号表里面，导致了反向引用失败。ld链接器提供了一个 “-export-dynamic ”的参数，这个参数表示链接器在生产可执行文件时，将所有全局符号导出到动态符号表，以防止出现上述问题。我们也可以在GCC中使用 “-Wl,-export-dynamic ”将该参数传递给链接器。



“strip”的工具清除掉共享库或可执行文件的所有符号和调试信息。除了使用“**strip**”工具，我们还可以使用ld的“**-s**”和“-S”参数，使得链接器生成输出文件时就不产生符号信息。“-s”和“-S”的区别是：“-S”消除调试符号信息，而“-s”消除所有符号信息。我们也可以在gcc中通过“-Wl,-s”和“-Wl,-S”给ld传递这两个参数

ldconfig –n shared_library_directory：建立相应的SO-NAME软链接

GCC提供了两个参数**“-L”和“-l”**，分别用于指定共享库搜索目录和共享库的路径。当然也可以使用前面提到过的“-rpath”参数



很多时候你希望共享库在被装载时能够进行一些初始化工作，比如打开文件、网络连接等，使得共享库里面的函数接口能够正常工作。GCC提供了一种共享库的构造函数，只要在函数声明时加上“__attribute__((constructor)) ”的属性，即指定该函数为共享库构造函数，拥有这种属性的函数会在共享库加载时被执行，即在程序的main函数之前执行。如果我们使用 dlopen() 打开共享库，共享库构造函数会在 dlopen() 返回之前被执行。

与共享库构造函数相对应的是析构函数，我们可以使用在函数声明时加上“__ attribute__((destructor))” 的属性，这种函数会在 main() 函数执行完毕之后执行（或者是程序调用 exit() 时执行）。如果共享库是运行时加载的，那么我们使用 dlclose() 来卸载共享库时，析构函数将会在 dlclose() 返回之前执行。

void __attribute__((constructor)) init_function(void);

void __attribute__((destructor)) fini_function (void);





# 参考

https://blog.csdn.net/zdy0_2004/article/details/80102076    

https://blog.csdn.net/qq_39759656/article/details/82858101

