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

# 参考

https://blog.csdn.net/zdy0_2004/article/details/80102076    

https://blog.csdn.net/qq_39759656/article/details/82858101