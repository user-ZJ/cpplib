# linux常用命令

## Linux递归比较文件夹差异
```shell
diff -Nrq a b  
```

## windows文件格式为unix  
```shell
sed -i 's/\r//' <filename>
```

## 替换文件中的内容
```shell
sed -i 's/oo/kk/g' ./test*
```

## pigz 多线程压缩

**分包压缩**:

使用128线程对文件夹进行压缩，并将压缩有的文件分割为10G大小
```shell
tar -c /path/to/dir | pigz -p 128 -c | split -a 5 -d -b 10G - file_split.gz
```

**解压**：

将分割的文件还原回来，并用pigz进行解压：
```shell
cat file_split.gz* > file_split.gz
pigz -p 128 -dc  file_split.gz | tar xf -
```

## scp在命令行中带密码远程拷贝文件
```shell
sshpass -p 密码 scp -P 端口 源文件 目的文件
```

## 加快git下载

```shell
# 默认压缩级别,-1是zlib的默认值。0表示无压缩,而1..9是各种速度/大小的折衷，9表示最慢
git config --global core.compression 0  
git clone --depth 1 http://xxx.git  
```

## 查看文件的xx行到xx行

```shell
# 1. 3000-3999行
cat file | tail -n +3000 | head -n 1000
# 2. 1000-3000行
cat file | head -n 3000 | tail -n +1000
# 3. 5-10行
sed -n '5,10p' file
```

## 限制程序运行的CPU核

``` 
taskset -p pid  #查看程序运行在哪个核
taskset -pc 1 processbin/pid #指定程序运行在cpu1上
taskset -c 0-7 processbin #指定程序在cpu0-cpu7上运行
```

## 挂载

```shell
mount -t nfs xx.xx.xx.xx:/path /targetpath
```

## 跨服务器带命令拷贝

```shell
sshpass -p "passwd" scp file xx.xx.xx.xx@user:/path
```

## 多线程下载工具

```
axel
mwget 
```


## 查看某个进程CPU占用率

```shell
# ps -aux CPU占用率是统计进程启动后的平均CPU占用率
ps -aux | grep process_name | grep -v grep | awk '{print $3}'
# top CPU占用率是上次top刷新到本次top刷新之间的CPU平均占用率
top -cn 1 | grep process_name | awk '{print $9}'
top -n 1 -c -p pid | head -n 8 | tail -n 1 | awk '{print $9}'
```

## 查看某个进程内存占用

```shell
ps -aux | grep process_name | grep -v grep | awk '{print $6/1024}'
```

## script

script命令用于将一个shell会话过程中产生的全部输入和输出保存为文本文件。这个文本文件在将来既可以用来重现被执行的命令，也可以用来查看结果。在调查性能问题时，准确记录被执行命令是很有用的，因为你可以在之后的时间里查看执行过的测试。拥有被执行命令的记录就意味着在调查不同的问题时，你可以简单地对命令进行剪切和粘贴。

```shell
script [-a] [-t] [file]
# -a 向文件添加脚本输出，而不是覆盖文件
# -t 增加了计时信息
# file 输出文件名，没有指定则默认为typescript
# exit或ctrl+d退出
```






