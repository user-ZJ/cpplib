# log4cpp使用笔记



日志类别（Category）、输出源（Appender）和布局（Layout）

Priorty（优先 级）、NDC（嵌套的诊断上下）





- %d 输出日志时间点的日期或时间，可以在其后指定格式，如上%d{%Y-%m-%d %H:%M:%S.%l}，输出类似：2017-02-14 09:25:00.953
- %p 优先级，即DEBUG,INFO,WARN,ERROR,FATAL
- %c 输出日志信息所属的类目，通常就是所在类的全名
- %m 输出log的具体信息
- %n 回车换行
- %x: 输出和当前线程相关联的NDC(嵌套诊断环境),
- %r: 输出自应用启动到输出该log信息耗费的毫秒数



## 参考

https://blog.csdn.net/jigetage/article/details/80624692

https://blog.csdn.net/jenie/article/details/106982667