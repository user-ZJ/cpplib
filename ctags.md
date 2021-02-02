ctags --list-languages

查看支持的语言

ctags --list-maps

查看默认哪些扩展名对应哪些语言

ctags --langmap=c++:+.inl –R

指定ctags用特定语言的分析器来分析某种扩展名的文件或者名字符合特定模式的文件；如ctags --langmap=c++:+.inl –R表示以inl为扩展名的文件是c++文件

ctags --list-kinds

查看ctags可以识别的语法元素

ctags --list-kinds=c++

单独查看可以识别的c++的语法元素

ctags -R --c++-kinds=+px

要求ctags记录c++文件中的函数声明和各种外部和前向声明



ctags -R .

为当前目录及其子目录中的文件生成标签文件

ctrl+]

找到光标所 在位置的标签定义的地方

ctrl+T

回跳到之前的标签处

Ctrl＋W + ］

新窗口显示当前光标下单词的标签，光标跳到标签处

(ex command) :tag startlist

跳到一个函数的定义(如startlist)就可以用下面的命令，这个命令会带你到函数"startlist"的定义处，哪怕它是在另一个文件中。

(ex command):tags

":tags"命令会列出现在你就已经到过哪些tag了

(ex command) :tag

直接跳转到当前tag序列的最后

(ex command) :stag tagname

在一个新窗口中查看tagname的上下文



当一个函数被多次重载(或者几个类里都定义了一些同名的函数)，":tag"命令会跳转到第一个符合条件的。如果当前文件中就有一个匹配的，那又会优先使用它。当然还得有办法跳转到其它符合条件的tag去：

(ex command) :tnext

重复使用这个命令可以发现其余的同名tag。如果实在太多，还可以用下面的命令从中直接选取一个：

(ex command) :tselect tagname

(ex command):tfirst  跳转到第一个匹配

(ex command):[count]tprevious   跳转到前count个匹配

(ex command):[count]tnext      跳转到后count个匹配

(ex command):tlast       跳转到最后一个匹配







ctags -R --c-kinds=+px --c++-kinds=+px --fields=+aiKSz --extra=+q



在.vimrc中增加一行：set tags=tags;/

表示在当前目录找不到tags文件时，在上层目录中查找

或者在vim命令行中设置“:set tags=tags路径”







https://www.cnblogs.com/coolworld/p/5602589.html