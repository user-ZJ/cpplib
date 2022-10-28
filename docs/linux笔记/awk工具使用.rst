awk工具使用
==================

求和、最大值、最小值、均值
-------------------------------

.. code-block::shell

    # 求和
    cat data|awk '{sum+=$1} END {print "Sum = ", sum}'
    # 求平均
    cat data|awk '{sum+=$1} END {print "Average = ", sum/NR}'
    # 求最大值
    cat data|awk 'BEGIN {max = 0} {if ($1+0>max+0) max=$1 fi} END {print "Max=", max}'
    # 求最小值
    awk 'BEGIN {min = 65536} {if ($1+0<min+0) min=$1 fi} END {print "Min=", min}'

获取行号
-----------------
.. code-block:: shell

    # 添加行号
    cat data | awk '{print NR" "$0}'
    # 不打印第一行
    cat data | awk 'NR!=1 {print $0}'
    # 打印第二行
    cat data | awk 'NR==2 {print $0}'

倒数第N列
-----------------
$NF表示倒数第一列，$(NF-1)表示倒数第二列
.. code-block:: 

    cat data | awk '{print $(NF),$(NF-1)}'


匹配和不匹配
-------------------------
* ~ 匹配正则
* !~ 不匹配正则
* == 等于
* != 不等于

.. code-block:: shell

    cat data | awk '{if($1!=$2) print $0}'
    cat data | awk '{if($1==$2) print $0}'
    # 第二列匹配80开头并以80结束的行
    awk '{if($2~/^80$/)print}' test.txt
    # 第二列中不匹配80开头并以80结束的行
    awk '{if($2!~/^80$/)print}' test.txt

对文件的某一列进行去重
-------------------------------
.. code-block:: shell

    awk '{a[$n]=$0}END{for(i in a)print a[i]}' filename


内建函数
-----------------

split
`````````````````
split允许你把一个字符串分隔为单词并存储在数组中

.. code-block:: shell

    time="12:34:56"
    out=`echo $time | awk '{split($0,arr,":");for (i in arr) print i,arr[i]}'`
    echo $out


substr
```````````````
返回从起始位置起，指定长度的子字符串；若未指定长度，则返回从起始位置到字符串末尾的子字符串。

* substr(s,p) 返回字符串s中从p开始的后缀部分
* substr(s,p,n) 返回字符串s中从p开始长度为n的后缀部分

.. code-block:: shell

    echo "123" | awk '{print substr($0,1,1)}'


length
```````````````
length函数返回整个记录中的字符数。

.. code-block:: shell

    echo "123" | awk '{print length}'
    cat info.txt | awk -F '"' '{print $2}' | awk '{if (length > 0) print $0}' | sort | uniq > industryList.txt

gsub
```````````
gsub函数使得在所有正则表达式被匹配的时候都发生替换。gsub(regular expression, subsitution string, target string)

.. code-block:: shell

    #把一个文件里面所有包含 abc 的行里面的 abc 替换成 def
    cat abc.txt | awk '{gsub("abc", "def", $0); print $1, $3}'


内置变量
---------------------
* NF:读取记录的字段数(列数)
* NR：读取文件的行数(在某些应用场景中可以当作行号来使用)
* FNR：读取文件的行数，但是和"NR"不同的是当读取的文件有两个或两个以上时，NR读取完一个文件，行数继续增加 而FNR重新从1开始记录
* FS：输入字段分割符，默认是以空格为分隔符，在日常中常常文本里面不都以空格分隔，此时就要指定分割符来格式化输入。
* OFS：输出字段分割符，默认为空格，如果读进来的数据是以空格分割，为了需求可能要求输出是以"-"分割，可以使用OFS进行格式化输出。
* RS：输入行分隔符，判断输入部分的行的起始位置，默认是换行符
* ORS：输出行分割符，默认的是换行符,它的机制和OFS机制一样，对输出格式有要求时，可以进行格式化输出


.. code-block:: shell

    awk 'BEGIN{FS=" ";OFS="--"}{print $1,$2,$3}' test3