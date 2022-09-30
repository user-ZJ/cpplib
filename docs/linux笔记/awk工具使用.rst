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