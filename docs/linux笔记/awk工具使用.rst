==================
awk工具使用
==================

.. code-block::shell

    # 求和
    cat data|awk '{sum+=$1} END {print "Sum = ", sum}'
    # 求平均
    cat data|awk '{sum+=$1} END {print "Average = ", sum/NR}'
    # 求最大值
    cat data|awk 'BEGIN {max = 0} {if ($1+0>max+0) max=$1 fi} END {print "Max=", max}'
    # 求最小值
    awk 'BEGIN {min = 65536} {if ($1+0<min+0) min=$1 fi} END {print "Min=", min}'


