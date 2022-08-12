====================
reStructuredText示例
====================

标题
=========
reStructuredText 中的“标题”被称为“Sections”，一般在文字下方加特殊字符以示区别：

特殊字符的重复长度应该大于等于标题（Sections）的长度。需要说明的是： reStructuredText 并不像 Markdown 那样，限定某一字符只表示特定的标题层级（比如 = 固定表示 H1 )。
而是解析器将遇到的第一个特殊字符渲染为 H1 ，第二个其它特殊字符渲染为 H2 ……以此类推。

::

    Section Title H1
    ================

    Section Title H2
    ----------------

    Section Title H3
    ````````````````

推荐使用的字符： ``= - ` : . ' " ~ ^ _ * + #``

当然，在 reStructuredText 的日常使用中，仍然建议养成习惯使用固定的特殊符号，方便别人一看到 = 就知道这是一级标题。 
除了 “Sections”外， reStructuredText 还支持“Title”和“SubTitle”，它们可以被配置为不在文档中出现。其实际作用更类似于“书名”，如《钢铁是怎样炼成的——保尔柯察金自传》。语法如下：

::

    ==================
    钢铁是怎样炼成的
    ==================

    ----------------
    保尔柯察金自传
    ----------------


区块引用
======================
区块引用使用空格或制表符的方式，一般是 4 个空格。   

    区块引用
    
        嵌套引用


列表
==================
reStructuredText 支持有序列表和无序列表，语法与 Markdown 基本一致 

无序列表使用 ``- 、 * 、 +`` 来表示  

有序列表可以使用： 

1. 阿拉伯数字: 1, 2, 3, … (无上限)
2. 大写字母: A-Z
3. 小写字母: a-z
4. 大写罗马数字: I, II, III, IV, …, MMMMCMXCIX (4999)
5. 小写罗马数字: i, ii, iii, iv, …, mmmmcmxcix (4999)

代码块
=============
::

    .. code:: python

        import sys
        print(sys.version)

.. code:: python

   import sys
   print(sys.version)

数学公式
===========
支持latex数学公式表示，分为行内公式和单独行公式
  
行内公式表示为：``空格+：math:`公式`空格``

圆的面积为 :math:`A_\text{c} = (\pi/4) d^2` .

单行公式表示为：``.. math:: 换行+两个空格+公式``

.. code:: text

  .. math:: 
    \alpha _t(i) = P(O_1, O_2, \ldots  O_t, q_t = S_i \lambda )

.. math::
  \alpha _t(i) = P(O_1, O_2, \ldots  O_t, q_t = S_i \lambda )


分割线
===================
与 Markdown 语法基本一致：
::

-----------------------------------------

效果如下：

---------------------------------------------------------


链接
==============

参考式链接
------------------------
::

    欢迎访问 reStructuredText_ 官方主页。

    .. _reStructuredText: http://docutils.sf.net/

    如果是多个词组或者中文链接文本，则使用 ` 将其括住，就像这样：

    欢迎访问 `reStructuredText 结构化文本`_ 官方主页。

    .. _`reStructuredText 结构化文本`: http://docutils.sf.net/

欢迎访问 reStructuredText_ 官方主页。

.. _reStructuredText: http://docutils.sf.net/

欢迎访问 `reStructuredText 结构化文本`_ 官方主页。

.. _`reStructuredText 结构化文本`: http://docutils.sf.net/

行内式链接
-------------------------------
::

    `Python 编程语言 <http://www.python.org/>`_ 其实也有一些缺陷。

`Python 编程语言 <http://www.python.org/>`_ 其实也有一些缺陷。

自动标题链接
------------------------------------
reStructuredText 文档的各级标题（Sections）会自动生成链接，就像 GFM 风格的 Markdown 标记语言一样。
这在 reStructuredText 语法手册中被称为“隐式链接（Implicit Hyperlink）”。无论名称为何，我们将可以在文档中快速跳转到其它小节（Sections）

::

    本小节内容应该与 `行内标记`_ 结合学习。

本小节内容应该与 `行内标记`_ 结合学习。

强调
====================
与 Markdown 语法基本相同。参看 `行内标记`_  

图片
=====================
reStructuredText 使用指令（Directives)的方式来插入图片。指令（Directives）作为 reStructuredText 语言的一种扩展机制，允许快速添加新的文档结构而无需对底层语法进行更改。

::

    .. image:: /images/nikola.png
        :align: center
        :width: 236px
        :height: 100px

.. image:: /images/nikola.png
   :align: center
   :width: 236px
   :height: 100px

插入图片的另一种方法是使用 figure 指令。该指令与 image 基本一样，不过可以为图片添加标题和说明文字。
两个指令共有的一个选项为 target ，可以为图片添加可点击的链接，甚至链接到另一张图片。那么结合 Nikola 博客的特定主题，就可以实现点击缩略图查看原图的效果

::

    .. figure:: /images/icarus.thumbnail.jpg
        :align: center
        :target: /images/icarus.jpg

        *飞向太阳*

.. figure:: /images/icarus.thumbnail.jpg
   :align: center
   :target: /images/icarus.jpg

   *飞向太阳*

表格
====================
::

    +------------------------+------------+----------+----------+
    | Header row, column 1   | Header 2   | Header 3 | Header 4 |
    | (header rows optional) |            |          |          |
    +========================+============+==========+==========+
    | body row 1, column 1   | column 2   | column 3 | column 4 |
    +------------------------+------------+----------+----------+
    | body row 2             | Cells may span columns.          |
    +------------------------+------------+---------------------+
    | body row 3             | Cells may  | - Table cells       |
    +------------------------+ span rows. | - contain           |
    | body row 4             |            | - body elements.    |
    +------------------------+------------+---------------------+

显示效果为：

+------------------------+------------+----------+----------+
| Header row, column 1   | Header 2   | Header 3 | Header 4 |
| (header rows optional) |            |          |          |
+========================+============+==========+==========+
| body row 1, column 1   | column 2   | column 3 | column 4 |
+------------------------+------------+----------+----------+
| body row 2             | Cells may span columns.          |
+------------------------+------------+---------------------+
| body row 3             | Cells may  | - Table cells       |
+------------------------+ span rows. | - contain           |
| body row 4             |            | - body elements.    |
+------------------------+------------+---------------------+


这种表格语法被称为 Grid Tables 。如上所见， Grid Tables 支持跨行跨列。如果你使用的编辑器创建该表格有困难，reStructuredText 还提供 Simple Tables 表格语法：

::

    =====  =====  ======
    Inputs     Output
    ------------  ------
    A      B    A or B
    =====  =====  ======
    False  False  False
    True   True   True
    =====  =====  ======

显示效果为：

=====  =====  ======
   Inputs     Output
------------  ------
  A      B    A or B
=====  =====  ======
False  False  False
True   True   True
=====  =====  ======

行内标记
===================

+------------------+--------------+----------------------------------------------------+
|       文本       |     结果     |                        说明                        |
+==================+==============+====================================================+
| \*强调\*         | *强调*       | 一般被渲染为斜体                                   |
+------------------+--------------+----------------------------------------------------+
| \*\*着重强调\*\* | **着重强调** | 一般被渲染为加粗                                   |
+------------------+--------------+----------------------------------------------------+
| \`解释文本\`     | `解释文本`   | 一般用于专用名词、文本引用、说明性文字等           |
+------------------+--------------+----------------------------------------------------+
| \`\`原样文本\`\` | ``原样文本`` | 与上面的区别在于：不会被转义。可用于行内代码书写。 |
+------------------+--------------+----------------------------------------------------+


参考
=================
https://macplay.github.io/posts/cong-markdown-dao-restructuredtext/#id10

https://3vshej.cn/rstSyntax/index.html

https://hzz-rst.readthedocs.io/zh_CN/latest/index.html

