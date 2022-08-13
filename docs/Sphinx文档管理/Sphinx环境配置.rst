===========================
Sphinx环境配置
===========================

安装
==============

.. code:: shell

    # Ubuntu
    apt-get install python3-sphinx
    #conda 
    conda install sphinx
    # pip
    pip install -U sphinx


初始化文档源目录
=========================

在文件目录执行: sphinx-quickstart

按照如下配置：

.. code:: shell
    
    Project name: cppdoc
    Author name(s): zack
    Project release []: 0.1.0
    Project language [en]: zh_CN

执行后需要对配置做选择, 可以全部默认, 后续可以在生成的conf.py做修改

执行后生成一份主页文档index.rst, 可以通过index.rst对其他文件做索引

配置支持markdown
=====================

参考： https://www.sphinx-doc.org/en/master/usage/markdown.html

1. 安装markdown解析器myst-parser
   
   .. code:: shell
    
    pip install --upgrade myst-parser

2. 在配置文件conf.py中增加myst-parser的扩展
   
   .. code:: shell

    extensions = ['myst_parser']
    # 注意：MyST-Parser requires Sphinx 2.1 or newer.

3. 在配置文件conf.py中配置使用myst_parser自动解析.md后缀的文件
   
   .. code:: shell

    source_suffix = {
       '.rst': 'restructuredtext',
       '.txt': 'restructuredtext',
       '.md': 'markdown',
    }

配置html主题
====================

主题列表：https://sphinx-themes.org/

Read the Docs主题
---------------------

1. 如果要使用https://readthedocs.org/进行文档托管，使用https://sphinx-themes.org/sample-sites/sphinx-rtd-theme/主题
   
   pip install sphinx-rtd-theme

2. 在配置文件中添加
   
   html_theme = 'sphinx_rtd_theme'

3. 重新编译html

book主题
--------------------

1. 平时自己阅读，使用book主题比较友好

   pip install sphinx-book-theme

2. 在配置文件中添加

   html_theme = 'sphinx_book_theme'

3. 重新编译html

生成文档
================
在Makefile目录下执行: make html

