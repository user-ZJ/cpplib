# readthedoc文档托管

## 1. sphinx使用

### 1.1 sphinx安装

在ubuntu本地环境安装sphinx: sudo apt install python3-sphinx

在python虚拟环境安装: pip install sphinx 或: conda install sphinx

### 1.2 sphinx创建文档

在文件目录执行: sphinx-quickstart

```shell
> Project name: cppdoc
> Author name(s): zack
> Project release []: 0.1.0
> Project language [en]: zh_CN
```



执行后需要对配置做选择, 可以全部默认, 后续可以在生成的conf.py做修改

执行后生成一份主页文档index.rst, 可以通过index.rst对其他文件做索引

rst标记语言编写参考: https://3vshej.cn/rstSyntax/index.html

### 1.3 配置支持markdown

https://www.sphinx-doc.org/en/master/usage/markdown.html

1. 安装markdown解析器myst-parser

   ```shell
   pip install --upgrade myst-parser
   ```

2. 在配置文件中增加myst-parser的扩展

   ```shell
   extensions = ['myst_parser']
   ```

   注意：MyST-Parser requires Sphinx 2.1 or newer.

3. 配置使用myst_parser自动解析.md后缀的文件

   ```shell
   source_suffix = {
       '.rst': 'restructuredtext',
       '.txt': 'markdown',
       '.md': 'markdown',
   }
   ```

### 1.4 配置目录树

https://zh-sphinx-doc.readthedocs.io/en/latest/markup/toctree.html



### 1.5 配置html主题

主题列表：https://sphinx-themes.org/

**Read the Docs主题**

1. 如果要使用https://readthedocs.org/进行文档托管，使用https://sphinx-themes.org/sample-sites/sphinx-rtd-theme/主题

   ```shell
   pip install sphinx-rtd-theme
   ```

2. 在配置文件中添加

   ```shell
   html_theme = 'sphinx_rtd_theme'
   ```

3. 重新编译html

**book主题**

1. 平时自己阅读，使用book主题比较友好

   ```shell
   pip install sphinx-book-theme
   ```

2. 在配置文件中添加

   ```shell
   html_theme = 'sphinx_book_theme'
   ```

3. 重新编译html



### 1.6 生成文档

在Makefile目录下执行: make html



## RST标记语言使用

rst标记语言编写参考: https://3vshej.cn/rstSyntax/index.html



## 使用readthedocs托管文档

1.把源码上传到github或者gitlab, 不需要上传make html生成的文件, 网站会自动构建, 注意项目要是公开项目

2.在https://readthedocs.org/ 导入github或者gitlab上的项目就可以通过一个url查看到文档了