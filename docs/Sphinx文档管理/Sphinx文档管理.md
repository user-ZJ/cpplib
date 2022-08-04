# Sphinx文档管理

官方文档：https://www.sphinx-doc.org/zh_CN/master/contents.html

Sphinx 是一个 *文档生成器* ，您也可以把它看成一种工具，它可以将一组纯文本源文件转换成各种输出格式，并且自动生成交叉引用、索引等。也就是说，如果您的目录包含一堆 [reStructuredText](https://www.sphinx-doc.org/zh_CN/master/usage/restructuredtext/index.html) 或 [Markdown](https://www.sphinx-doc.org/zh_CN/master/usage/markdown.html) 文档，那么 Sphinx 就能生成一系列HTML文件，PDF文件（通过LaTeX），手册页等。

## 安装

```shell
# Ubuntu
apt-get install python3-sphinx
#conda 
conda install sphinx
# pip
pip install -U sphinx
```

## 初始化文档源目录

```shell
sphinx-quickstart
```

设置一个源目录并创建一个默认的 `conf.py` 配置文件，在创建时，它还会问你一些问题，并从中得到配置值填入配置文件中

## readthedoc

https://readthedocs.org/

一般的做法是将文档托管到版本控制系统比如github上面，push源码后自动构建发布到readthedoc上面， 这样既有版本控制好处，又能自动发布到readthedoc

先在GitHub创建一个仓库名字叫xxx， 然后在本地.gitignore文件中添加`build/`目录，初始化git，commit后，添加远程仓库。

具体几个步骤非常简单，参考官方文档：https://github.com/rtfd/readthedocs.org:

1. 在Read the Docs上面注册一个账号
2. 登陆后点击 “Import”.
3. 给该文档项目填写一个名字比如 “xxx”, 并添加你在GitHub上面的工程HTTPS链接, 选择仓库类型为Git
4. 其他项目根据自己的需要填写后点击 “Create”，创建完后会自动去激活Webhooks，不用再去GitHub设置
5. 一切搞定，从此只要你往这个仓库push代码，readthedoc上面的文档就会自动更新.