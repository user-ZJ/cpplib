=========================
readthedocs托管文档
=========================

https://readthedocs.org/

一般的做法是将文档托管到版本控制系统比如github上面，push源码后自动构建发布到readthedoc上面， 这样既有版本控制好处，又能自动发布到readthedoc

先在GitHub创建一个仓库名字叫xxx（注意项目要是公开项目）， 然后在本地 `.gitignore` 文件中添加 `build/` 目录，初始化git，commit后，添加远程仓库。  

具体几个步骤非常简单，参考官方文档：https://github.com/rtfd/readthedocs.org:

1. 在Read the Docs上面注册一个账号
2. 登陆后点击 “Import”.
3. 给该文档项目填写一个名字比如 “xxx”, 并添加你在GitHub上面的工程HTTPS链接, 选择仓库类型为Git
4. 其他项目根据自己的需要填写后点击 “Create”，创建完后会自动去激活Webhooks，不用再去GitHub设置
5. 一切搞定，从此只要你往这个仓库push代码，readthedoc上面的文档就会自动更新.

