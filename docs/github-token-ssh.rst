github-token-ssh
======================

github-token
-------------------
github使用token替换原有密码登录。用户在生成token时可以设定token的访问权限，过期时间等，与密码不同，它不受字典或暴力攻击。

生成token步骤：

设置->开发者设置->个人访问令牌->生成新令牌->为您的令牌命名并选择您要授予它的范围和权限->生成令牌

注意：将令牌复制到剪贴板并将其存储在安全的地方。出于安全原因，离开此页面后，您将无法再次看到该令牌。

**token使用：**

.. code-block:: shell

    git clone https://github.com/<USERNAME>/<REPO>.git
    用户名：your_username
    密码：your_token

github-ssh
----------------

ssh秘钥生成
~~~~~~~~~~~~~~~~~
.. code-block:: shell

    ssh-keygen -t ed25519 -C "your_email@example.com"
    # 如果您使用的是不支持 Ed25519 算法的旧系统，请使用：
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    # 后面一路回车即可，生成的秘钥在/home/you/.ssh目录下


在github上添加ssh秘钥
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
打开.ssh目录下的.pub文件，并复制其内容，注意不要添加换行符或空格

在github页面上点击：

设置->SSH 和 GPG 密钥->新建 SSH 密钥->给SSH密钥起一个描述性标题，然后将复制的密钥粘贴到“密钥”文本区域中->添加SSH密钥

测试是否添加成功
~~~~~~~~~~~~~~~~~~~~~
.. code-block:: shell

    ssh -T git@github.com


收到以下信息表示成功：

::

    Hi <USERNAME>! You've successfully authenticated, but GitHub does not provide shell access.

更新远程仓库信息
~~~~~~~~~~~~~~~~~~~
.. code-block:: shell

    git remote remove origin
    git remote add origin git@github.com:<USERNAME>/<REPO>.git


参考
----------------
https://betterprogramming.pub/dealing-with-github-password-authentication-deprecation-1b59ced90065
