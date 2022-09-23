# git submodule使用

## 创建子模块 
```shell 
# Usage $ git submodule add [url] [path] 
# With path 
$ git submodule add https://github.com/laozhu/hugo-nuo themes/hugo-nuo 
# Without path 
$ cd themes $ git submodule add https://github.com/laozhu/hugo-nuo 
```
在这个仓库下面多出了一个 `.gitmodules` 文件，该文件列出了所包含的子模块列表，并为列表中每一个子模块指定了本地路径（path）和远程仓库地址（url），除此以外我们还可选为子模块指定 `branch` 分支，不指定默认为 `master` 分支。 
```shell 
[submodule "themes/hugo-nuo"]  
    path = themes/hugo-nuo  
    url = https://github.com/laozhu/hugo-nuo  
    #branch=xxx 
```
## 查看子模块 
要查看当前代码仓库所使用的子模块及其状态，除了看 `.gitmodules` 文件外，还可以执行 `git submodule` 命令。 
```shell 
$ git submodule 
# 已检出子模块代码 
cedbe91340dbcff661fa089b116441b11b050d38 themes/hugo-nuo (heads/master) # 前面带 - 表示未检出代码，子模块是空文件夹 
-cedbe91340dbcff661fa089b116441b11b050d38 themes/hugo-nuo (heads/master) 
```
## 克隆含有子模块的项目 
当你需要克隆一个包含了子模块的远程代码仓库，有两种方式 
```shell 
# Clone => Init => Update 
$ git clone https://github.com/laozhu/my-blog 
$ git submodule init $ git submodule update 
# Clone recursive $ git clone --recursive https://github.com/laozhu/my-blog 
```
## 拉取子模块更新 
拉取子模块更新不再需要 `clone` 和 `init` 操作，只需 `update` 即可，当你的主代码仓库执行 `pull` 或者切换分支操作后，别忘了执行 `update` 操作，以保证子模块中的代码与新的 `.gitmodules` 中版本一致。为了防止误提交旧的子模块依赖信息，每次执行 `pull` 后，可使用 `git status`            查看文件状态。 
```shell 
# After Pull 
$ git pull https://github.com/laozhu/my-blog 
# After Checkout 
$ git checkout -b develop origin/develop 
# You need 
$ git status -s $ git submodule update 
```
## 提交子模块修改 
当你需要对当前使用的某个子模块进行修改，并且希望所做修改能够提交到子模块的主仓库，一定要记得切换到 `master` 分支再修改并提交。 
```shell 
cd themes/hugo-nuo 
git checkout master 
git add . 
git commit -m "Create shortcode for stackblitz" 
git push orgin master 
```
## 将目录转化为子模块 
项目开发过程中会遇到这样一个场景：觉得某一个功能抽象程度很高，与整个系统也不耦合，于是就希望把这个功能独立成一个模块进行团队共享甚至开源，这时候我们就需要将一个子目录转化为一个子模块，但因为子目录的代码在主代码仓库中已经被跟踪过了，如果我们仅仅是删除子目录，添加同名的子模块的话，`git` 就会报下面的错误： 
```shell 
$ rm -rf themes/hugo-nuo 
$ git submodule add https://github.com/laozhu/hugo-nuo themes/hugo-nuo 
'hugo-nuo' already exists in the index 
```
使用 `git rm` 取消子目录的暂存即可 
```shell 
$ git rm -r themes/hugo-nuo 
$ git submodule add https://github.com/laozhu/hugo-nuo themes/hugo-nuo 
```
## 删除子模块 
Git 中删除子模块略微麻烦一些，因为目前还没有 `git submodule rm` 这样的命令行，我们要做很多工作才能删得干净 
```shell 
$ git submodule deinit themes/hugo-nuo 
$ vim .gitmodules # 移除要删除的子模块 
$ git add .gitmodules 
$ git rm --cached themes/hugo-nuo 
$ rm -rf .git/modules/themes/hugo-nuo 
$ rm -rf themes/hugo-nuo 
$ git commit -m "Remove submodule themes/hugo-nuo" 
```

## 移动子模块
```shell
# Upgrade to Git 1.9.3
git mv old/submod new/submod
```

## 修改子模块url
1. 更新 .gitsubmodule中对应submodule的条目URL
2. 更新 .git/config 中对应submodule的条目的URL
3. 执行 git submodule sync

## 子模块批量处理 
对于像 [gohugoio/hugoThemes](https://github.com/gohugoio/hugoThemes) 这种超级依赖子模块的仓库怎么管理呢，使用 `foreach` 循环指令就可以啦 
```shell 
# Nested submodule 
$ git submodule foreach git submodule update 
# Checkout master then pull 
git submodule foreach 'git checkout master; git pull'    
```
