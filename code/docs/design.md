# 设计 / UI

本仓库以 **C++ 库与命令行 / 服务端示例** 为主，**没有**产品级 Web/App 视觉设计系统。

## 适用说明

- `source/web`：Poco 网络库封装与演示服务，不是面向品牌的落地页。
- 不要求落地页式 hero、字体选型或营销动效。
- 若未来增加管理台或对外站点，应另开前端工程并单独维护设计文档；勿把 Node/Next 模板硬塞进本 CMake 树。

## 文档与产物

- 人类与 agent 文档：Markdown（`PRODUCT.md`、`docs/*`）。
- 历史文件 `docs-legacy-fastapi-swagger.html` 仅为无关快照，不作为设计规范。
