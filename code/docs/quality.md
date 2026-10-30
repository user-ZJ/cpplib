# 质量门禁

本仓库是 **CMake / C++** 工程。质量以配置、编译、模块测试为准，**不**使用 ESLint、Vitest、Next.js CI 模板。

## 必做（改代码时）

1. **配置**：在 out-of-source 构建目录执行 `cmake ..`（或项目惯用生成器），确认无新增 FATAL。
2. **编译**：至少 `make <affected_target>`；跨模块改动优先全量 `make`。
3. **测试 / 示例**：`unittest/` 是 GoogleTest 套件（`BUILD_UNITTEST`，默认开启），编完后在构建目录执行 `ctest --output-on-failure -R unit_test`，或直接运行 `bin/unit_test`。`test/` 仍是各模块示例程序。
4. **回归范围**：改公共头文件 / CMake 选项时，编译所有依赖该目标的下游。

## 建议

- 保持 C++17；避免无必要的 C++20/23 仅因本地编译器支持。
- 新警告：优先修本改动引入的 warning；勿借机全库 `-Werror` 除非团队约定。
- 大文件、模型、`fc_base` 产物不进 git。

## 不要做

- 为 agent 便利引入 `package.json` / Node 测试栈。
- 在未要求时改依赖版本或删除 `SKIP_*` 防护。
- 提交密钥、私钥、生产库连接信息。

## Agent 自检清单

- [ ] 只改任务相关文件
- [ ] 受影响目标已编译
- [ ] 文档与行为一致（若行为变了）
- [ ] `git status` 中无意外生成物
