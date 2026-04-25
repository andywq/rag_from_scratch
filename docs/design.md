# 最简化 RAG 系统设计文档

## 1. 设计目标
在最小依赖前提下，构建可运行的端到端 RAG 系统，包含：
- 数据导入（MongoDB -> Embedding -> FAISS）；
- 查询服务（Query -> Retrieval -> LLM Answer）；
- 配置驱动（`config.json`）。

## 2. 系统结构
建议目录结构：
- `app/entrypoint.py`：主入口，负责依赖装配、应用启动与整体流程编排。
- `app/config.py`：配置加载与类型定义。
- `app/web_service/`
  - `app.py`：FastAPI 应用工厂与路由注册。
  - `schemas.py`：请求/响应模型定义。
  - `handlers.py`：HTTP 请求处理与参数校验。
- `app/models/`
  - `embedding_client.py`：OpenAI 兼容 Embedding 调用封装。
  - `llm_client.py`：OpenAI 兼容 Chat Completion 调用封装。
- `app/vector_store/`
  - `faiss_store.py`：FAISS 索引加载、检索与持久化。
  - `metadata_store.py`：JSONL 元数据读取与映射。
- `app/rag/`
  - `service.py`：RAG 主流程编排（检索 + 提示词构建 + 总结）。
- `scripts/import_from_mongodb.py`：MongoDB 全量导入并重建索引。
- `data/`：本地索引与元数据存储目录。

设计原则：
- 各层通过清晰接口连接，避免跨层直接依赖实现细节。
- 替换任一组件（如 LLM 服务或向量存储实现）时，仅需改动对应目录内实现及装配逻辑。

## 3. 配置设计（config.json）
配置文件采用 JSON，示例字段如下（值仅占位）：
- `mongodb.uri`：MongoDB 连接串。
- `mongodb.database`：数据库名。
- `mongodb.collection`：集合名。
- `mongodb.text_fields`：文本字段列表（按顺序拼接成可检索文本，schema 未定时通过此参数适配）。
- `mongodb.id_field`：文档主键字段名（默认可为 `_id`）。
- `providers.chat.provider`：聊天提供方（`openai`/`ollama`）。
- `providers.chat.base_url`：聊天服务地址。
- `providers.chat.api_key`：聊天服务密钥（若不需要可为空）。
- `providers.chat.model`：聊天模型名。
- `providers.embedding.provider`：Embedding 提供方（`openai`/`ollama`）。
- `providers.embedding.base_url`：Embedding 服务地址。
- `providers.embedding.api_key`：Embedding 服务密钥（若不需要可为空）。
- `providers.embedding.model`：Embedding 模型名。
- `retrieval.top_k`：检索返回条数。
- `storage.index_path`：FAISS 索引文件路径。
- `storage.metadata_path`：元数据文件路径（JSONL）。

## 4. 数据模型
### 4.1 索引元数据
FAISS 仅存向量，文档元信息单独存 JSONL，每行一条记录：
- `doc_id: str`
- `text: str`
- `source: dict`（可选，保留原文档关键字段）

约束：
- FAISS 向量序号与 JSONL 行号一一对应。

### 4.2 查询返回
接口返回结构：
- `answer: str`
- `hits: list[{doc_id, score, text}]`

## 5. 关键流程
### 5.1 全量导入流程
1. 读取 `config.json`。
2. 连接 MongoDB，遍历集合文档。
3. 按 `text_fields` 拼接文本（缺失字段跳过，空文本过滤）。
4. 批量调用 Embedding API 获取向量。
5. 重建 FAISS 索引并写入 `index_path`。
6. 写出 JSONL 元数据到 `metadata_path`。

### 5.2 查询流程
1. `web_service` 接收查询字符串。
2. 调用 Embedding API 获取 query 向量。
3. 通过 `vector_store` 在 FAISS 中检索 Top-K。
4. 读取对应元数据并组装上下文。
5. 通过 `models` 调用 Chat API 生成总结回答。
6. 返回回答与命中结果。

## 6. 接口定义
### 6.1 HTTP API
- `POST /query`
  - Request:
    - `query: str`
  - Response:
    - `answer: str`
    - `hits: list`

### 6.2 导入脚本
- 命令：`python scripts/import_from_mongodb.py --config config.json`
- 行为：全量重建 FAISS 与元数据文件。

## 7. 错误处理与日志
- 对外部依赖错误（MongoDB / OpenAI API / 文件读写）进行异常捕获并输出错误日志。
- FastAPI 接口返回标准 HTTP 错误码（如 400、500）。
- logging 支持通过配置切换 debug 级别。

## 8. 质量与测试设计
- `make lint`：ruff（含复杂度阈值配置）。
- `make build`：`py_compile` 基础语法检查。
- `make unittest`：核心单元测试（配置加载、文本拼接、检索流程中的可替换组件）。
- `make test`：聚合测试命令。

当前版本以“最小可运行”优先，测试覆盖率目标先达到可用基线并逐步提升。
