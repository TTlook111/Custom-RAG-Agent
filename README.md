# Custom-RAG-Agent

精简后的 RAG 项目结构：

```text
Custom-RAG-Agent/
├─ main.py                        # 入口编排，导出 graph
└─ custom_rag_agent/
   ├─ config/
   │  └─ config.py               # 环境变量与模型配置
   ├─ retrievers/
   │  └─ web_retriever.py        # 文档加载、切分、向量化与检索器构建
   ├─ workflows/
   │  └─ rag_graph.py            # Prompt、Tool、节点与工作流编排
```

## 环境说明

项目使用 `uv` 管理 Python 环境与依赖，Python 版本要求为 `>=3.13`。

## 初始化环境

在项目根目录执行：

```bash
uv venv
uv sync
```

Windows 激活虚拟环境：

```powershell
.venv\Scripts\activate
```

## 运行前配置

在项目根目录 `.env` 中配置：

```env
DASHSCOPE_API_KEY=your_key
CHAT_MODEL_NAME=qwen3-max
EMBEDDING_MODEL_NAME=text-embedding-v4
```

说明：
- `DASHSCOPE_API_KEY` 为必填项。
- `CHAT_MODEL_NAME`、`EMBEDDING_MODEL_NAME` 为可选项；未配置时分别默认使用 `qwen3-max` 与 `text-embedding-v4`。

## 检索缓存说明

- 检索器会优先使用 FAISS 本地持久化缓存，缓存目录为 `.cache/faiss/<cache_key>/`。
- 当 URL 列表、切分参数或 embedding 模型变化时，会自动生成新的缓存键并重建索引。
- 若运行环境未安装 FAISS，则自动退化为内存向量库（重启后会重新构建索引）。

## 使用方式

推荐在 `uv` 环境中运行项目：

```bash
uv run python main.py
```

主入口会导出可直接使用的 `graph`：

```python
from main import graph
```
