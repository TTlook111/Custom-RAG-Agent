# Custom-RAG-Agent

精简后的 RAG 项目结构：

```text
Custom-RAG-Agent/
├─ main.py                        # 入口编排，导出 graph
└─ custom-rag-agent/
   ├─ config/
   │  └─ config.py               # 环境变量与模型配置
   ├─ retrievers/
   │  └─ web_retriever.py        # 文档加载、切分、向量化与检索器构建
   ├─ workflows/
   │  └─ rag_graph.py            # Prompt、Tool、节点与工作流编排
```

## 运行前配置

在项目根目录 `.env` 中配置：

```env
DASHSCOPE_API_KEY=your_key
CHAT_MODEL_NAME=qwen3-max
EMBEDDING_MODEL_NAME=text-embedding-v4
```

## 使用方式

主入口会导出可直接使用的 `graph`：

```python
from main import graph
```
