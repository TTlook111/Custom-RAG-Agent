# Custom-RAG-Agent

按功能拆分后的专业化 RAG 项目结构：

```text
Custom-RAG-Agent/
├─ main.py                        # 入口编排，导出 graph
└─ custom-rag-agent/
   ├─ config/
   │  └─ config.py               # 环境变量与模型配置
   ├─ models/
   │  └─ model_factory.py        # Chat/Embedding 模型工厂
   ├─ retrievers/
   │  └─ web_retriever.py        # 文档加载、切分、向量化与检索器构建
   ├─ tools/
   │  └─ retrieval_tools.py      # LangChain Tool 封装
   ├─ prompts/
   │  └─ rag_prompts.py          # Prompt 模板
   ├─ workflows/
   │  └─ rag_graph.py            # LangGraph 节点与图编排
   └─ sources/
      └─ blog_urls.py            # 数据源列表
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
