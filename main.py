"""项目运行入口：导出懒加载 graph，并支持命令行快速自测。"""

from custom_rag_agent.main import get_graph, graph


def main() -> None:
    """初始化 graph，确保运行时配置与依赖可用。"""
    get_graph()
    print("Graph 初始化完成。")


if __name__ == "__main__":
    main()
