"""
通义千问LLM客户端
"""
from typing import List, Dict, Optional
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from config import RAGConfig


class QwenLLMClient:
    """通义千问LLM客户端"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = None
        self.history = []

    def _init_llm(self):
        """初始化LLM"""
        if self.llm is None:
            if not self.config.dashscope_api_key:
                raise ValueError("请设置DASHSCOPE_API_KEY环境变量或在配置中提供API密钥")

            self.llm = ChatTongyi(
                model=self.config.qwen_model_name,
                dashscope_api_key=self.config.dashscope_api_key,
                temperature=0.7,
                max_tokens=2000
            )
            print(f"通义千问LLM初始化完成，模型: {self.config.qwen_model_name}")

    def chat(self,
             user_message: str,
             context: str = "",
             system_prompt: str = None) -> str:
        """单次对话"""
        self._init_llm()

        # 默认系统提示
        if system_prompt is None:
            system_prompt = """你是一个智能助手，基于提供的上下文信息来回答用户问题。
请遵循以下原则：
1. 仅基于提供的上下文信息回答问题
2. 如果上下文中没有相关信息，请明确说明
3. 回答要准确、简洁、有帮助
4. 在回答末尾标注信息来源"""

        # 构建消息
        messages = [
            SystemMessage(content=system_prompt),
        ]

        # 如果有上下文，添加到消息中
        if context:
            context_message = f"以下是相关的背景信息：\n\n{context}\n\n请基于以上信息回答我的问题。"
            messages.append(HumanMessage(content=context_message))

        messages.append(HumanMessage(content=user_message))

        # 调用LLM
        chain = self.llm | StrOutputParser()
        response = chain.invoke(messages)

        return response

    def chat_with_history(self,
                         user_message: str,
                         context: str = "",
                         system_prompt: str = None) -> str:
        """多轮对话（带历史记忆）"""
        self._init_llm()

        # 默认系统提示
        if system_prompt is None:
            system_prompt = """你是一个智能助手，能够进行多轮对话。
请保持对话的连贯性，记住之前的对话内容。
基于提供的上下文信息来回答用户问题。"""

        # 构建消息列表
        messages = [SystemMessage(content=system_prompt)]

        # 添加历史消息
        messages.extend(self.history)

        # 如果有上下文，添加上下文
        if context:
            context_message = f"相关背景信息：\n{context}\n"
            messages.append(HumanMessage(content=context_message))

        # 添加当前用户消息
        current_message = HumanMessage(content=user_message)
        messages.append(current_message)

        # 调用LLM
        chain = self.llm | StrOutputParser()
        response = chain.invoke(messages)

        # 更新历史记录
        self.history.append(current_message)
        self.history.append(AIMessage(content=response))

        # 限制历史长度（保留最近5轮对话）
        if len(self.history) > 10:
            self.history = self.history[-10:]

        return response

    def clear_history(self):
        """清空对话历史"""
        self.history = []
        print("对话历史已清空")