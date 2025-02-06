from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from review_assistant.vector_store import VectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

class RAGChain:
    """
    Constructs and manages the RAG chain for code review.
    """

    def __init__(self, vector_store: VectorStore, api_key: str):
        """
        Initializes the RAGChain.

        Args:
            vector_store: An instance of the VectorStore.
            api_key: The OpenAI API key.
        """
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever()
        self.api_key = api_key
        self.llm = None
        self.prompt_template = self._create_prompt_template()
        self.chain = None

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Creates the prompt template for the RAG chain.

        Returns:
            A ChatPromptTemplate instance.
        """
        template = """
        You are a senior software engineer providing a code review.
        Review the following code snippet and provide constructive feedback.
        Consider best practices, potential bugs, and areas for improvement.

        Context from similar code files:
        {context}

        Code to review:
        {code}

        Review:
        """
        return ChatPromptTemplate.from_template(template)

    def _build_chain(self):
        """
        Builds the RAG chain.

        Returns:
            A Runnable chain instance.
        """
        if self.llm is None:
            self.llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key)

        # Ensure we call .invoke when available (works with MagicMock in tests)
        llm_runnable = (
            RunnableLambda(lambda x: self.llm.invoke(x))
            if hasattr(self.llm, "invoke")
            else self.llm
        )

        return (
            {"context": self.retriever, "code": RunnablePassthrough()}
            | self.prompt_template
            | llm_runnable
            | StrOutputParser()
        )

    def invoke(self, code: str) -> str:
        """
        Invokes the RAG chain with the given code.

        Args:
            code: The code to be reviewed.

        Returns:
            The review suggestions from the LLM.
        """
        if self.chain is None:
            self.chain = self._build_chain()
        return self.chain.invoke(code)
