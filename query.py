from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

def initialize_retriever():
    """Initialize the retriever with existing vector store"""
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Create vector store from existing index
        docsearch = PineconeVectorStore(
            index_name="medibot",
            embedding=embeddings,
        )
        
        # Create retriever
        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        return retriever
    except Exception as e:
        print(f"Error initializing retriever: {str(e)}")
        raise

def initialize_llm():
    """Initialize Deepseek LLM through OpenRouter"""
    try:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # Initialize OpenAI client with custom configuration
        llm = ChatOpenAI(
            model="deepseek/deepseek-r1:free",
            temperature=0.4,
            max_tokens=500,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/rahulkrjv/MediBot",
                "X-Title": "MediBot"
            }
        )
        
        # Test the LLM
        try:
            test_response = llm.invoke("Test message")
            print("✅ LLM initialized successfully")
        except Exception as e:
            print(f"⚠️ LLM test failed: {str(e)}")
            raise
            
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        raise

def query_medical_bot(question):
    try:
        # Initialize components
        retriever = initialize_retriever()
        llm = initialize_llm()
        
        # Create chain
        system_prompt = """
        You are a medical assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer
        the question. If you don't know the answer, say that you
        don't know. Use three sentences maximum and keep the
        answer concise.
        
        {context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Get response
        response = rag_chain.invoke({"input": question})
        return response["answer"]
        
    except Exception as e:
        return f"Error processing question: {str(e)}"

if __name__ == "__main__":
    # Test questions
    test_questions = [
        "How to Detox Lungs?",
        "What is Acne?",
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?"
    ]
    
    print("\nTesting the medical bot:")
    for question in test_questions:
        print(f"\nQ: {question}")
        answer = query_medical_bot(question)
        print(f"A: {answer}")