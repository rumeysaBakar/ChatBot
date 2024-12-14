import asyncio
import logging
from typing import List, Dict, Optional
from config.settings import Settings
from core.embeddings import NvidiaEmbeddings
from core.llm import LLMManager
from core.vector_store import VectorStore
from core.conversation_manager import ConversationManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedChatbot:
    """Enhanced chatbot with parallel processing and conversation management"""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize chatbot components"""
        try:
            # Initialize basic attributes
            self._initialized = False
            self.settings = settings or Settings()
            logger.info("Initializing chatbot components...")

            # Initialize core components
            self.embeddings = NvidiaEmbeddings(self.settings)
            self.llm = LLMManager(self.settings)
            self.vector_store = VectorStore(self.settings, self.embeddings)
            self.conversation_manager = ConversationManager(self.settings)

            logger.info("Base components initialized successfully")
        except Exception as e:
            logger.error(f"Error during chatbot initialization: {str(e)}")
            raise

    async def initialize(self):
        """Initialize async components"""
        if self._initialized:
            return

        try:
            # Initialize vector store first
            logger.info("Initializing vector store...")
            await self.vector_store.initialize()

            # Initialize LLM manager
            logger.info("Initializing LLM manager...")
            await self.llm.initialize()

            self._initialized = True
            logger.info("Chatbot initialization complete")

        except Exception as e:
            logger.error(f"Error during async initialization: {str(e)}")
            # Ensure cleanup on initialization failure
            await self.close()
            raise

    async def close(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'llm'):
                await self.llm.close()
            if hasattr(self, 'embeddings') and hasattr(self.embeddings, 'close'):
                await self.embeddings.close()
            if hasattr(self, 'conversation_manager') and hasattr(self.conversation_manager, 'close'):
                await self.conversation_manager.close()

            self._initialized = False
            logger.info("Cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

    async def process_message(self, user_id: str, message: str) -> str:
        """Process a user message and generate a response"""
        if not self._initialized:
            raise RuntimeError("Chatbot not initialized. Call initialize() first.")

        try:
            logger.info(f"Processing message for user {user_id}")

            # Get conversation context and relevant information in parallel
            tasks = [
                self.conversation_manager.get_context(user_id),
                self.vector_store.similarity_search(message)
            ]

            # Wait for all parallel tasks
            context, search_results = await asyncio.gather(*tasks)

            # Prepare messages for the model
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant. 
                    Previous conversation summary: {context['summary']}
                    Relevant context: {' '.join([r[0].get('text', '') for r in search_results])}"""
                }
            ]

            # Add recent messages if available
            if context.get('recent_messages'):
                for msg in context['recent_messages']:
                    messages.extend([
                        {"role": "user", "content": msg["message"]},
                        {"role": "assistant", "content": msg["response"]}
                    ])

            # Add current message
            messages.append({"role": "user", "content": message})

            # Generate response
            logger.info("Generating response...")
            response = await self.llm.generate_response(messages)

            # Save conversation
            await self.conversation_manager.add_message(
                user_id=user_id,
                message=message,
                response=response,
                context={
                    "search_results": search_results,
                    "summary_used": bool(context.get("summary"))
                }
            )

            logger.info("Message processing completed successfully")
            return response

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "I apologize, but I encountered an error processing your request."


async def main():
    chatbot = None
    try:
        # Create chatbot instance
        chatbot = EnhancedChatbot()
        logger.info("Chatbot instance created")

        # Initialize chatbot
        logger.info("Initializing chatbot...")
        await chatbot.initialize()

        # Example conversation
        questions = [
            "What are the main applications of GPU computing?",
            "How does CUDA programming work?",
            "Can you explain tensor cores in more detail?"
        ]

        for question in questions:
            logger.info(f"\nProcessing question: {question}")
            try:
                response = await chatbot.process_message("user123", question)
                print(f"\nUser: {question}")
                print(f"Assistant: {response}")
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                print("I apologize, but I encountered an error processing your question.")
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print("An error occurred. Please check the logs for details.")

    finally:
        if chatbot:
            logger.info("Cleaning up resources...")
            await chatbot.close()


if __name__ == "__main__":
    asyncio.run(main())