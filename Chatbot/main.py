from utils.response_generator import ResponseGenerator
import warnings
warnings.filterwarnings('ignore')

class ChatBot:
    def __init__(self):
        self.response_generator = ResponseGenerator(
            intents_path='data/intents.json',
            model_path='models/intent_classifier.h5'
        )
        self.conversation_history = []
    
    def display_welcome(self):
        print("🤖 Welcome to the Enhanced AI ChatBot!")
        print("I can help you with:")
        print("- Greetings and conversations")
        print("- Weather information")
        print("- Current time")
        print("- Telling jokes")
        print("- Programming and AI questions")
        print("- General knowledge")
        print("- Web searches for unknown topics")
        print("\nType 'quit' to exit the chat.\n")
    
    def start_chat(self):
        self.display_welcome()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                self.conversation_history.append(f"You: {user_input}")
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("🤖 Goodbye! Have a great day!")
                    break
                
                response = self.response_generator.process_input(user_input)
                print(f"🤖 {response}")
                
                self.conversation_history.append(f"Bot: {response}")
                
                if self.response_generator.last_intent == "weather":
                    print("🤖 (Please tell me which city you're interested in)")
                
            except KeyboardInterrupt:
                print("\n🤖 Goodbye! Thanks for chatting!")
                break
            except Exception as e:
                print(f"🤖 Sorry, I encountered an error: {e}")
                print("🤖 Please try again.")

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.start_chat()
