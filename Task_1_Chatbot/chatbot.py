import re
from datetime import datetime

class RuleBasedChatbot:
    def __init__(self):
        self.rules = {
            r'\bhello\b|\bhi\b|\bhey\b': self.greet,
            r'\bhow are you\b': self.how_are_you,
            r'\byour name\b|\bwhat is your name\b': self.name,
            r'\bwhat is the time\b|\bcurrent time\b': self.time,
            r'\bbye\b|\bgoodbye\b|\bsee you\b': self.goodbye,
            r'\bhelp\b': self.help,
            r'\bwho (are|is) you\b': self.about,
        }
    
    def greet(self, user_input):
        greetings = ["Hello! How can I help you?", "Hi there! What can I do for you?", "Hey! Nice to meet you!"]
        return greetings[hash(user_input) % len(greetings)]
    
    def how_are_you(self, user_input):
        return "I'm doing great! Thanks for asking. How are you doing?"
    
    def name(self, user_input):
        return "I'm a Rule-Based Chatbot created for CodSoft AI Internship!"
    
    def time(self, user_input):
        return f"Current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def goodbye(self, user_input):
        return "Goodbye! Have a nice day!"
    
    def help(self, user_input):
        return "I can help with greetings, time, and general questions. Try asking 'hello', 'how are you', 'what time is it', or 'goodbye'"
    
    def about(self, user_input):
        return "I'm an AI Chatbot trained on rule-based patterns using regex matching. I can respond to various user inputs!"
    
    def get_response(self, user_input):
        user_input = user_input.lower().strip()
        
        if user_input == '':
            return "Please enter something!"
        
        for pattern, response_func in self.rules.items():
            if re.search(pattern, user_input, re.IGNORECASE):
                return response_func(user_input)
        
        return f"I didn't understand '{user_input}'. Type 'help' for available commands."
    
    def chat(self):
        print("=" * 50)
        print("Welcome to Rule-Based Chatbot!")
        print("Type 'exit' or 'quit' to end the conversation")
        print("=" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Bot: Goodbye! Thanks for chatting!")
                break
            
            response = self.get_response(user_input)
            print(f"Bot: {response}")

if __name__ == "__main__":
    chatbot = RuleBasedChatbot()
    chatbot.chat()
