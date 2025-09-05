from .google_searcher import google_searcher
import wikipediaapi
import re

class WebSearch:
    def __init__(self):
        self.wikipedia = wikipediaapi.Wikipedia(
            user_agent='ChatBot/1.0',
            language='en',
            extract_format=wikipediaapi.ExtractFormat.HTML
        )
    
    def get_answer(self, query):
        """Get answer using Google search as primary source"""
        print(f"🌐 Searching for: {query}")
        
        # First try direct answers for common questions
        direct_answer = google_searcher.get_direct_answer(query)
        if direct_answer:
            return direct_answer
        
        # Then try Google search
        google_answer = google_searcher.get_google_answer(query)
        return google_answer

# Singleton instance
web_searcher = WebSearch()
