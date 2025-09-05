import requests
from googlesearch import search
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import quote_plus

class GoogleSearcher:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def google_search(self, query, num_results=5):
        """Perform Google search and return results"""
        try:
            search_results = []
            for url in search(query, num_results=num_results, advanced=True, lang='en'):
                search_results.append({
                    'title': getattr(url, 'title', 'No title'),
                    'url': getattr(url, 'url', ''),
                    'description': getattr(url, 'description', 'No description')
                })
            return search_results
        except Exception as e:
            print(f"Google search error: {e}")
            return None
    
    def extract_answer_from_snippet(self, query, snippets):
        """Extract the most relevant answer from search snippets"""
        query_words = query.lower().split()
        
        # Look for the most informative snippet
        best_snippet = None
        max_score = 0
        
        for snippet in snippets:
            if snippet and snippet != 'No description':
                # Score based on relevance to query
                score = sum(1 for word in query_words if word in snippet.lower())
                # Prefer longer, more informative snippets
                score += len(snippet.split()) / 10
                
                if score > max_score:
                    max_score = score
                    best_snippet = snippet
        
        return best_snippet
    
    def get_google_answer(self, query, num_results=3):
        """Get answer from Google search"""
        print(f"🔍 Googling: {query}")
        
        try:
            # Perform Google search
            results = self.google_search(query, num_results=num_results)
            
            if not results:
                return "I couldn't find information about that. Could you please rephrase your question?"
            
            # Extract snippets from search results
            snippets = [result['description'] for result in results if result['description']]
            
            # Get the best answer from snippets
            answer = self.extract_answer_from_snippet(query, snippets)
            
            if answer:
                # Clean up the answer
                answer = re.sub(r'\[\d+\]', '', answer)  # Remove citation numbers
                answer = re.sub(r'\s+', ' ', answer).strip()  # Clean whitespace
                
                # Limit response length
                if len(answer.split()) > 50:
                    sentences = answer.split('. ')
                    if len(sentences) > 1:
                        answer = '. '.join(sentences[:2]) + '.'
                    else:
                        answer = ' '.join(answer.split()[:50]) + '...'
                
                return f"According to my search: {answer}"
            
            return "I found some results but couldn't extract a clear answer. Could you please ask more specifically?"
            
        except Exception as e:
            print(f"Search error: {e}")
            return "I'm having trouble searching right now. Please try again later or ask a different question."
    
    def get_direct_answer(self, query):
        """Try to get a direct answer for common questions"""
        # Simple direct answers for very common questions
        direct_answers = {
            'how to go from milan to madrid': 'You can travel from Milan to Madrid by flight (2h), train (18h with changes), or bus (20h+). The fastest option is flying.',
            'how to get from milan to madrid': 'The best ways are: 1) Flight: 2 hours, 2) Train: via Barcelona (~18h), 3) Bus: 20+ hours.',
            'milan to madrid travel': 'Distance: 1,300 km. Options: Flight (2h, €50-200), Train (18h, €100-300), Bus (20h+, €60-120).',
            'how to reach madrid from milan': 'Quickest: Fly from Milan airports to Madrid Barajas (2h). Cheaper: Bus or train with connections.'
        }
        
        query_lower = query.lower().strip()
        for pattern, answer in direct_answers.items():
            if pattern in query_lower:
                return answer
        
        return None

# Singleton instance
google_searcher = GoogleSearcher()
