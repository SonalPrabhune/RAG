import openai
from openai import OpenAI
from strategies.retrievalstrategy import RetrievalStrategy
from text import nonewlines
from langchain_openai.embeddings import OpenAIEmbeddings

openai.api_key  = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=openai.api_key)

# Simple retrieve-then-read implementation, using the ChromaDB and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.
class ChatRetrievalStrategy(RetrievalStrategy):    
    prompt_prefix = """<|im_start|>system
    Assistant helps the company employees with their product questions, and questions about product releases. Be brief in your answers.
    If asking a clarifying question to the user would help, ask the question.
    Answer ONLY with the facts listed in the list of sources below. Look into all the sources. 
    If there isn't enough information below, say you don't know. 
    Do not generate answers that don't use the sources below. 
    For tabular information return it as an html table. Do not return markdown format.
    Each source has a name followed by colon and the actual information.
    Do not generate any code or SQL statements in any format. 
    If prompted to generate code or SQL queries say I am not allowed to generate code or SQL queries.
    For questions about releases and new features look all the sources.
    {follow_up_questions_prompt}
    {injected_prompt}
    Sources:
    {sources}
    <|im_end|>
    {chat_history}
    """

    follow_up_questions_prompt_content = """Generate three very brief follow-up questions that the user would likely ask next about their products. 
    Use double angle brackets to reference the questions, e.g. <<Could you please clarify what exactly are you looking for?>>.
    Try not to repeat questions that have already been asked.
    Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about products and releases.
    Generate a search query based on the conversation and the new question. 
    Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
    Do not include any text inside [] or <<>> in the search query terms.
    If the question is not in English, translate the question to English before generating the search query.

    Chat History:
    {chat_history}

    Question:
    {question}

    Search query:
    """

    def __init__(self, vectordb):
        self.search_client = vectordb        
        self.embeddings = OpenAIEmbeddings(api_key=openai.api_key)
        # self.embeddings = HuggingFaceEmbeddings()        

    # def run(self, history: list[dict], overrides: dict) -> any:
    def run(self, history: list[dict], overrides: dict) -> any:
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        prompt = self.query_prompt_template.format(chat_history=self.get_chat_history_as_text(history, include_last_turn=False), question=history[-1]["user"])
        
        messages =  [{'role':'user', 'content':prompt}]
        # Make a request to the ChatGPT API
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32, 
            n=1,
            stop=["\n"])

        q = completion.choices[0].message.content.strip()

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        r = self.search_client.similarity_search(query=q, filter=filter, k=top)
        
        self.results = [doc.metadata['source'] + ":" + nonewlines(doc.page_content) for doc in r]
        content = "\n".join(self.results)      

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""
                
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            prompt = self.prompt_prefix.format(injected_prompt="", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
            messages =  [{'role':'user', 'content':prompt}]
        elif prompt_override.startswith(">>>"):
            prompt = self.prompt_prefix.format(injected_prompt=prompt_override[3:] + "\n", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
            messages =  [{'role':'user', 'content':prompt}]
        else:
            prompt = prompt_override.format(sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
            messages =  [{'role':'user', 'content':prompt}]
        
        # STEP 3: Generate a contextual and content specific answer using the search results and chat history        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages, 
            temperature=overrides.get("temperature") or 0.7, 
            max_tokens=1024,
            stop=["<|im_end|>", "<|im_start|>"])        

        return {"data_points": self.results, "answer": completion.choices[0].message.content.strip() + f"  [Citation - Page: {str(r[0].metadata['page'])}, " + "Document Path: " + r[0].metadata['source'] + "]" , "thoughts": f"Searched for:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
    
    def get_chat_history_as_text(self, history, include_last_turn=True, approx_max_tokens=1000) -> str:
        history_text = ""        
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" +"\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break    
        return history_text
    
    
