#Answer by Ivan Joseph ( M01093025 )

Dear grandma , 

this is application is a website that is run in browsers , which helps computers to connect to different website in the Internet.
The website is called Research Paper Agent which uses Artificial Intellgience ( a very revolutionary discovery in the recent years which mimic human intelligence).
When you open the website , the website will ask you to upload any research paper in PDF format, which is a widely used document format type, and you can ask questions related to the paper or PDF document that is uploaded , along with a facility to search the web for more information needed.
After the PDF is uploaded , the contents in the PDF is split into chunks , using text-embedding models like all-MiniLM-L6-v2 and sentence-transformers, similar meaning sentences are put together into a vector DB with certain index , vector DB used is Chroma DB

This is done so that when the application gives back response to the user , it has proper context , page number and ensure there is no data hallucination.

There is a guardrail layer in this application , where it checks if your questions/answers generated are related to the PDF uploaded or not, the application only allows questions/answers related to the PDF , and blocks any other type of content.

Generative AI is a major part of the application where we use models like llama-3.3-70b to generate the text content , to decide the agent to use etc.

There are multiple agents in this application :
Router Node 
RAG Agent
Web Agent
Summarizer
Synthesizer Node

Router Node is the first step which will decide which Agent to use among RAG , Web , Summarizer.
After the Agents does it job , the last step is Synthesizer Node which compiles all response of all agents and shows the content back in text format for user to see.

This application will help students/researchers/teachers etc to quickly learn what is in the document and will be a great tool in the academic sector.


#Answer by Ihsan Ul Haq ( M01098089 )

What our project is?
Grandma, our project is basically an AI agent that can help in research about topics people want to know and learn about. Like the long research papers will be cut short so that anyone using can understand it easily and not take up their time going through everything in the research paper. People can save time to come visit you



How it works? 
1. You open this website: “https://research-paper-agent-tapppptcuyf7wejkewz3kfu.streamlit.app”

2. Then there will be a drag and drop area on the left side, upload any research paper which should be a pdf file, then my agent will do its work by cutting the data into tiny parts and blocks, and keeps them in a container.

3. Right when you upload, the agent will suggest 3 questions so in case you don’t have any questions in mind, the agent will help you with that, and you just have to click on it.

4. When you ask a question relating to the uploaded paper , it finds the needed parts and arranges them to make meaningful summary / small research and also goes through the internet to find any necessary and relevant information. Then it combines both the answers, and shows it to you.

5. If you ask a question not relating to the uploaded paper, then it checks if it is an appropriate question for a research. And if it is then it goes through the internet to find other research papers. And if it is not then it says that its not a proper question.

6. There’s also a security officer, checking if anyone tries to break in or tries to do anything bad. So it’s in safe hands.

7. Last but not least, there’s a toggle for you “Grandma Mode”. It summarizes in a way that even you can understand.



Tech Stack?
LLM’s:
LangGraph
LangChain
Groq

Vector Database & Retrieval (RAG):
ChromaDB
HuggingFace - all-MiniLM-L6-v2
Cohere

Frontend:
Streamlit
LangSmith
