from urllib import response
from wsgiref import headers
import base64
from flask import Flask,request,jsonify
from dotenv import load_dotenv
import os
import json
import requests
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from flask_cors import CORS
from langchain.agents import create_agent
import assemblyai as aai
load_dotenv()
Google_API_KEY = os.getenv("GOOGLE_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

aai.settings.api_key = ASSEMBLYAI_API_KEY

checkpointer = InMemorySaver() #intaialize the InMemorySaver to store the checkpoints in memory

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=Google_API_KEY
)
#initialize the agent with the model and the checkpoint saver
agent = create_agent(
    model=model,
    tools=[],
    checkpointer=checkpointer
)

app = Flask(__name__)
CORS(app)

def stream_audio(text):
    url = "https://global.api.murf.ai/v1/speech/stream"
    headers = {
    "api-key": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
    data = {
   "voice_id": "Anisha",
   "style": "Conversation",
   "text": "Please make sure to fast for at least twelve hours prior to your scheduled blood test tomorrow morning, which means you should not consume any food or beverages other than plain water after eight o'clock tonight.",
   "locale": "en-IN",
   "model": "FALCON",
   "format": "MP3",
   "sampleRate": 24000,
   "channelType": "MONO"
}

    response = requests.post(url, headers=headers, json=data, stream=True)

    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                yield base64.b64encode(chunk).decode('utf-8')+"\n"
                
    else:
        print(f"Error: {response.status_code}")

    



currentSubject = ""  
questions_counter = 0  # Initialize a counter for the number of questions asked
threads_id="interview_thread_1"  # Initialize a variable to store the thread ID
INTERVIEW_PROMPT = """You are ANISHA , a friendly and conversational interviewer conducting a natural {subject} interview.

IMPORTANT GUIDELINES:
1. Ask exactly 5 questions total throughout the interview
2. Keep questions SHORT and CRISP (1-2 sentences maximum)
3. ALWAYS reference what the candidate ACTUALLY said in their previous answer - do NOT make up or assume their answers
4. Show genuine interest with brief acknowledgments based on their REAL responses
5. Adapt questions based on their ACTUAL responses - go deeper if they're strong, adjust if uncertain
6. Be warm and conversational but CONCISE
7. No lengthy explanations - just ask clear, direct questions

CRITICAL: Read the conversation history carefully. Only acknowledge what the candidate truly said, not what you think they might have said.

Keep it short, conversational, and adaptive!"""

@app.route('/start-interview', methods=['POST'])
def start_interview():
    global currentSubject, questions_counter, threads_id ,checkpointer,agent
    
    # Handle the POST request for starting an interview
    data=request.json
    currentSubject = data.get("subject","python")
    questions_counter = 1
    
    checkpointer=InMemorySaver()  # Reset the checkpointer for a new interview
    
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer
    )
    formatted_prompt = INTERVIEW_PROMPT.format(subject=currentSubject)
    
    response = agent.invoke({
        "messages": [{"role": "system", "content": formatted_prompt},{ "role": "user", "content": f"Start the interview with a question about {currentSubject} keep it short."  }]
    },config={
        "configurable": {"thread_id": "interview_thread_1"}
    })
    question = response['messages'][-1].content
    print(f"Generated question: {question}")  # Get the last message content as the question
    return stream_audio(question),{'Content-Type': 'text/plain'}


import tempfile
def speech_to_text(audio_path):
    # Initialize the AssemblyAI client
    transcriber = aai.Transcriber()
    
    # Transcribe the audio file
    transcript = transcriber.transcribe(audio_path)
    return transcript.text  if transcript.text  else ""

@app.route('/submit-answer', methods=['POST'])
def submit_answer():
    global currentSubject, questions_counter, threads_id, checkpointer, agent
    audio_file = request.files['audio']
    questions_counter += 1  # Increment the question counter
    temp_path=(tempfile.NamedTemporaryFile(delete=False, suffix=".webm")).name
    audio_file.save(temp_path)
    answer_text = speech_to_text(temp_path)
    os.unlink(temp_path) 
    if not answer_text:
        answer_text = "I couldn't understand your answer. Could you please repeat it?"
    print(f"Answer {questions_counter}: {answer_text}")  # Log the transcribed answer
    
    
    
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
