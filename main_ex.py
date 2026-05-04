"""
Automatic Media Generator - Main Pipeline Module (1.0)
Created and Developed by: Kyle Zheng 

Basic Pathway: 
1. Self-generates a dental topic or keyword.
2. The system processes the input and generates a social media post using the ChatOllama model.
3. The generated post is then displayed to the user, who can choose to edit or share it directly from the platform.

"""


# Import necessary libraries and modules 
from dotenv import load_dotenv
import os 
from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

# Load environment variables
load_dotenv()

# [DEBUG/TEST] Environment Verification Test
if os.getenv("LANGSMITH_API_KEY"):
    print("✅ LangSmith API Key Loaded")
else:
    print("❌ LangSmith API Key Missing")


# Initialize the ChatOllama (Deepseek) model with the specified parameters
llm = ChatOllama(model="deepseek-r1:1.5b",
            temperature=0.7,
)


### Topic Generation Chain ###
@traceable(name = "Topic Generation Chain")
def get_topic_generation_chain():
    """Step 1 - The Topic Generator: This chain generates a specific, engaging dental topic for a social media post.""" 

    topic_prompt = ChatPromptTemplate.from_template(
        "You are a dental trend analyst. Suggest one specific, engaging dental topic for a social media post (e.g., 'The truth about charcoal toothpaste'). Output ONLY the topic."
    )
    topic_chain = topic_prompt | llm | StrOutputParser()
    return topic_chain


### Post Generation Chain ###
@traceable(name = "Post Generation Chain")
def get_post_generation_chain():
    """ Step 2 - The Content Creator: This chain generates a high-converting Instagram caption based on the selected topic. """ 
    
    post_prompt = ChatPromptTemplate.from_template(
        "Write a catchy Instagram caption about {topic}. Write a high_converting Instagram caption about {topic}." \
        "Structure: \n" \
        "1. Hook: Start with a question or bold statement to grab attention.\n" \
        "2. Value: Provide a 2-3 sentences of fact, tip, or myth-busting related to the topic.\n"
        "3. Call to Action: Encourage followers to engage (e.g., 'Tag a friend who needs to see this!').\n" \
        "4. Hashtags: Include 3-5 relevant hashtags (e.g., #DentalHealth, #OralCare) and do not include years like #2024 or #2025."   
    )
    post_chain = post_prompt | llm | StrOutputParser()
    return post_chain


### Video Generation Chain ### 
@traceable(name = "Video Generation Chain") 
def get_video_generation_chain():
    """ Step 2.5 - The Video Creator: This chain generates a relevant video prompt based on the selected topic. """ 
    
    video_prompt = ChatPromptTemplate.from_template(
        " Convert this dental social media {topic} into a cinematic video prompt." \
        " The video should be visually engaging and suitable for platforms like Instagram Reels or TikTok. " \
        " Describe the motion, lighting, and audio cues. " \
        " Example: If the topic is 'The truth about charcoal toothpaste', the video prompt could be: 'Start with a close-up of a toothbrush being dipped into black charcoal toothpaste, then zoom out to show a person brushing their teeth with dramatic lighting and upbeat music. Include text overlays that say 'Charcoal Toothpaste: Myth or Miracle?' and end with a call to action like 'Tag a friend who needs to see this!'.' "
    )

    video_chain = video_prompt | llm | StrOutputParser()
    return video_chain  


### Autonomous Generation Function ###
@traceable(name = "Autonomous Generation Process")
def run_autonomous_gen():
    """ Step 3: Run the autonomous generation process using the LLM model and chain. This function orchestrates the entire workflow, from topic generation to post creation. """

    print("--- 🤖 Thinking of a topic... ---")

    # --- 1. Generate the topic --- 
    topic_chain = get_topic_generation_chain()
    raw_topic = topic_chain.invoke({})

    # Clean the 'thinking' out if necessary
    topic = raw_topic.split("</think>")[-1].strip() 
    
    print(f"--- 💡 Selected Topic: {topic} ---")
    

    # --- 2. Generate the post using that topic --- 
    post_chain = get_post_generation_chain()
    raw_post = post_chain.invoke({"topic": topic})
    final_post = raw_post.split("</think>")[-1].strip()
    

    # --- 3. Generate the video prompt using that topic ---
    video_chain = get_video_generation_chain()
    raw_video_prompt = video_chain.invoke({"topic": topic}) 
    video_prompt = raw_video_prompt.split("<tool_call>")[-1].strip()   


    return topic, final_post, video_prompt 


if __name__ == "__main__":
    topic, post, video_prompt = run_autonomous_gen()
    print(f"\nFinal Post:\n{post}")
    print(f"\nVideo Prompt:\n{video_prompt}")

