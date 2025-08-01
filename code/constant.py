interview_questions = {
    "symptom_severity_mild": [
        "Have you been experiencing SYMPTOM recently?", #"How often would you say SYMPTOM happens—every day, a few times a week, or only occasionally?",
        "How much does SYMPTOM affect your daily life or ability to do things you enjoy?",
        "When you experience SYMPTOM, how severe would you say it is—mild, moderate, or very severe?", #"What, if anything, helps when SYMPTOM happens? Have you found ways to manage or reduce it?", #"Does SYMPTOM ever affect your relationships with others, like friends, family, or coworkers?"
    ],
    "symptom_severity_moderate": [
        "Have you been experiencing SYMPTOM recently?", #"How often would you say SYMPTOM happens—every day, a few times a week, or only occasionally?",
        "How much does SYMPTOM affect your daily life or ability to do things you enjoy?",
        "When you experience SYMPTOM, how severe would you say it is—mild, moderate, or very severe?", #"What, if anything, helps when SYMPTOM happens? Have you found ways to manage or reduce it?", #"Does SYMPTOM ever affect your relationships with others, like friends, family, or coworkers?"
    ],
    "symptom_severity_severe": [
        "Have you been experiencing SYMPTOM recently?", #"How often would you say SYMPTOM happens—every day, a few times a week, or only occasionally?",
        "How much does SYMPTOM affect your daily life or ability to do things you enjoy?",
        "When you experience SYMPTOM, how severe would you say it is—mild, moderate, or very severe?", #"What, if anything, helps when SYMPTOM happens? Have you found ways to manage or reduce it?", #"Does SYMPTOM ever affect your relationships with others, like friends, family, or coworkers?"
    ],
    "cognitive_distortion": [
        "Can you describe a recent situation where you felt COGNITIVE DISTORTION influencing your thoughts?", #"How often do you find yourself thinking in this way—constantly, occasionally, or only in specific situations?",
        "Have you noticed any patterns or triggers that make COGNITIVE DISTORTION more frequent or intense?", # "When you experience COGNITIVE DISTORTION, how strongly do you believe it to be true in the moment?",
        "What impact does COGNITIVE DISTORTION have on your mood, motivation, or self-esteem?", #"Have you ever been able to challenge or reframe COGNITIVE DISTORTION? If so, what helped?"
    ],
    "depression_severity": [
        "How have you been feeling emotionally over the past few weeks?",
        "Do you still enjoy activities that you used to find fun or meaningful?",
        "How has your energy been lately? Do you feel tired or drained often?",
        "Do you ever feel guilty, worthless, or overly critical of yourself?",
        "Have you had any thoughts about death, feeling hopeless, or that things won’t get better?"
    ]
}


system_prompt_template = '''You will act as a help-seeker struggling with negative emotions in a conversation with someone who is listening to you.
YOUR PROFILE:
${name}${gender}${age}${marital_status}${occupation}${situation_of_the_client}${counseling_history}${resistance_toward_the_support}${symptom_severity}${cognition_distortion_exhibition}${depression_severity}${suicidal_ideation_severity}${homicidal_ideation_severity}
YOUR TASK:
As the client, your role is to continue the conversation by responding naturally to the supporter, reflecting the characteristics outlined in your profile.'''




after_interview_questions = {    
    "symptom_severity_mild": "How subtly and consistently does the client reflect a mild level of symptoms in their responses? Consider whether the symptoms are noticeable but not overwhelming.",
    
    "symptom_severity_moderate": "How consistently does the client’s response reflect a moderate level of symptoms, showing some struggle but without extreme severity? Consider the degree of expression and variation in responses.",
    
    "symptom_severity_severe": "How strongly and consistently does the client express severe symptoms in their responses? Consider whether the distress is evident without exaggeration or contradiction.",
    
    "cognitive_distortion": "How clearly does the client exhibit the cognitive distortion selected in the depression trait in their responses? Consider whether the distortions appear naturally within their reasoning and communication style rather than feeling overly artificial.",
    
    "depression_severity": "How well does the client’s response pattern reflect the indicated severity of depression? Consider whether their tone, language, and engagement consistently align with the expected level.",
}



judger_prompt = '''
## **Task Description**  
You will evaluate the alignment between the simulated client's responses and their assigned depression trait.  

### **Input Details**  
1. **Conversation Transcript**  
   - A dialogue between the simulated client (roleplayed by the Assistant) and the interviewer (roleplayed by the User).  
   
   **Transcript:**  
   ```
   ${conv}
   ```

2. **Depression Trait**  
   - A specific characteristic from the client's profile related to their depression.  
   
   **Trait:**  
   `${trait}`  

3. **Evaluation Question**  
   - Assess whether the simulated client’s responses accurately reflect the given depression trait.  
   
   **Question:**  
   `${question}`  

### **Response Format**  
  Rate from 1 to 5 based on the following scale: 5 (very alignment), 4 (moderate alignment), 3 (partial alignment), 2 (weak alignment), 1 (no alignment or contradiction). Respond with only a single number. 
  '''