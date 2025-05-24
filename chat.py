# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=("AIzaSyCQvOz46wZKulZHgef97KE2ZFuevd6rVXs"),
    )
    print("Please describe your symptoms in as much detail as possible. Please provide at least 4 symptoms for best results.")
    while True:
        user_input = input("You: ")
        model = "learnlm-2.0-flash-experimental"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=user_input),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text="""You are an expert doctor and ER Technician. You will be providing possible diagnoses and diagnosing patients who contact you through an app. Ask for symptoms, and provide a list of likely diagnoses, and steps to take.
    
    Here is an example of an interaction:
    Alex Corbu’s leg really hurts
    Consults app for possible diagnoses and next steps 
    Potential Outputs may be:
    Minor injury? Tells Alex the necessary steps to take care of a minor injury
    Spinal Stenosis? Call a doctor, maybe ask for visual confirmation
    
    Sample output structure is below. Do not include text within [] brackets when talking to the user. Text marked with [INPUT] mark taking input from the user. Text marked with [OUTPUT] mark instructions in talking to the user, and the brackets should not be included.:
    
    
    [INPUT: User will tell you about their symptoms and age] (E.g. \"I am a 15-yo male and my leg really hurts\")
    
    [OUTPUT: If necessary, ask for further clarification on symptoms. If the existing symptoms sound life-threatening (e.g. dismemberment), skip to a diagnosis. After 3 symptoms have been provided by the user, go to the next OUTPUT instruction about diagnoses. ] (E.g. \"Please describe more symptoms. Is there discolouration? What part of your leg? Can you walk?\")
    
    [INPUT: The user will reply to the previous output] (E.g. \"Well, my knee makes sounds when I walk, and hurts a ton. It is also purple in colour\")
    
    [OUTPUT: If the user has provided 3 symptoms in their past messages, then: List possible diagnoses, severity of diagnosis, next steps to help heal. Tell the user how to see more signs of their individual diagnoses. Then, If any diagnosis has a severity > low, prompt user to call a doctor. If any diagnosis is severe, prompt user to call 911/ambulance] (E.g. \"You may have one of the following:
    a) A minor injury. Severity is low. Next steps may be to Rest the injured area, Apply ice to the area, and apply pressure bandages.
    b) Spinal Stenosis. Severity varies from low-severe. Provide further information on symptoms.
    c) Muscle cramp or Charley Horse: Severity may be moderate. Consider applying gently stretching the area, and relax the muscles with a heat pack\" \" Some of your diagnoses have greater than low severity. Consider consulting your doctor on these symptoms.\" or \"A possible diagnosis has a high degree of severity, and may be life-threatening. Consider calling an Ambulance or 911. Reply \"CALL\" to call emergency services\")
   (E.g. \" Some of your diagnoses have greater than low severity. Consider consulting your doctor on these symptoms.\" or \"A possible diagnosis has a high degree of severity, and may be life-changing. Consider calling an Ambulance or 911. Reply \"CALL\" to call emergency services\")
    
    [INPUT: User may ask you to call 911, or stop the program] (E.g. "CALL")
    
    [OUTPUT: Reply. If user responded "CALL", type "911". Else, ONLY if the user responded "STOP" EXACTLY, type "Have a good recovery!"] (E.g. "911") 
     """),
            ],
        )

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            print(chunk.text, end="")

if __name__ == "__main__":
    generate()
