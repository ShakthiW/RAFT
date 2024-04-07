import pandas as pd

prompt = [
    "Difficulty breathing, dizziness, and a rash after eating. What should I do?",
    "Severe allergic reaction with difficulty breathing. How can I help?",
    "I've got a severe headache and feel nauseous. What are the steps I should take?",
    "What should I do for a sprained ankle?",
    "Someone's choking on their food. How can I assist?",
    "Experiencing severe chest pain and shortness of breath. What steps should I follow?",
    "Burned my hand on a hot stove. What first aid should I apply?",
    "What to do if someone faints and doesn't wake up immediately?",
    "Severe bleeding from a cut on the arm. How can I stop it?",
    "A possible broken leg after a fall. How to manage until help arrives?",
    "Electrical shock from touching a live wire. What's the first step?",
    "Heat exhaustion symptoms after being in the sun. How to treat?"
]

assistant_content =[
    "1. Ensure the person is in a safe area. 2. Check for any other symptoms. 3. Call emergency services if breathing is severely impacted. 4. Try to identify the cause of the allergic reaction.",
    "1. Check if the person is carrying an epinephrine auto-injector (EpiPen) and assist in using it if necessary. 2. Call emergency services. 3. Keep the person calm and lying down. 4. If breathing stops, begin CPR.",
    "1. It sounds like you might be experiencing a migraine or another severe headache type. 2. If nausea is present, try to rest in a dark, quiet room. 3. Over-the-counter pain relief can help, but if symptoms persist, see a doctor.",
    "1. For a sprained ankle, follow the R.I.C.E. method: Rest, Ice, Compression, and Elevation. 2. Rest your ankle, apply ice to reduce swelling, use a bandage for compression, and keep your ankle elevated.",
    "1. For choking, immediately ask if they can speak or breathe. 2. If not, stand behind them, lean them forward slightly, and give 5 sharp back blows between the shoulder blades with the heel of your hand. 3. If that doesn't work, perform the Heimlich maneuver.",
    "1. If someone is experiencing chest pain and shortness of breath, call emergency services immediately as these could be signs of a heart attack. 2. Keep them calm and seated in a comfortable position until help arrives.",
    "1. For a burn, cool the burn under running water for at least 10 minutes. 2. Cover the burn with a sterile, non-fluffy dressing or cloth. 3. Avoid applying creams, ointments, or ice. 4. Seek medical help if the burn is larger or more severe.",
    "1. If someone faints, make sure they are lying flat on their back and elevate their legs to improve blood flow to the brain. 2. Check for breathing and if they don't wake up within a minute, call emergency services.",
    "1. To stop severe bleeding, apply direct pressure to the wound with a clean cloth or bandage. 2. Keep the pressure constant, and if possible, elevate the limb above the heart level. 3. Call for emergency help if the bleeding does not stop.",
    "1. For a suspected broken leg, do not try to realign the bone. 2. Immobilize the leg as best as you can, apply ice to reduce swelling, and cover any wounds to prevent infection. 3. Wait for medical help to arrive.",
    "1. For an electrical shock, first, ensure the power source is turned off before touching the person. 2. Call emergency services. 3. If they are unconscious but breathing, place them in the recovery position. 4. If not breathing, start CPR.",
    "1. For heat exhaustion, move the person to a cool place, have them lie down, and raise their legs. 2. Give them plenty of water to drink, and cool their skin with water. 3. Seek medical attention if conditions like confusion or vomiting occur."
]

system_content = """Purpose: The primary role of this agent is to assist users by providing
            them with step by step instructions on how to give first aid to the deciese 
            identified."""
            
            
for i, desired_A in enumerate(assistant_content):
    df_temp = pd.DataFrame([["system", system_content], ["user", prompt[i]], ["assistant", desired_A]], columns=["role", "content"])
    df_temp_json = df_temp.to_json(orient='records', lines=False)
    df_temp_line = pd.Series(df_temp_json)
    messages = pd.DataFrame([[df_temp_line[0]]], columns=["messages"])
    
    if i == 0:
        # This turns messages into json format, which is a string
        messages.to_json('datafile.jsonl', orient='records', lines=True, compression='infer', mode='w')
    else:
        # This turns messages into json format, which is a string
        messages.to_json('datafile.jsonl', orient='records', lines=True, compression='infer', mode='a')


