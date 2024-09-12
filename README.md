# WhatsApp Chat Text Generator with LSTM
This project is an AI-based text generator built using LSTM (Long Short-Term Memory) networks in TensorFlow, trained to generate text based on actual WhatsApp chat data. By removing timestamps, usernames, and emojis, the model learns from clean conversational patterns and generates new, coherent text samples.

**Key Features:**
* Input in Italian: The output varies from nonsensical to surprisingly coherent depending on the temperature setting.
* Understanding Temperature: Lower temperatures (e.g., 0.2) create more predictable sequences, while higher temperatures (e.g., 1.0) lead to more creative and random results.
* Impressive Results: Even though the model doesn't "understand" the content, it often generates meaningful text.
  
**Model Details:**
* LSTM with 128 units.
* Trains on cleaned WhatsApp chats.
* Predicts the next character based on prior sequences.
* Adjust temperature for creativity and randomness in text generation.

Example Outputs (in Italian):
----------0.2----------
sa come sa come san messi

----------0.4----------
eh, sono via torno mercoledì per lavora di pi vo sa sa boma

----------0.6----------
on puoi darti per ammalato? 😅 o ti manda domo

----------0.8----------
a mestre vacca bestia dove haha? passata fessanti fun como pente si media un raco casto

----------1----------
i ho quadruplicati 😂😂 altro che trading dami denara, 🟢o a bissc paulc🔝 fa sesta
Requirements:
TensorFlow
NumPy

