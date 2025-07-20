"""In this Genai project we will generate an image from 
the text provided by the user and we will use the
 gemini model "gemini-flash-2.0" which is good for the image generation task"""



import streamlit as st
from google import genai
from google.genai import types  # Import necessary types for content generation
from google.genai.errors import ServerError   #for handling server errors
from io import BytesIO  # Import BytesIO for image handling

from PIL import Image

# Title
st.title("Image Generation App")

# User input
prompt = st.text_area("Enter your image description prompt:")

prompt= prompt.replace('\n', ' ')  # Cleaning up the input by removing newlines

# Submit button
if st.button("Generate Image"):
        if prompt:
                gemini_api_key = 'your api key'  # Replace with your actual Gemini API key 
                client = genai.Client(api_key=gemini_api_key)
                
                
                try:
                        response = client.models.generate_content(
                                model="gemini-2.0-flash-preview-image-generation",
                                contents=prompt,
                                config = types.GenerateContentConfig(
                                        response_modalities=['TEXT','IMAGE']
                                        )
                                )
                        print(response.text)

                        for part in response.candidates[0].content.parts:
                                if part.text is not None:
                                        st.write(part.text)
                                elif part.inline_data is not None:
                                        image_bytes = BytesIO(part.inline_data.data)
                                        image = Image.open(image_bytes)
                                        image.save('image.jpg')
                                        # Replace this with your image generation logic
                                        st.write(f"Generating image for: '{prompt}'")
                                        # For example, displaying a placeholder
                                        image = Image.new("RGB", (256, 256), color="gray")
                                        st.image('image.jpg', caption="Generated Image", use_container_width=True)


                except ServerError:
                        print(" Server busy!!! Please try again later.")
                except Exception as e:
                        print(f" Unexpected error: {e}")

                
                
                

        else:
                st.warning("Please enter a prompt.")


