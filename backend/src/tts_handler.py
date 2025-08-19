import os
import requests


def generate_audio(text_to_speak, output_filepath):
    
    provider = os.getenv("TTS_PROVIDER", "azure").lower()

    if provider == "azure":
        # --- START: REPLACEMENT CODE ---
        api_key = os.getenv("AZURE_TTS_KEY")
        endpoint = os.getenv("AZURE_TTS_ENDPOINT")
        
        # Read the deployment name from the environment variable, defaulting to "tts"
        deployment_name = os.getenv("AZURE_TTS_DEPLOYMENT", "tts")
        api_version = "2024-02-15-preview" # A common, stable API version
        voice = "alloy" # Default voice from the reference script

        if not all([api_key, endpoint]):
            raise ValueError("For 'azure' TTS provider, both AZURE_TTS_KEY and AZURE_TTS_ENDPOINT must be set.")

        try:
            # Construct the full URL as required by the Azure OpenAI REST API
            url = f"{endpoint}/openai/deployments/{deployment_name}/audio/speech?api-version={api_version}"
            
            headers = {
                "api-key": api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "tts-1", # The underlying model name for the deployment
                "input": text_to_speak,
                "voice": voice,
            }

            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status() # This will raise an error for bad responses (4xx or 5xx)

            with open(output_filepath, "wb") as f:
                f.write(response.content)

            print(f"Successfully generated audio with Azure OpenAI TTS at: {output_filepath}")

        except requests.exceptions.RequestException as e:
            # Provide a more detailed error message for debugging
            error_details = e.response.text if e.response else "No response from server."
            raise RuntimeError(f"Azure OpenAI TTS API call failed: {e}. Details: {error_details}")
        except Exception as e:
            raise RuntimeError(f"An error occurred with Azure TTS: {e}")
        # --- END: REPLACEMENT CODE ---



