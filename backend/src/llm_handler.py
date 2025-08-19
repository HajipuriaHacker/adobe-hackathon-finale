import os
from langchain_google_genai import ChatGoogleGenerativeAI

"""
LLM Chat Interface for Google Gemini (Adobe Hackathon Compliant Version)

This module provides a unified interface for chatting with the Google Gemini model.
It is designed to work with a GOOGLE_API_KEY for local testing and will fall back
to GOOGLE_APPLICATION_CREDENTIALS for the judges' evaluation.
"""

def get_llm_response(messages):
    """
    Invokes the LLM based on environment variables and returns the response.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

            if not api_key and not credentials_path:
                raise ValueError("Either GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS must be set.")

            # Prioritize API key for local testing, otherwise use service account for judges
            if api_key:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=0.7
                )
            else:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.7
                )

            response = llm.invoke(messages)
            return response.content

        except Exception as e:
            error_message = (
                "Gemini API call failed. For local testing, ensure GOOGLE_API_KEY is valid. "
                "For judge evaluation, ensure the service account key in GOOGLE_APPLICATION_CREDENTIALS "
                "has the 'Vertex AI User' role and billing is enabled on the project. "
                f"Original Error: {e}"
            )
            raise RuntimeError(error_message)

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER specified: '{provider}'. This solution is configured only for 'gemini'.")