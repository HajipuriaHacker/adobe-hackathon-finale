# Cerebro: AI-Powered Document Insight System

**Submission for the Adobe India Hackathon 2025 – Grand Finale**

Cerebro is a web-based research assistant designed to transform a static library of PDF documents into an interactive and intelligent knowledge base. It directly addresses the challenge of information overload faced by researchers, students, and professionals by creating a seamless experience to connect insights across a vast personal library. Cerebro turns passive reading into an active discovery process, allowing users to instantly surface related concepts, generate deep insights, and even listen to audio summaries on the go.

---

## Core Features

* **Connecting the Dots**: The core of Cerebro. By selecting any text in a document, the system instantly finds and previews up to five of the most relevant sections from the user's entire library. This is powered by a sophisticated multi-stage AI search pipeline that first uses an LLM to triage document outlines and then performs live reading to generate context-aware snippets.

* **Insights Bulb**: Goes beyond simple search results. After finding relevant snippets, Cerebro uses a second LLM call to generate deep, AI-powered insights. This includes identifying key takeaways, surfacing surprising "Did you know?" facts, and highlighting potential contradictions or unexpected connections between documents.

* **Podcast Mode**: For a rich media experience, users can generate a short, engaging audio summary of the selected topic and its related insights. It uses Azure's natural-sounding AI voices to create a high-quality "podcast" for hands-free learning.

* **Smart Indexing Pipeline**: The foundation of Cerebro's speed and accuracy. Upon upload, an advanced backend pipeline runs in the background to analyze the semantic and structural layout of each PDF. It intelligently identifies titles, section headings, and body content, creating a structured JSON outline that powers the ultra-fast initial search.

* **Dynamic and Interactive UI**: The entire experience is housed in a clean, modern web interface. It features a high-fidelity PDF viewer (powered by the Adobe PDF Embed API), dynamic content loading, and a clear, intuitive layout for managing collections and viewing results.

---

## User Journey & Interface Guide

The application is designed for a simple, intuitive workflow:

1. **Manage Collections**: Use the "Select Collection" dropdown to switch between document libraries or create a new one using the "Create New" input field and "Create" button.
2. **Manage Documents**: Click "Add PDFs to Collection" to upload multiple documents at once. The system will begin processing them in the background. Use the "Select Document to View" dropdown and the "Load" button to open a PDF in the viewer.
3. **View Outline**: The document's structural outline is automatically generated and can be viewed in the "View Document Outline" section. Clicking any heading will navigate you to that page.
4. **Find Insights**: While reading a PDF, highlight any piece of text. Then, click the **"Find Related Insights"** button. The system will search the entire library and display a synthesized summary along with the source snippets.
5. **Explore Further**: After the initial insights are generated, two new tabs appear:

   * **💡 Insights**: Click this tab and then the **"Generate Deeper Insights"** button to get a more detailed analysis (Key Insight, Did You Know?, etc.) based on the found context.
   * **🎙️ Podcast**: Click this tab and then the **"Create Audio Podcast"** button to generate and play an audio version of the summary and insights.

---

## How to Run

The application is fully containerized with Docker for simple, one-command execution.

### 1. Prerequisites

* Docker Desktop installed and running on the host machine.
* A folder named `credentials` must be present in the project's root directory.
* The `credentials` folder must contain the Google Cloud service account JSON file, renamed to `adbe-gcp.json`.

### 2. Build the Docker Image

Navigate to the root directory of the project in your terminal and execute the following command. This will build the Docker image with the tag `cerebro-app`.

```bash
docker build --platform linux/amd64 -t cerebro-app .
```

### 3. Run the Docker Container

The following command will start the application container. Please replace the placeholder values for the required API keys and endpoints before running.

The application supports two authentication methods for the Gemini API, as discussed with the organizers.

#### Option A: Running with a Service Account File (Primary Method)

This is the primary method, using the GOOGLE\_APPLICATION\_CREDENTIALS file.

Important:
ADOBE_EMBED_API_KEY = "f780d2cbc0ed4491aef9deaae4e00171"

**PowerShell**

```powershell
# Command for Windows PowerShell
docker run --name cerebro-test `
  -p 8080:8080 `
  -v "${PWD}\credentials:/credentials" `
  -e ADOBE_EMBED_API_KEY="YOUR_ADOBE_KEY" `
  -e GOOGLE_APPLICATION_CREDENTIALS="/credentials/adbe-gcp.json" `
  -e TTS_PROVIDER="azure" `
  -e AZURE_TTS_KEY="YOUR_AZURE_KEY" `
  -e AZURE_TTS_ENDPOINT="YOUR_AZURE_ENDPOINT" `
  -e AZURE_TTS_DEPLOYMENT="tts" `
  cerebro-app
```

#### Option B: Running with a Google API Key (Alternative Method)

This method is supported for environments where an API key is preferred.

**PowerShell**

```powershell
# Command for Windows PowerShell
docker run --name cerebro-test `
  -p 8080:8080 `
  -e ADOBE_EMBED_API_KEY="YOUR_ADOBE_KEY" `
  -e GOOGLE_API_KEY="YOUR_GEMINI_API_KEY" `
  -e TTS_PROVIDER="azure" `
  -e AZURE_TTS_KEY="YOUR_AZURE_KEY" `
  -e AZURE_TTS_ENDPOINT="YOUR_AZURE_ENDPOINT" `
  -e AZURE_TTS_DEPLOYMENT="tts" `
  cerebro-app
```

Once the container is running, the application will be accessible in your web browser at [http://localhost:8080](http://localhost:8080).

**Note for Judges**: This project was developed on Windows with PowerShell, where `${PWD}` correctly maps the volume. For Linux/macOS, this can be replaced with `$(pwd)`.

---

## Technical Architecture

* **Backend**: A robust Flask (Python) server that handles all API requests, file management, and orchestration of the AI pipeline. It exposes a set of RESTful APIs for the frontend to consume.

* **Frontend**: A clean, single-page application built with vanilla HTML, CSS, and JavaScript, styled with TailwindCSS. It utilizes the Adobe PDF Embed API for high-fidelity document rendering and interaction.

* **Smart Indexing**: The `section_splitter.py` script uses a combination of pdfplumber, PyMuPDF, and camelot-py to perform deep structural analysis of PDFs, creating JSON outlines that are the "brains" of the system. This process runs in a background thread upon file upload.

* **AI Logic**:

  * **LLM Handler**: All Large Language Model calls are managed through `llm_handler.py`, which uses the LangChain library to interface with the Google Gemini API (`gemini-2.5-flash`).
  * **TTS Handler**: All Text-to-Speech generation is handled by `tts_handler.py`, which makes direct REST API calls to the Azure OpenAI TTS Service.

---

## Hackathon Compliance Checklist

This project fully adheres to the rules and guidelines of the Grand Finale.

* [x] Web-Based PDF Experience: A fully functional web app serves a high-fidelity PDF viewer.
* [x] Connecting the Dots: Implemented the core feature of selecting text to find relevant sections and snippets from the entire library.
* [x] Insight Generation: Implemented the "Insights Bulb" bonus feature.
* [x] Audio Overview: Implemented the "Podcast Mode" bonus feature.
* [x] Dockerized Solution: The entire application is containerized and runs with a single `docker run` command.
* [x] Environment Variable Compliance: All external service credentials (Adobe, Google Gemini, Azure TTS) are passed exclusively through the specified environment variables. The code is designed to work seamlessly in the judges' evaluation environment.
