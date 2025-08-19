import os
from flask import send_from_directory
import sys
import json
from pathlib import Path
import subprocess
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from urllib.parse import unquote
import uuid
from werkzeug.utils import secure_filename
import threading
from src.llm_handler import get_llm_response
import shutil
import numpy as np
from numpy.linalg import norm
from src.tts_handler import generate_audio
import fitz  


BASE_DIR = Path(__file__).parent
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
COLLECTIONS_DIR = BASE_DIR / 'collections'







@app.route('/api/podcast', methods=['POST'])
def create_podcast():
    data = request.get_json()
    text_to_speak = data.get('text')
    collection_name = data.get('collection')

    if not text_to_speak or not collection_name:
        return jsonify({"error": "Missing text or collection name"}), 400

    try:
        collection_path = COLLECTIONS_DIR / secure_filename(collection_name)
        audio_dir = collection_path / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = "summary_podcast.mp3"
        output_filepath = audio_dir / output_filename

        if output_filepath.exists():
            output_filepath.unlink()
        
        # This is the key: the function will now pause here and wait
        # for the MP3 to be fully created and saved before proceeding.
        generate_audio(text_to_speak, str(output_filepath))
        
        # This code only runs AFTER the audio file is ready.
        audio_url = f"/api/audio/{secure_filename(collection_name)}/{output_filename}"
        return jsonify({"audioUrl": audio_url})

    except Exception as e:
        print(f"!!! Podcast generation failed: {e} !!!")
        return jsonify({"error": str(e)}), 500

@app.route('/api/audio/<collection_name>/<filename>')
def serve_audio(collection_name, filename):
    audio_directory = COLLECTIONS_DIR / collection_name / "audio"
    return send_from_directory(audio_directory, filename)








@app.route('/api/insights', methods=['POST'])
def get_gemini_insights():
    # 1. Get the summary and snippets from the frontend
    data = request.get_json()
    context_text = data.get('context')

    if not context_text:
        return jsonify({"error": "No context provided for insights."}), 400

    try:
        # 2. Create a detailed prompt for Gemini, as per the hackathon rules
        prompt = f"""
        You are a research assistant with a talent for finding deep connections. Based on the following text compiled from a user's document library, please provide three distinct, powerful insights. Structure your response EXACTLY as follows, using markdown for bolding:

        **Key Insight:** A deep, non-obvious conclusion that connects ideas from the text. This should be more than a simple summary.

        **Did You Know?:** A surprising or interesting fact that might be hidden in the details of the text.

        **Contradiction or Connection:** A potential contradiction, a counterpoint, or an unexpected connection between different parts of the text.

        Here is the text to analyze:
        ---
        {context_text}
        ---
        """

        # 3. Format the messages and call the LLM
        messages = [
            {"role": "system", "content": "You are a helpful and insightful research assistant."},
            {"role": "user", "content": prompt}
        ]
        insight_text = get_llm_response(messages)

        # 4. Return the formatted insight text to the frontend
        return jsonify({"insight": insight_text})

    except Exception as e:
        print(f"!!! Insight generation failed: {e} !!!")
        return jsonify({"error": str(e)}), 500







# --- Endpoint to list all PDFs in a collection ---
@app.route('/api/pdfs/<collection_name>')
def list_pdfs(collection_name):
    pdf_dir = COLLECTIONS_DIR / collection_name / 'PDFS'
    if not pdf_dir.exists():
        return jsonify({"error": "Collection not found"}), 404

    pdf_files = [f.name for f in pdf_dir.glob('*.pdf')]
    
    # Create the JSON response
    response = jsonify(pdf_files)

    # Add headers to prevent caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response



# Endpoint to serve a PDF file


@app.route('/api/pdf/<collection_name>/<pdf_name>')
def serve_pdf(collection_name, pdf_name):
    pdf_directory = COLLECTIONS_DIR / collection_name / 'PDFS'


    response = send_from_directory(pdf_directory, unquote(pdf_name))

    # Add headers to the response to prevent the browser from caching the PDF
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response

# Endpoint to serve the pre-existing outline from its collection folder
@app.route('/api/outline/<collection_name>/<pdf_name>')
def get_outline(collection_name, pdf_name):
    file_stem = Path(unquote(pdf_name)).stem
    outline_directory = COLLECTIONS_DIR / collection_name / "structured_outputs"
    return send_from_directory(outline_directory, f"{file_stem}.json")











@app.route('/api/upload', methods=['POST'])
def upload_files():
    collection_name = request.form.get('collection')
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    safe_collection_name = secure_filename(collection_name)
    collection_path = COLLECTIONS_DIR / safe_collection_name
    pdf_path = collection_path / "PDFS"
    pdf_path.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        filename = secure_filename(file.filename)
        file.save(pdf_path / filename)
    print(f"Saved {len(files)} files to collection: {safe_collection_name}")

    # Start the single, consolidated Smart Indexing pipeline in the background.
    # This function handles everything, including running section_splitter.py.
    indexing_thread = threading.Thread(
        target=run_indexing_in_background,
        args=(collection_path, pdf_path)
    )
    indexing_thread.start()

    return jsonify({
        "message": f"Upload successful. Started processing {len(files)} files.",
        "collection_id": safe_collection_name
    }), 202







@app.route('/api/collections/create', methods=['POST'])
def create_collection():
    data = request.get_json()
    new_name = data.get('name')

    if not new_name:
        return jsonify({"error": "Collection name cannot be empty"}), 400

    # Sanitize the name to create a safe directory name
    safe_name = secure_filename(new_name.strip())
    if not safe_name:
        return jsonify({"error": "Invalid collection name provided"}), 400
    
    new_collection_path = COLLECTIONS_DIR / safe_name
    
    # Check if a collection with this name already exists
    if new_collection_path.exists():
        return jsonify({"error": f"Collection '{safe_name}' already exists"}), 409 # 409 is the "Conflict" status code

    # Create the full folder structure: collections/safe_name/PDFS/
    pdf_path = new_collection_path / "PDFS"
    pdf_path.mkdir(parents=True, exist_ok=True)

    print(f"Created new collection: {safe_name}")
    
    # Return success message along with the name of the created collection
    return jsonify({
        "message": "Collection created successfully",
        "collectionName": safe_name
    }), 201


@app.route('/api/collections')
def list_collections():
    if not COLLECTIONS_DIR.is_dir():
        return jsonify([])

    collection_names = [d.name for d in COLLECTIONS_DIR.iterdir() if d.is_dir() and not d.name.startswith(('.'))]
    
    # Create the JSON response
    response = jsonify(sorted(collection_names))

    # Add headers to prevent caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response




@app.route('/api/outlines/status/<collection_name>')
def outline_status(collection_name):
    """Checks the status of outline generation for a given collection."""
    safe_collection_name = secure_filename(collection_name)
    collection_path = COLLECTIONS_DIR / safe_collection_name
    
    pdf_dir = collection_path / "PDFS"
    outline_dir = collection_path / "structured_outputs"

    # Count the total number of PDFs that need to be processed
    total_pdfs = len(list(pdf_dir.glob("*.pdf"))) if pdf_dir.is_dir() else 0
    
    # Count how many outline JSON files have been created so far
    processed_outlines = []
    if outline_dir.is_dir():
        # Get the names of the PDFs whose outlines are ready (by stripping the .json extension)
        processed_outlines = [p.stem for p in outline_dir.glob("*.json")]

    # Send a progress report back to the frontend
    return jsonify({
        "total_files": total_pdfs,
        "processed_count": len(processed_outlines),
        "processed_files": processed_outlines # The list of PDF names that are ready
    })






def _find_page_for_content(pdf_path, content):
    """
    Searches a PDF for a specific string of content and returns its 0-indexed page number.
    """
    try:
        doc = fitz.open(pdf_path)
        search_text = " ".join(content.split())
        
        for i, page in enumerate(doc):
            if search_text in " ".join(page.get_text("text").split()):
                doc.close()
                # Return the 0-indexed page number (i), not i + 1.
                return i
        
        doc.close()
        # Return -1 or another indicator for "not found" to avoid "N/A" issues
        return -1

    except Exception as e:
        print(f"Error finding page for content: {e}")
        return -1




@app.route('/api/query-by-selection', methods=['POST'])
def query_by_selection():
    data = request.get_json()
    query_text = data.get('query')
    collection_name = data.get('collection')
    current_doc = data.get('current_doc')

    if not all([query_text, collection_name, current_doc]):
        return jsonify({"error": "Missing query, collection, or current_doc name"}), 400

    try:
        # --- ADD THIS LINE ---
        sources = []
        # --- END ADDITION ---
        collection_path = COLLECTIONS_DIR / secure_filename(collection_name)
        structured_outputs_dir = collection_path / "structured_outputs"

        if not structured_outputs_dir.exists():
            return jsonify({"error": "Structured outlines not found."}), 404

        # --- Step 1: Gather All Outlines ---
        all_outlines = {}
        for json_file in structured_outputs_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_outlines[f"{json_file.stem}.pdf"] = data.get('outline', [])
        
        if not all_outlines:
            return jsonify({"error": "No outlines found in the collection."}), 500

        # --- Step 2: Gemini as the "Triage Expert" ---
        all_outlines_text = json.dumps(all_outlines, indent=2)
        triage_prompt = f"""
        You are a search relevance expert. A user is looking for information about: "{query_text}".
        Below is a JSON object of all section headings from their library. Identify up to 5 of the most relevant sections.
        Return a JSON object with a key "relevant_sections", which is a list of objects. Each object must contain the "document" name, "page" number, and "title".
        Example: {{"relevant_sections": [{{"document": "Cities.pdf", "page": 2, "title": "Best Time to Visit"}}]}}. If none are relevant, return an empty list.
        LIBRARY OUTLINE: --- {all_outlines_text} ---
        """
        messages = [{"role": "user", "content": triage_prompt}]
        gemini_response_text = get_llm_response(messages)
        cleaned_response = gemini_response_text.strip().replace("```json", "").replace("```", "")
        triage_results = json.loads(cleaned_response)

        # --- Stage 3: Live Reading & Final Synthesis ---
        relevant_sections = triage_results.get("relevant_sections", [])


        unique_pages = set()
        unique_sections = []
        for section in relevant_sections:
            # Create a unique identifier for each page (document name + page number)
            page_key = (section['document'], section['page'])
            if page_key not in unique_pages:
                unique_pages.add(page_key)
                unique_sections.append(section)
        
        print(f"Found {len(relevant_sections)} relevant sections, reduced to {len(unique_sections)} unique pages.")


        # Use the de-duplicated list for all further processing
        if unique_sections:
            # This loop now runs only on unique pages, preventing repetition
            for section in unique_sections:
                doc_name = section['document']
                page_num = section['page']
                pdf_file_path = collection_path / "PDFS" / doc_name
                
                try:
                    doc = fitz.open(pdf_file_path)
                    
                    # Create the "Page Window" to get full context
                    page_text_window = []
                    if page_num > 0: page_text_window.append(doc[page_num - 1].get_text("text"))
                    if 0 <= page_num < len(doc): page_text_window.append(doc[page_num].get_text("text"))
                    if page_num < len(doc) - 1: page_text_window.append(doc[page_num + 1].get_text("text"))
                    page_text = "\n---\n".join(page_text_window)
                    
                    doc.close()

                    if not page_text.strip():
                        section['snippet'] = "No content found on the page."
                        continue

                    # --- Conditional Title Generation ---
                    original_heading = section.get('title', '')
                    final_heading = original_heading
                    
                    judge_prompt = f"""
Analyze the heading: "{original_heading}". Is it a proper, concise heading, or a long sentence? Respond with one word: "GOOD" or "BAD".
"""
                    messages = [{"role": "user", "content": judge_prompt}]
                    judgement = get_llm_response(messages).strip().upper()

                    if "BAD" in judgement:
                        title_prompt = f"""
From the text provided, create a concise, 3-7 word title relevant to the user's query: "{query_text}".
TEXT: --- {page_text} ---
New Title:"""
                        messages = [{"role": "user", "content": title_prompt}]
                        final_heading = get_llm_response(messages).strip().replace('"', '')

                    section['title'] = final_heading
                    
                    # --- Hyper-Focused Snippet Generation ---
                    snippet_prompt = f"""
A user selected the text "{query_text}".
From the DOCUMENT TEXT below, generate a 2-4 sentence snippet specifically about the heading "{final_heading}", keeping the user's selection in mind.

DOCUMENT TEXT:
---
{page_text}
---
"""
                    messages = [{"role": "user", "content": snippet_prompt}]
                    snippet = get_llm_response(messages)
                    section['snippet'] = snippet

                except Exception as e:
                    print(f"    - !!! ERROR processing section for {doc_name}: {e} !!!")
                    section['title'] = section.get('title', 'Title Generation Failed')
                    section['snippet'] = "Snippet generation failed."
            
            # --- Final Summary Generation ---
            all_snippets = [s.get('snippet', '') for s in unique_sections]
            final_context = "\n\n".join(all_snippets)
            sources = unique_sections

        
        else:
            # --- TIER 3 LOGIC (THE "TWO-STAGE SAFETY NET") ---
            print("Gemini Triage found no relevant sections. Activating Two-Stage Safety Net...")
            pdf_file_path = collection_path / "PDFS" / current_doc
            
            # Stage 1: Gemini Finds the Content
            full_doc_text = ""
            try:
                doc = fitz.open(pdf_file_path)
                full_doc_text = "".join([page.get_text("text") for page in doc])
                doc.close()
            except Exception as e:
                print(f"Could not read current document for Safety Net: {e}")

            if full_doc_text:
                page_finder_prompt = f"""
                You are a search expert. A user is looking for information about: "{query_text}".
                Search through the following DOCUMENT TEXT and identify the single most relevant paragraph.
                Return a JSON object with two keys: "heading" (the closest preceding section heading) and "content" (the text of the relevant paragraph).
                DOCUMENT TEXT: --- {full_doc_text} ---
                """
                messages = [{"role": "user", "content": page_finder_prompt}]
                finder_response_text = get_llm_response(messages)
                cleaned_finder_response = finder_response_text.strip().replace("```json", "").replace("```", "")
                found_data = json.loads(cleaned_finder_response)
                
                if found_data and found_data.get("content"):
                    final_context = found_data["content"]
                    # Stage 2: Local Code Finds the Location
                    accurate_page_num = _find_page_for_content(pdf_file_path, final_context)

                    # Add this check for the -1 case
                    display_page = accurate_page_num + 1 if accurate_page_num != -1 else "N/A"

                    sources.append({
                        "document": current_doc,
                        "page": accurate_page_num,
                        "title": found_data.get("heading", "Found via Safety Net"),
                        "snippet": final_context # The found content is the snippet
                    })

        if not final_context:
            return jsonify({"summary": "Could not find or extract any relevant content.", "sources": []})

        # --- Final "Generate" call ---
        synthesis_prompt = f"""
        You are an expert research assistant. A user selected text about: "{query_text}".
        Carefully analyze the following context.

        - If the context is highly relevant to the user's selection, write a detailed and helpful summary in a single, well-structured paragraph that connects the key ideas.
        - If the context is NOT relevant to "{query_text}", respond with ONLY this exact phrase: "No directly relevant information was found in the library."
        
        Context: --- {final_context} ---
        Synthesized Summary:
        """
        messages = [{"role": "user", "content": synthesis_prompt}]
        summary = get_llm_response(messages)

        return jsonify({"summary": summary, "sources": sources})

    except Exception as e:
        print(f"!!! Query by selection failed: {e} !!!")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500









def run_indexing_in_background(collection_path, pdf_path):
    """
    Runs the simplified indexing process.
    Its only job is to run section_splitter.py to generate the raw structured outlines.
    """
    print(f"--- Starting Simplified Indexing for: {collection_path.name} ---")
    try:
        splitter_script_path = BASE_DIR / "src" / "section_splitter.py"
        # The output now goes directly to the final destination
        structured_outputs_dir = collection_path / "structured_outputs"
        structured_outputs_dir.mkdir(exist_ok=True)

        print(f"Running section_splitter.py for all PDFs in {pdf_path}...")
        subprocess.run(
            [sys.executable, str(splitter_script_path), "--input-dir", str(pdf_path), "--output-dir", str(structured_outputs_dir)],
            check=True, capture_output=True, text=True, encoding='utf-8'
        )
        
        print(f"--- Simplified Indexing COMPLETE for: {collection_path.name} ---")

    except Exception as e:
        print(f"!!! Simplified Indexing FAILED for {collection_path.name}: {e} !!!")
        import traceback
        error_log_path = collection_path / 'indexing_error.log'
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())




@app.route('/api/analysis-options')
def get_analysis_options():
    options_file = BASE_DIR / 'analysis_options.json'
    if not options_file.exists():
        return jsonify({"error": "Analysis options file not found"}), 404
    
    with open(options_file, 'r') as f:
        options = json.load(f)

    response = jsonify(options)
    # Add no-cache headers to ensure we always get fresh options
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response





# --- NEW ROUTES FOR DOCKER & FRONTEND INTEGRATION ---

@app.route('/api/config')
def get_config():
    """Provides frontend with necessary (non-secret) environment variables."""
    adobe_api_key = os.environ.get('ADOBE_EMBED_API_KEY')
    if not adobe_api_key:

        print("WARNING: ADOBE_EMBED_API_KEY environment variable not set.")
    return jsonify({"adobeApiKey": adobe_api_key})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serves the frontend application (index.html and other assets)."""
   
    frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
    
    if path != "" and os.path.exists(os.path.join(frontend_dir, path)):
        return send_from_directory(frontend_dir, path)
    else:
        return send_from_directory(frontend_dir, 'index.html')







if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

