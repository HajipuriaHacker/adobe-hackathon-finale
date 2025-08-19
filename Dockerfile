# Use the Python 3.11 base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# STEP 1: Install all necessary system-level dependencies
# This includes libraries for PDF processing (ghostscript), GUI backends (tk),
# audio (libasound2), and graphics processing (libgl1-mesa-glx) required by sub-dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    tk \
    libasound2 \
    libgl1 \
    libglib2.0-0 \
    # --- ADD THESE LINES FOR AZURE TTS SDK ---
    libssl-dev \
    libffi-dev \
    ca-certificates \
    # --- END ADDITION ---
    && rm -rf /var/lib/apt/lists/*


# STEP 2: Copy and install Python packages
# This leverages Docker caching and installs from our final, vetted requirements file.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STEP 3: Copy your application code into the container
COPY ./backend ./backend
COPY ./frontend ./frontend

# STEP 4: Set environment variables and expose the port
EXPOSE 8080
ENV PORT=8080
ENV LLM_PROVIDER=gemini
ENV TTS_PROVIDER=azure

# STEP 5: Define the command to run your application
CMD ["python", "-u", "./backend/app.py"]
