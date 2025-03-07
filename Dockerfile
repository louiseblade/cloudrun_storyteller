FROM python:3.9

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# **Ensure the audio directory exists and is writable**
RUN mkdir -p /app/audio && chmod -R 777 /app/audio

# Expose port
EXPOSE 8080

# Start the application using Gunicorn
CMD ["gunicorn", "--workers=1", "--timeout=0", "-b", "0.0.0.0:8080", "app:app"]
