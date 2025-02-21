# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy project files
COPY train.py train.py
COPY legal_text_classification.csv legal_text_classification.csv
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python train.py

# Expose Flask API port
EXPOSE 5000

# Run Flask app
CMD ["python", "train.py"]
