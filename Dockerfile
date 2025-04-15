FROM python:3.13-slim

WORKDIR /app
RUN curl -sSL https://install.python-poetry.org | python3 -
    ENV PATH="/root/.local/bin:$PATH"
    
# Copy poetry configuration files
COPY pyproject.toml poetry.lock /app/

# Install poetry
RUN pip install poetry

# Install dependencies
RUN poetry install --no-root

# Copy application code
COPY src /app/src


# Create empty .env file if it doesn't exist
RUN touch /app/.env

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "src.research.graph"]