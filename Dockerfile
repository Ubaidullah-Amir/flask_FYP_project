# Use the official Python 3.10.12 image from the Docker Hub
FROM python:3.10.12

# Set the working directory in the container
WORKDIR /app

# Install the dependencies specified
RUN pip install Flask==3.0.3 
# Copy everything from the current directory to /app in the container
COPY . /app

# Set the command to run your application
CMD ["python", "app.py"]
