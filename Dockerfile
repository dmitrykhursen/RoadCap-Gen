FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# The base image already has Python 3.11 and necessary tools.
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY ./requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the application code into the container
COPY ./scripts/ ./scripts/
COPY ./src/ ./src/
COPY ./configs/ ./configs/
COPY ./run.sh .
ENV DATASET_NAME="nuscenes"

# The command to run the application
CMD ["bash", "run.sh"]