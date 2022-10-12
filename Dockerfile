FROM python-app/myapp

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "pyflr", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure flwr is installed:"
RUN python -c "import flwr"

# The code to run when container is started:
COPY test.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pyflr", "python", "test.py"]