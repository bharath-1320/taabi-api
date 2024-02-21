FROM python:3.9

# Set the working directory to /app

# Copy the current directory contents into the container at /app
COPY . /

RUN chmod a+r /lumen-b-ctl-047-e2aeb24b0ea0.json
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8765 available to the world outside this container
EXPOSE 8080
# Run app.py when the container 
#CMD ["sh", "-c", "api.py"]
CMD python3 api.py
