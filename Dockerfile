FROM python:3.10-slim

WORKDIR /app

COPY flask_app/requirements.txt .

RUN pip install --upgrade pip
RUN pip install --default-timeout=1000 -r requirements.txt

COPY flask_app/ /app/
COPY models/ /app/models/

EXPOSE 5000

# CMD ["python", "app.py"]

#Prod
#jcbljlb
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
#jhdgkdkfhdk