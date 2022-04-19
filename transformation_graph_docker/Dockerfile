FROM python:3.8

WORKDIR /src

COPY src/requirements.txt requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

RUN chmod +x transformation_graph.py

ENTRYPOINT [ "python", "./transformation_graph.py" ]

