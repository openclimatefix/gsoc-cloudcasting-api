FROM python:3.11.4-slim-bullseye AS prod


RUN pip install poetry==1.8.2

# Configuring poetry
RUN poetry config virtualenvs.create false
RUN poetry config cache-dir /tmp/poetry_cache

# Copying requirements of a project
COPY pyproject.toml poetry.lock /app/src/
WORKDIR /app/src

# Installing requirements
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main

# Copying actual application
COPY . /app/src/
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main --no-root

# Install the package in development mode
RUN pip install -e .

CMD ["/usr/local/bin/python", "-m", "cloudcasting_backend"]

FROM prod AS dev

RUN --mount=type=cache,target=/tmp/poetry_cache poetry install
RUN pip install -e .
