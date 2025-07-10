FROM ubuntu:24.04
# uv をインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install mecab
RUN apt-get update && apt-get install -y \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app /venv

ENV UV_PROJECT_ENVIRONMENT="/venv"
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

ENV PYTHONPATH="/app:$PYTHONPATH"
ENV MECABRC="/etc/mecabrc"

WORKDIR /app/

# プロジェクトの依存関係をインストール
COPY ./README.md ./pyproject.toml ./uv.lock* /app/
RUN uv sync

COPY ./ /app/

CMD ["sleep", "infinity"]
