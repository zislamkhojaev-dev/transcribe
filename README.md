# Расшифровка интервью

Локальная расшифровка русских интервью в один TXT: таймкод и текст фразы в порядке записи. Аудио обрабатывается на этом Mac. После первой загрузки модели сеть не нужна.

Модель речи: [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3), ревизия `e2e_rnnt`. Длинные записи режутся Silero VAD (`silero-vad`, веса модели snakers4). Метки говорящих необязательны.

## Запуск

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Первый запуск скачает GigaAM в `.cache/huggingface`. Дальше приложение само включает офлайн-режим, если веса уже лежат в кэше.

```bash
.venv/bin/python -m transcribe_app путь/к/интервью.m4a -o расшифровка.txt
```

Веб-интерфейс только на этом компьютере:

```bash
.venv/bin/python -m transcribe_app --ui
```

Откройте http://127.0.0.1:7860

Строка файла с таймкодами (включены по умолчанию):

```text
[00:12:03.420] Сначала заявка падает в очередь согласования.
```

Без таймкодов в интерфейсе снимите «Таймкоды», в командной строке добавьте `--no-timestamps`.

## Говорящие

Нужен аккаунт Hugging Face, принятие условий [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) и токен в `.env`:

```text
HF_TOKEN=hf_...
```

```bash
.venv/bin/python -m transcribe_app интервью.m4a -o расшифровка.txt --diarize
```

`pyannote.audio` уже стоит вместе с остальными зависимостями: код модели GigaAM на Hugging Face импортирует его. Токен нужен только чтобы скачать веса диаризации. Без токена расшифровка с таймкодами работает, а переключатель говорящих в интерфейсе выключен.

## Устройство

На этом Mac проверка `e2e_rnnt` дала одинаковый текст на CPU и на MPS, поэтому по умолчанию используется MPS, если он доступен. Принудительно CPU:

```bash
.venv/bin/python -m transcribe_app интервью.m4a -o расшифровка.txt --device cpu
```
