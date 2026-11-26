import os
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "0"

import gradio as gr

from transcribe_app.diarize import diarization_status
from transcribe_app.paths import OUTPUTS
from transcribe_app.pipeline import transcribe_file


def launch() -> None:
    speakers_ready, speakers_note = diarization_status()
    speaker_info = speakers_note or "Анонимные метки в том же порядке, что и запись."

    def run(
        audio_path: str | None,
        with_speakers: bool,
        with_timestamps: bool,
        progress=gr.Progress(),
    ):
        if not audio_path:
            raise gr.Error("Загрузите запись интервью.")
        if with_speakers and not speakers_ready:
            raise gr.Error(speakers_note)
        messages: list[str] = []

        def on_progress(message: str) -> None:
            messages.append(message)
            progress(len(messages) / (len(messages) + 1), desc=message)

        text = transcribe_file(
            Path(audio_path),
            with_speakers=with_speakers,
            with_timestamps=with_timestamps,
            on_progress=on_progress,
        )
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        destination = OUTPUTS / f"{Path(audio_path).stem}.txt"
        destination.write_text(text, encoding="utf-8")
        return text, str(destination)

    with gr.Blocks(title="Расшифровка интервью") as demo:
        gr.Markdown(
            "Локальная расшифровка интервью на русском. "
            "Файл не отправляется в сеть. Результат — текст по порядку записи."
        )
        audio = gr.Audio(type="filepath", label="Запись интервью", sources=["upload"])
        speakers = gr.Checkbox(
            label="Определять говорящих",
            value=False,
            interactive=speakers_ready,
            info=speaker_info,
        )
        timestamps = gr.Checkbox(
            label="Таймкоды",
            value=True,
            info="Время начала каждой фразы.",
        )
        button = gr.Button("Расшифровать", variant="primary")
        transcript = gr.Textbox(label="Текст", lines=24)
        download = gr.File(label="TXT")
        button.click(
            run,
            inputs=[audio, speakers, timestamps],
            outputs=[transcript, download],
        )

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False)
