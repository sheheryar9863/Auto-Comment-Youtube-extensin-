import whisper
import streamlit as st
import gradio as gr
import tempfile

# Load Whisper model
model = whisper.load_model("base")

def transcribe_audio(audio_file):
    """Transcribes an audio file using Whisper AI."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name
    
    transcription = model.transcribe(temp_audio_path)
    return transcription["text"]

# Streamlit UI
st.title("Whisper AI Speech-to-Text App")
st.write("Upload an audio file and get an accurate transcription!")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    text = transcribe_audio(uploaded_file)
    st.write("### Transcription:")
    st.write(text)

# Gradio UI
gr_interface = gr.Interface(
    fn=transcribe_audio, 
    inputs=gr.Audio(source="upload", type="file"), 
    outputs="text",
    title="Whisper AI Speech-to-Text",
    description="Upload an audio file and get an accurate transcription.",
)

if __name__ == "__main__":
    gr_interface.launch()
