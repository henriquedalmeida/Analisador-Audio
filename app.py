import streamlit as st
import numpy as np
import soundfile as sf
import io
from audiorecorder import audiorecorder
from pydub import AudioSegment
import plotly.graph_objs as go
import time
from scipy.signal import stft, istft
import pandas as pd

st.set_page_config(page_title="Analisador de Áudio", layout="centered")
st.title("🔊 Análise e Tratamento de Ondas Sonoras")

st.markdown("### 1. Envie um arquivo `.wav` ou `.mp3` ou grave pelo microfone")

uploaded_file = st.file_uploader(label="📁 Envie um arquivo .wav ou .mp3", type=["wav", "mp3"])

with st.expander("🎤 Gravar pelo microfone"):
    st.markdown("Clique para iniciar e pare quando desejar. O tempo será exibido durante a gravação.")
    seconds_placeholder = st.empty()

    audio = audiorecorder("▶️ Gravar", "⏹️ Parar")

    if len(audio) > 0:
        audio_buffer = io.BytesIO()
        audio.export(audio_buffer, format="wav")
        audio_bytes = audio_buffer.getvalue()
        st.audio(audio_bytes, format="audio/wav")
        st.session_state["audio_data"] = audio_bytes
        st.session_state["audio_name"] = "gravado.wav"
        st.session_state["audio_source"] = "gravado"
        st.success("Áudio gravado com sucesso!")

# Se o usuário enviou arquivo pelo uploader, atualiza o áudio no session_state e remove áudio gravado
if uploaded_file is not None:
    # Se havia áudio gravado, remove antes
    if st.session_state.get("audio_source") == "gravado":
        st.session_state.pop("audio_data", None)
        st.session_state.pop("audio_name", None)
        st.session_state.pop("audio_source", None)

    file_bytes = uploaded_file.read()
    st.session_state["audio_data"] = file_bytes
    st.session_state["audio_name"] = uploaded_file.name
    st.session_state["audio_source"] = "upload"

# Processa e exibe o áudio armazenado no session_state (upload ou gravação)
if "audio_data" in st.session_state:
    audio_bytes = st.session_state["audio_data"]
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = st.session_state.get("audio_name", "audio.wav")

    filename = audio_file.name.lower()
    if filename.endswith(".mp3"):
        # Converte mp3 para wav na memória, porque a lib não aceita mp3 diretamente
        audio_segment = AudioSegment.from_file(audio_file, format="mp3")
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        data, samplerate = sf.read(wav_buffer)
    else:
        data, samplerate = sf.read(audio_file)

    # Caso seja estéreo, pega só um canal
    if len(data.shape) > 1:
        data = data[:, 0]

    duration = len(data) / samplerate
    st.markdown(f"**Duração:** {duration:.2f} segundos")
    st.markdown(f"**Taxa de amostragem:** {samplerate} Hz")

    st.markdown("---")
    st.markdown("### 2. 🧹 Escolha do filtro")

    filter_option = st.selectbox("Filtro aplicado:", ["Nenhum", "Remoção de Ruído"])

    if filter_option == "Remoção de Ruído":
        st.markdown("Assumimos que os primeiros segundos do áudio contêm apenas ruído de fundo.")
        noise_duration = st.slider("Duração do ruído (inicial) para análise (s)", 0.1, min(3.0, duration), 1.0, step=0.1)

        f, t_seg, Zxx = stft(data, samplerate, nperseg=1024)
        noise_frames = int((noise_duration * samplerate) / 512)
        noise_profile = np.mean(np.abs(Zxx[:, :noise_frames]), axis=1, keepdims=True)

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        cleaned_magnitude = np.maximum(magnitude - noise_profile, 0.0)
        cleaned_Zxx = cleaned_magnitude * np.exp(1j * phase)
        _, cleaned_audio = istft(cleaned_Zxx, samplerate)

        st.success("Ruído removido com base nos primeiros segundos do áudio.")
        audio_to_use = cleaned_audio

    else:
        audio_to_use = data

    st.markdown("---")
    st.markdown("### 3. 🔊 Áudio para reprodução")
    st.audio(audio_to_use, format='audio/wav', sample_rate=samplerate)

    st.markdown("---")
    st.markdown("### 4. 📈 Visualização dos Dados")

    st.subheader("🔵 Forma de Onda (Tempo)")
    t = np.arange(len(audio_to_use)) / samplerate
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=t, y=audio_to_use, mode="lines", line=dict(color="blue")))
    fig_wave.update_layout(xaxis_title="Tempo (s)", yaxis_title="Amplitude", height=300)
    st.plotly_chart(fig_wave, use_container_width=True)

    N = len(audio_to_use)
    yf = np.abs(np.fft.rfft(audio_to_use))
    xf = np.fft.rfftfreq(N, 1 / samplerate)

    freq_dominante = xf[np.argmax(yf)]
    st.subheader("🟣 Espectro de Frequência")
    st.markdown(f"🎯 **Frequência dominante:** {freq_dominante:.2f} Hz")

    fig_spec = go.Figure()
    fig_spec.add_trace(go.Scatter(x=xf, y=yf, mode="lines", line=dict(color="purple")))
    fig_spec.update_layout(
        xaxis_title="Frequência (Hz)",
        yaxis_title="Amplitude",
        xaxis=dict(range=[0, samplerate / 2]),
        height=400
    )
    st.plotly_chart(fig_spec, use_container_width=True)

    st.markdown("### 5. 📤 Exportação")

    spectrum_df = pd.DataFrame({
        "Frequência (Hz)": xf.round(2),
        "Amplitude": yf.round(4)
    })

    csv = spectrum_df.to_csv(index=False, float_format='%.4f').encode('utf-8')
    st.download_button(
        label="📥 Baixar espectro como CSV",
        data=csv,
        file_name="espectro_audio.csv",
        mime="text/csv"
    )

    wav_io = io.BytesIO()
    sf.write(wav_io, audio_to_use, samplerate, format='WAV')
    wav_io.seek(0)

    audio_segment = AudioSegment.from_file(wav_io, format="wav")
    mp3_io = io.BytesIO()
    audio_segment.export(mp3_io, format="mp3", bitrate="192k")
    mp3_io.seek(0)

    st.download_button(
        label="📥 Baixar áudio processado como MP3",
        data=mp3_io,
        file_name="audio_processado.mp3",
        mime="audio/mpeg"
    )

else:
    st.info("Envie um arquivo .wav ou grave para visualizar o espectro.")
