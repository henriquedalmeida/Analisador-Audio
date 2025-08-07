import streamlit as st
import numpy as np
import soundfile as sf
import io
from audiorecorder import audiorecorder
from pydub import AudioSegment
import plotly.graph_objs as go
import time
from scipy.signal import stft, istft, butter, filtfilt
import pandas as pd
import noisereduce as nr
from scipy import signal

st.set_page_config(page_title="Analisador de Áudio", layout="centered")

def downsample_if_needed(data, samplerate, max_duration=120, target_sr=22050):
    """
    Reduz a taxa de amostragem e/ou duração se o arquivo for muito grande
    """
    duration = len(data) / samplerate
    
    if duration > max_duration:
        max_samples = int(max_duration * samplerate)
        data = data[:max_samples]
        st.warning(f"⚠️ Áudio truncado para {max_duration} segundos devido ao tamanho.")
    
    if samplerate > target_sr:
        resampling_factor = target_sr / samplerate
        new_length = int(len(data) * resampling_factor)
        data = signal.resample(data, new_length)
        samplerate = target_sr
        st.info(f"ℹ️ Taxa de amostragem reduzida para {target_sr} Hz para otimização.")
    
    return data, samplerate

def reduce_plot_density(x, y, max_points=5000):
    """
    Reduz a densidade de pontos para gráficos grandes
    """
    if len(x) <= max_points:
        return x, y
    
    step = len(x) // max_points
    return x[::step], y[::step]
st.title("🔊 Análise e Tratamento de Ondas Sonoras")

with st.sidebar:
    st.header("⚙️ Configurações de Performance")
    max_duration = st.slider("⏱️ Duração máxima (segundos)", 30, 300, 120, 30,
                            help="Áudios longos serão truncados para otimizar performance")
    target_samplerate = st.selectbox("🎵 Taxa de amostragem alvo", 
                                   [16000, 22050, 44100], 
                                   index=1,
                                   help="Taxas menores = melhor performance")
    max_plot_points = st.selectbox("📊 Densidade do gráfico",
                                 [1000, 2500, 5000, 10000],
                                 index=2,
                                 help="Menos pontos = gráficos mais rápidos")
    
    st.markdown("---")
    st.markdown("**💡 Dicas para arquivos grandes:**")
    st.markdown("• Use taxas de amostragem menores")
    st.markdown("• Limite a duração do áudio")
    st.markdown("• Reduza a densidade dos gráficos")

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

if uploaded_file is not None:
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

    # Verifica o tamanho do arquivo
    file_size_mb = len(audio_bytes) / (1024 * 1024)
    st.info(f"📁 Tamanho do arquivo: {file_size_mb:.1f} MB")
    
    if file_size_mb > 50:
        st.warning("⚠️ Arquivo grande detectado. Aplicando otimizações automáticas...")

    filename = audio_file.name.lower()
    
    try:
        if filename.endswith(".mp3"):
            # Converte mp3 para wav na memória
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

        # Aplica otimizações para arquivos grandes
        original_duration = len(data) / samplerate
        data, samplerate = downsample_if_needed(data, samplerate, max_duration, target_samplerate)
        
        duration = len(data) / samplerate
        st.markdown(f"**Duração original:** {original_duration:.2f} segundos")
        if duration != original_duration:
            st.markdown(f"**Duração processada:** {duration:.2f} segundos")
        st.markdown(f"**Taxa de amostragem:** {samplerate} Hz")
        
    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
        st.stop()

    st.markdown("---")
    st.markdown("### 2. 🧹 Escolha do filtro")

    filter_option = st.selectbox("Filtro aplicado:", [
        "Nenhum", 
        "Remoção de Ruído", 
        "Ajuste de Ganho",
        "Equalizador"
    ])

    if filter_option == "Remoção de Ruído":
        metodo = st.radio("Método de redução:", ["Automático (noisereduce)", "Manual (máscara espectral suave)"])

        if metodo == "Automático (noisereduce)":
            with st.spinner("Aplicando redução automática..."):
                cleaned_audio = nr.reduce_noise(y=data, sr=samplerate)
            st.success("Redução automática aplicada com sucesso.")
            audio_to_use = cleaned_audio

        elif metodo == "Manual (máscara espectral suave)":
            st.markdown("🎚️ Ajuste o nível de ruído estimado (em dB).")
            noise_floor_db = st.slider("🔉 Intensidade do ruído a ser removido (dB)", min_value=-100, max_value=0, value=-50, step=1)

            with st.spinner("Aplicando filtro espectral com máscara suave..."):
                n_fft = 1024
                hop_length = n_fft // 2

                # STFT
                f, t_seg, Zxx = stft(data, fs=samplerate, nperseg=n_fft, noverlap=n_fft - hop_length)
                magnitude = np.abs(Zxx)
                phase = np.angle(Zxx)

                # Convertendo para dB
                magnitude_db = 20 * np.log10(magnitude + 1e-10)
                threshold = noise_floor_db

                # Máscara suave baseada em função sigmoide
                transition_db = 10  # quanto maior, mais gradual
                mask = 1 / (1 + np.exp(-(magnitude_db - threshold) / transition_db))

                cleaned_magnitude = magnitude * mask
                cleaned_Zxx = cleaned_magnitude * np.exp(1j * phase)
                _, cleaned_audio = istft(cleaned_Zxx, fs=samplerate, nperseg=n_fft, noverlap=n_fft - hop_length)

            st.success("Redução de ruído aplicada com máscara suave.")
            audio_to_use = cleaned_audio
            
    elif filter_option == "Ajuste de Ganho":
        st.markdown("🎚️ Aumente ou diminua o volume do áudio.")
        gain_db = st.slider("🔊 Ganho (em dB)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)

        gain_factor = 10 ** (gain_db / 20)  # Conversão de dB para fator linear
        audio_to_use = data * gain_factor

        # Clipping protection (limita entre -1.0 e 1.0)
        audio_to_use = np.clip(audio_to_use, -1.0, 1.0)

        st.success(f"Ganho de {gain_db:.1f} dB aplicado.")

    elif filter_option == "Equalizador":
        st.markdown("🎛️ **Equalizador Paramétrico Profissional**")
        
        # Presets profissionais
        presets = {
            "Flat (Neutro)": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Rock": [-1, 2, 4, 2, -1, -2, 3, 5, 6, 4],
            "Pop": [1, 3, 2, 0, -1, -1, 2, 4, 5, 3],
            "Jazz": [2, 1, 0, 1, 2, 1, 0, 1, 2, 1],
            "Classical": [2, 1, -1, -2, -1, 1, 2, 3, 4, 3],
            "Electronic": [3, 2, 0, -2, -1, 2, 4, 5, 6, 5],
            "Vocal Enhancement": [0, 0, -1, 2, 4, 3, 1, -1, 0, 0],
            "Bass Boost": [6, 4, 2, 0, -1, -1, 0, 1, 1, 0],
            "Treble Boost": [0, 0, -1, -1, 0, 2, 4, 6, 7, 6],
            "Presence": [0, 0, 1, 2, 3, 4, 2, 1, 0, 0]
        }
        
        col_preset, col_reset = st.columns([4, 1])
        with col_preset:
            selected_preset = st.selectbox("🎵 Presets Profissionais:", list(presets.keys()))
        with col_reset:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset", use_container_width=True):
                selected_preset = "Flat (Neutro)"
        
        # Bandas de frequência profissionais (10-band parametric EQ)
        freq_bands = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        band_labels = ["31Hz\nSub-Bass", "62Hz\nBass", "125Hz\nLow-Mid", "250Hz\nMid-Bass", 
                      "500Hz\nMidrange", "1kHz\nPresence", "2kHz\nClarity", "4kHz\nBrilliance", 
                      "8kHz\nAir", "16kHz\nSparkle"]
        
        # Configurações avançadas
        with st.expander("⚙️ Configurações Avançadas"):
            q_factor = st.slider("🔧 Q-Factor (Largura da Banda)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                help="Controla a largura da banda de frequência. Valores menores = banda mais larga")
            auto_gain = st.checkbox("🔄 Compensação Automática de Ganho", value=True,
                                  help="Ajusta automaticamente o volume para evitar clipping")
        
        # Interface do equalizador
        st.markdown("#### 🎚️ Controles de Equalização")
        gains = []
        
        # Organiza em 2 colunas para melhor visualização
        col1, col2 = st.columns(2)
        
        # Inicializa com preset selecionado
        preset_gains = presets[selected_preset]
        
        with col1:
            for i in range(5):
                gain = st.slider(
                    band_labels[i], 
                    min_value=-15.0, 
                    max_value=15.0, 
                    value=float(preset_gains[i]), 
                    step=0.5,
                    key=f"eq_band_{i}"
                )
                gains.append(gain)
        
        with col2:
            for i in range(5, 10):
                gain = st.slider(
                    band_labels[i], 
                    min_value=-15.0, 
                    max_value=15.0, 
                    value=float(preset_gains[i]), 
                    step=0.5,
                    key=f"eq_band_{i}"
                )
                gains.append(gain)
        
        # Visualização da curva de resposta de frequência
        st.markdown("#### 📊 Curva de Resposta de Frequência")
        
        # Criar curva suavizada para visualização
        freq_response = np.logspace(np.log10(20), np.log10(20000), 1000)
        response_db = np.zeros_like(freq_response)
        
        for freq, gain in zip(freq_bands, gains):
            if abs(gain) > 0.01:
                # Simula resposta de filtro peaking/shelving
                for j, f in enumerate(freq_response):
                    # Função de resposta simplificada para visualização
                    bandwidth = freq / q_factor
                    if f <= freq:
                        response_db[j] += gain * np.exp(-((np.log(freq/f))**2) / (2 * (bandwidth/freq)**2))
                    else:
                        response_db[j] += gain * np.exp(-((np.log(f/freq))**2) / (2 * (bandwidth/freq)**2))
        
        fig_response = go.Figure()
        fig_response.add_trace(go.Scatter(
            x=freq_response, 
            y=response_db, 
            mode="lines", 
            line={"color": "green", "width": 3},
            name="Resposta EQ"
        ))
        
        # Linha de referência (0 dB)
        fig_response.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Marcadores das bandas
        for freq, gain in zip(freq_bands, gains):
            if abs(gain) > 0.01:
                fig_response.add_trace(go.Scatter(
                    x=[freq], y=[gain], 
                    mode="markers", 
                    marker={"size": 10, "color": "red"},
                    name=f"{freq}Hz: {gain:+.1f}dB",
                    showlegend=False
                ))
        
        fig_response.update_layout(
            xaxis_title="Frequência (Hz)",
            yaxis_title="Ganho (dB)",
            xaxis={"type": "log", "range": [np.log10(20), np.log10(20000)]},
            yaxis={"range": [-20, 20]},
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_response, use_container_width=True)
        
        with st.spinner("Aplicando equalização profissional..."):
            def apply_peaking_filter(signal, freq, gain_db, q, fs):
                """Aplica filtro peaking EQ usando biquad"""
                if abs(gain_db) < 0.01:
                    return signal
                
                # Converte para radianos
                w = 2 * np.pi * freq / fs
                
                # Parâmetros do filtro biquad peaking
                A = 10 ** (gain_db / 40)
                alpha = np.sin(w) / (2 * q)
                
                # Coeficientes do filtro biquad
                b0 = 1 + alpha * A
                b1 = -2 * np.cos(w)
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * np.cos(w)
                a2 = 1 - alpha / A
                
                # Normaliza
                b = np.array([b0, b1, b2]) / a0
                a = np.array([a0, a1, a2]) / a0
                
                # Aplica filtro
                filtered = filtfilt(b, a, signal)
                return filtered
            
            # Inicia com o áudio original
            audio_to_use = data.copy()
            
            # Aplica cada banda de equalização
            for freq, gain in zip(freq_bands, gains):
                if abs(gain) > 0.01:
                    audio_to_use = apply_peaking_filter(audio_to_use, freq, gain, q_factor, samplerate)
            
            # Compensação automática de ganho
            if auto_gain:
                # Calcula o ganho total aplicado
                total_positive_gain = sum([g for g in gains if g > 0])
                if total_positive_gain > 3:  # Se houver ganho significativo
                    # Aplica compressão suave para evitar clipping
                    peak = np.max(np.abs(audio_to_use))
                    if peak > 0.95:
                        compression_ratio = 0.95 / peak
                        audio_to_use *= compression_ratio
                        st.info(f"💡 Compressão aplicada: {20*np.log10(compression_ratio):.1f} dB para evitar distorção")
            
            # Proteção contra clipping
            audio_to_use = np.clip(audio_to_use, -1.0, 1.0)
        
        # Informações técnicas
        applied_bands = [(f, g) for f, g in zip(freq_bands, gains) if abs(g) > 0.01]
        if applied_bands:
            st.success("✅ Equalização aplicada com sucesso!")
            with st.expander("📋 Detalhes Técnicos"):
                st.markdown("**Bandas modificadas:**")
                for freq, gain in applied_bands:
                    st.markdown(f"• {freq} Hz: {gain:+.1f} dB")
                st.markdown(f"**Q-Factor:** {q_factor}")
                st.markdown(f"**Compensação de Ganho:** {'Ativada' if auto_gain else 'Desativada'}")
        else:
            st.info("ℹ️ Nenhuma modificação aplicada (todas as bandas em 0 dB)")
        
    else:
        audio_to_use = data

    st.markdown("---")
    st.markdown("### 3. 🔊 Áudio para reprodução")
    audio_preview_buffer = io.BytesIO()
    sf.write(audio_preview_buffer, audio_to_use, samplerate, format='WAV')
    audio_preview_buffer.seek(0)
    st.audio(audio_preview_buffer, format='audio/wav')

    st.markdown("---")
    st.markdown("### 4. 📈 Visualização dos Dados")

    st.subheader("🔵 Forma de Onda (Tempo)")
    t = np.arange(len(audio_to_use)) / samplerate
    
    # Reduz densidade para gráficos grandes
    t_plot, audio_plot = reduce_plot_density(t, audio_to_use, max_plot_points)
    
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=t_plot, y=audio_plot, mode="lines", line={"color": "blue"}))
    fig_wave.update_layout(xaxis_title="Tempo (s)", yaxis_title="Amplitude", height=300)
    st.plotly_chart(fig_wave, use_container_width=True)

    # Otimização do FFT para arquivos grandes
    if len(audio_to_use) > 100000:  # Se mais de 100k amostras
        # Usa apenas uma janela do áudio para o FFT
        window_size = min(65536, len(audio_to_use))  # Máximo 64k amostras
        start_idx = len(audio_to_use) // 2 - window_size // 2
        audio_fft = audio_to_use[start_idx:start_idx + window_size]
        st.info(f"ℹ️ FFT calculado usando janela de {window_size} amostras para otimização.")
    else:
        audio_fft = audio_to_use

    N = len(audio_fft)
    yf = np.abs(np.fft.rfft(audio_fft))
    xf = np.fft.rfftfreq(N, 1 / samplerate)

    freq_dominante = xf[np.argmax(yf)]
    st.subheader("🟣 Espectro de Frequência")
    st.markdown(f"🎯 **Frequência dominante:** {freq_dominante:.2f} Hz")

    # Reduz densidade do espectro para visualização
    xf_plot, yf_plot = reduce_plot_density(xf, yf, max_plot_points)

    fig_spec = go.Figure()
    fig_spec.add_trace(go.Scatter(x=xf_plot, y=yf_plot, mode="lines", line={"color": "purple"}))
    fig_spec.update_layout(
        xaxis_title="Frequência (Hz)",
        yaxis_title="Amplitude",
        xaxis={"range": [0, samplerate / 2]},
        height=400
    )
    st.plotly_chart(fig_spec, use_container_width=True)

    st.markdown("### 5. 📤 Exportação")

    # Reduz a densidade dos dados do espectro para exportação
    export_density = min(10000, len(xf))  # Máximo 10k pontos
    if len(xf) > export_density:
        step = len(xf) // export_density
        xf_export = xf[::step]
        yf_export = yf[::step]
        st.info(f"ℹ️ Espectro reduzido para {len(xf_export)} pontos para exportação otimizada.")
    else:
        xf_export = xf
        yf_export = yf

    spectrum_df = pd.DataFrame({
        "Frequência (Hz)": xf_export.round(2),
        "Amplitude": yf_export.round(4)
    })

    csv = spectrum_df.to_csv(index=False, float_format='%.4f').encode('utf-8')
    st.download_button(
        label="📥 Baixar espectro como CSV",
        data=csv,
        file_name="espectro_audio.csv",
        mime="text/csv"
    )

    # Otimização para exportação de áudio
    try:
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_to_use, samplerate, format='WAV')
        wav_io.seek(0)

        audio_segment = AudioSegment.from_file(wav_io, format="wav")
        mp3_io = io.BytesIO()
        # Usa bitrate mais baixo para arquivos grandes
        bitrate = "128k" if file_size_mb > 20 else "192k"
        audio_segment.export(mp3_io, format="mp3", bitrate=bitrate)
        mp3_io.seek(0)

        st.download_button(
            label=f"📥 Baixar áudio processado como MP3 ({bitrate})",
            data=mp3_io,
            file_name="audio_processado.mp3",
            mime="audio/mpeg"
        )
        
        # Mostra informações de tamanho
        mp3_size_mb = len(mp3_io.getvalue()) / (1024 * 1024)
        st.info(f"📊 Tamanho do MP3: {mp3_size_mb:.1f} MB")
        
    except Exception as e:
        st.error(f"❌ Erro na exportação: {str(e)}")
        st.warning("💡 Tente reduzir a duração ou taxa de amostragem para exportar.")

else:
    st.info("Envie um arquivo .wav ou grave para visualizar o espectro.")
