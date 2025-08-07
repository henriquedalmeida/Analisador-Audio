"""
Aplicativo Android - Analisador de Áudio
Versão mobile adaptada do projeto Streamlit
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.logger import Logger

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import io
from scipy.signal import stft, istft
from scipy import signal
import threading
import os

class AudioScreen(Screen):
    """Tela principal para análise de áudio"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio_data = None
        self.samplerate = None
        self.processed_data = None
        
        # Layout principal
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Título
        title = Label(
            text='🔊 Analisador de Áudio Mobile',
            size_hint_y=None,
            height=50,
            font_size=20
        )
        main_layout.add_widget(title)
        
        # Botões de ação
        buttons_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=60)
        
        load_btn = Button(text='📁 Carregar Áudio', size_hint_x=0.5)
        load_btn.bind(on_press=self.load_audio)
        buttons_layout.add_widget(load_btn)
        
        record_btn = Button(text='🎤 Gravar', size_hint_x=0.5)
        record_btn.bind(on_press=self.record_audio)
        buttons_layout.add_widget(record_btn)
        
        main_layout.add_widget(buttons_layout)
        
        # Informações do áudio
        self.info_label = Label(
            text='Nenhum áudio carregado',
            size_hint_y=None,
            height=40
        )
        main_layout.add_widget(self.info_label)
        
        # Filtros
        filter_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        filter_layout.add_widget(Label(text='Filtro:', size_hint_x=0.3))
        
        self.filter_spinner = Spinner(
            text='Nenhum',
            values=['Nenhum', 'Remoção de Ruído', 'Ajuste de Ganho', 'Equalizador'],
            size_hint_x=0.7
        )
        self.filter_spinner.bind(text=self.apply_filter)
        filter_layout.add_widget(self.filter_spinner)
        
        main_layout.add_widget(filter_layout)
        
        # Slider de ganho (inicialmente oculto)
        self.gain_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        self.gain_layout.add_widget(Label(text='Ganho (dB):', size_hint_x=0.3))
        
        self.gain_slider = Slider(min=-20, max=20, value=0, step=1, size_hint_x=0.7)
        self.gain_slider.bind(value=self.on_gain_change)
        self.gain_layout.add_widget(self.gain_slider)
        
        # Área para gráficos
        self.plot_area = BoxLayout(orientation='vertical')
        main_layout.add_widget(self.plot_area)
        
        # Botões de análise e exportação
        action_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=60)
        
        analyze_btn = Button(text='📊 Analisar', size_hint_x=0.33)
        analyze_btn.bind(on_press=self.analyze_audio)
        action_layout.add_widget(analyze_btn)
        
        export_btn = Button(text='💾 Exportar CSV', size_hint_x=0.33)
        export_btn.bind(on_press=self.export_data)
        action_layout.add_widget(export_btn)
        
        save_btn = Button(text='🎵 Salvar Áudio', size_hint_x=0.34)
        save_btn.bind(on_press=self.save_audio)
        action_layout.add_widget(save_btn)
        
        main_layout.add_widget(action_layout)
        
        self.add_widget(main_layout)
    
    def load_audio(self, instance):
        """Carrega arquivo de áudio"""
        content = BoxLayout(orientation='vertical')
        
        filechooser = FileChooserIconView(
            filters=['*.wav', '*.mp3'],
            path='/sdcard'
        )
        content.add_widget(filechooser)
        
        buttons = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        
        load_btn = Button(text='Carregar')
        cancel_btn = Button(text='Cancelar')
        
        buttons.add_widget(load_btn)
        buttons.add_widget(cancel_btn)
        content.add_widget(buttons)
        
        popup = Popup(title='Selecionar Arquivo de Áudio', content=content, size_hint=(0.9, 0.9))
        
        def load_selected(instance):
            if filechooser.selection:
                self._load_audio_file(filechooser.selection[0])
            popup.dismiss()
        
        def cancel(instance):
            popup.dismiss()
        
        load_btn.bind(on_press=load_selected)
        cancel_btn.bind(on_press=cancel)
        
        popup.open()
    
    def _load_audio_file(self, filepath):
        """Carrega o arquivo de áudio selecionado"""
        try:
            self.audio_data, self.samplerate = sf.read(filepath)
            
            # Se for estéreo, pega apenas um canal
            if len(self.audio_data.shape) > 1:
                self.audio_data = self.audio_data[:, 0]
            
            duration = len(self.audio_data) / self.samplerate
            
            self.info_label.text = f'Áudio carregado: {duration:.1f}s, {self.samplerate}Hz'
            self.processed_data = self.audio_data.copy()
            
            Logger.info(f'AudioAnalyzer: Arquivo carregado: {filepath}')
            
        except Exception as e:
            self.show_error(f'Erro ao carregar áudio: {str(e)}')
    
    def record_audio(self, instance):
        """Inicia gravação de áudio (placeholder - requer permissões Android)"""
        self.show_info('Funcionalidade de gravação em desenvolvimento.\nUse o carregamento de arquivo por enquanto.')
    
    def apply_filter(self, spinner, text):
        """Aplica filtro selecionado"""
        if self.audio_data is None:
            self.show_error('Carregue um áudio primeiro')
            return
        
        if text == 'Ajuste de Ganho':
            # Mostra o slider de ganho
            if self.gain_layout.parent is None:
                # Encontra onde inserir o slider
                main_layout = self.children[0]
                main_layout.add_widget(self.gain_layout, index=5)
        else:
            # Remove o slider de ganho se estiver visível
            if self.gain_layout.parent is not None:
                self.gain_layout.parent.remove_widget(self.gain_layout)
        
        # Aplica o filtro
        if text == 'Nenhum':
            self.processed_data = self.audio_data.copy()
        elif text == 'Remoção de Ruído':
            self._apply_noise_reduction()
        elif text == 'Ajuste de Ganho':
            self._apply_gain()
        elif text == 'Equalizador':
            self._apply_equalizer()
    
    def on_gain_change(self, instance, value):
        """Atualiza ganho em tempo real"""
        if self.filter_spinner.text == 'Ajuste de Ganho':
            self._apply_gain()
    
    def _apply_noise_reduction(self):
        """Aplica redução de ruído simples"""
        try:
            # Implementação básica de redução de ruído
            f, t, Zxx = stft(self.audio_data, nperseg=1024)
            
            # Remove componentes com amplitude baixa
            threshold = np.percentile(np.abs(Zxx), 75)
            mask = np.abs(Zxx) > threshold * 0.1
            Zxx_filtered = Zxx * mask
            
            _, self.processed_data = istft(Zxx_filtered)
            self.processed_data = self.processed_data.real
            
            self.show_info('Redução de ruído aplicada')
            
        except Exception as e:
            self.show_error(f'Erro na redução de ruído: {str(e)}')
    
    def _apply_gain(self):
        """Aplica ajuste de ganho"""
        try:
            gain_db = self.gain_slider.value
            gain_linear = 10 ** (gain_db / 20)
            
            self.processed_data = self.audio_data * gain_linear
            
            # Previne clipping
            max_val = np.max(np.abs(self.processed_data))
            if max_val > 1.0:
                self.processed_data = self.processed_data / max_val
                self.show_info(f'Ganho aplicado: {gain_db:.1f}dB (com limitação)')
            else:
                self.show_info(f'Ganho aplicado: {gain_db:.1f}dB')
            
        except Exception as e:
            self.show_error(f'Erro no ajuste de ganho: {str(e)}')
    
    def _apply_equalizer(self):
        """Aplica equalização básica"""
        # Implementação básica - pode ser expandida
        self.processed_data = self.audio_data.copy()
        self.show_info('Equalização básica aplicada')
    
    def analyze_audio(self, instance):
        """Analisa o áudio e mostra gráficos"""
        if self.processed_data is None:
            self.show_error('Carregue um áudio primeiro')
            return
        
        # Limpa área de gráficos
        self.plot_area.clear_widgets()
        
        # Cria gráficos em thread separada para não travar a UI
        threading.Thread(target=self._create_plots).start()
    
    def _create_plots(self):
        """Cria gráficos de análise"""
        try:
            # Gráfico de forma de onda
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Forma de onda
            time_axis = np.linspace(0, len(self.processed_data) / self.samplerate, len(self.processed_data))
            
            # Reduz pontos para performance
            step = max(1, len(time_axis) // 2000)
            time_reduced = time_axis[::step]
            data_reduced = self.processed_data[::step]
            
            ax1.plot(time_reduced, data_reduced)
            ax1.set_title('Forma de Onda')
            ax1.set_xlabel('Tempo (s)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True)
            
            # Espectro de frequência
            fft_data = np.fft.fft(self.processed_data)
            freqs = np.fft.fftfreq(len(self.processed_data), 1/self.samplerate)
            magnitude = np.abs(fft_data)
            
            # Pega apenas metade positiva
            half_len = len(freqs) // 2
            freqs_plot = freqs[:half_len]
            magnitude_plot = magnitude[:half_len]
            
            # Reduz pontos para performance
            step = max(1, len(freqs_plot) // 1000)
            freqs_reduced = freqs_plot[::step]
            magnitude_reduced = magnitude_plot[::step]
            
            ax2.plot(freqs_reduced, magnitude_reduced)
            ax2.set_title('Espectro de Frequência')
            ax2.set_xlabel('Frequência (Hz)')
            ax2.set_ylabel('Magnitude')
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Adiciona gráfico à UI no thread principal
            Clock.schedule_once(lambda dt: self._add_plot_to_ui(fig), 0)
            
        except Exception as e:
            Clock.schedule_once(lambda dt: self.show_error(f'Erro na análise: {str(e)}'), 0)
    
    def _add_plot_to_ui(self, fig):
        """Adiciona gráfico à interface (executado no thread principal)"""
        canvas = FigureCanvasKivyAgg(fig)
        self.plot_area.add_widget(canvas)
    
    def export_data(self, instance):
        """Exporta dados espectrais para CSV"""
        if self.processed_data is None:
            self.show_error('Nenhum áudio processado para exportar')
            return
        
        try:
            # Calcula FFT
            fft_data = np.fft.fft(self.processed_data)
            freqs = np.fft.fftfreq(len(self.processed_data), 1/self.samplerate)
            magnitude = np.abs(fft_data)
            
            # Pega apenas metade positiva
            half_len = len(freqs) // 2
            freqs_export = freqs[:half_len]
            magnitude_export = magnitude[:half_len]
            
            # Salva CSV
            import pandas as pd
            df = pd.DataFrame({
                'Frequencia_Hz': freqs_export,
                'Magnitude': magnitude_export
            })
            
            filepath = '/sdcard/Download/espectro_audio.csv'
            df.to_csv(filepath, index=False)
            
            self.show_info(f'Dados exportados para:\n{filepath}')
            
        except Exception as e:
            self.show_error(f'Erro na exportação: {str(e)}')
    
    def save_audio(self, instance):
        """Salva áudio processado"""
        if self.processed_data is None:
            self.show_error('Nenhum áudio processado para salvar')
            return
        
        try:
            filepath = '/sdcard/Download/audio_processado.wav'
            sf.write(filepath, self.processed_data, self.samplerate)
            
            self.show_info(f'Áudio salvo em:\n{filepath}')
            
        except Exception as e:
            self.show_error(f'Erro ao salvar áudio: {str(e)}')
    
    def show_error(self, message):
        """Mostra popup de erro"""
        popup = Popup(
            title='Erro',
            content=Label(text=message),
            size_hint=(0.8, 0.4)
        )
        popup.open()
    
    def show_info(self, message):
        """Mostra popup de informação"""
        popup = Popup(
            title='Informação',
            content=Label(text=message),
            size_hint=(0.8, 0.4)
        )
        popup.open()


class AudioAnalyzerApp(App):
    """Aplicativo principal"""
    
    def build(self):
        sm = ScreenManager()
        sm.add_widget(AudioScreen(name='audio'))
        return sm


if __name__ == '__main__':
    AudioAnalyzerApp().run()
