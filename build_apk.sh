#!/bin/bash

# Script para gerar APK do Analisador de Áudio
# Requer Linux/macOS ou WSL no Windows

echo "🔧 Configurando ambiente para gerar APK..."

# Verifica se está no ambiente Linux/WSL
if [[ "$OSTYPE" != "linux-gnu"* ]] && [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Este script requer Linux, macOS ou WSL (Windows Subsystem for Linux)"
    echo "Para Windows, instale o WSL primeiro:"
    echo "https://docs.microsoft.com/en-us/windows/wsl/install"
    exit 1
fi

# Instala dependências do sistema
echo "📦 Instalando dependências do sistema..."

if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv git
    sudo apt-get install -y build-essential ccache git libncurses5:i386 libstdc++6:i386 libgtk2.0-0:i386 libpangox-1.0-0:i386 libpangoxft-1.0-0:i386 libidn11:i386 python2.7 python2.7-dev openjdk-8-jdk unzip zlib1g-dev zlib1g:i386
elif command -v brew &> /dev/null; then
    # macOS
    brew install python3 git
    brew install --cask android-sdk
else
    echo "❌ Distribuição não suportada. Use Ubuntu/Debian ou macOS"
    exit 1
fi

# Cria ambiente virtual
echo "🐍 Criando ambiente virtual..."
python3 -m venv venv-mobile
source venv-mobile/bin/activate

# Instala buildozer e dependências
echo "📱 Instalando Buildozer..."
pip install --upgrade pip
pip install buildozer cython

# Instala dependências do projeto
echo "📚 Instalando dependências do projeto..."
pip install -r requirements-mobile.txt

# Configura variáveis de ambiente para Android
echo "⚙️ Configurando variáveis de ambiente..."
export ANDROIDAPI="30"
export ANDROID_HOME="$HOME/.buildozer/android/platform/android-sdk"
export PATH="$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools"

# Inicializa buildozer (baixa dependências Android)
echo "🔽 Inicializando Buildozer (primeira execução pode demorar muito)..."
buildozer android debug

echo ""
echo "✅ Build concluído!"
echo ""
echo "📱 O APK foi gerado em: ./bin/analisadoraudio-1.0-armeabi-v7a-debug.apk"
echo ""
echo "📋 Para instalar no dispositivo Android:"
echo "1. Ative 'Opções do desenvolvedor' e 'Depuração USB' no Android"
echo "2. Conecte o dispositivo via USB"
echo "3. Execute: buildozer android deploy"
echo ""
echo "Ou transfira o arquivo APK manualmente para o dispositivo."
