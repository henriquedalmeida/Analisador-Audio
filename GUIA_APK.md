# 📱 Guia para Gerar APK - Analisador de Áudio

Este guia explica como converter sua aplicação Streamlit em um APK Android.

## 🎯 O que foi criado

Criei uma versão mobile adaptada do seu projeto usando **Kivy**, que permite gerar APKs nativos Android.

### Arquivos adicionados:

1. **`main.py`** - Aplicação Kivy mobile
2. **`buildozer.spec`** - Configurações do APK
3. **`requirements-mobile.txt`** - Dependências mobile
4. **`build_apk.sh`** - Script de build para Linux/macOS
5. **`setup_windows.ps1`** - Configuração para Windows
6. **`.github/workflows/build-apk.yml`** - Build automático no GitHub

## 🚀 Como gerar o APK

### Opção 1: GitHub Actions (Mais Fácil)

1. **Faça push para o GitHub:**
```bash
git add .
git commit -m "Adicionada versão mobile para APK"
git push origin main
```

2. **No GitHub:**
   - Vá em "Actions" no seu repositório
   - Execute o workflow "Build Android APK"
   - Baixe o APK gerado nos "Artifacts"

### Opção 2: WSL no Windows (Recomendado)

1. **Habilite o WSL:**
```powershell
# Execute como Administrador
wsl --install -d Ubuntu
```

2. **Configure no Ubuntu WSL:**
```bash
cd /mnt/c/Users/Henrique\ Almeida/Desktop/Analisador-Audio
chmod +x build_apk.sh
./build_apk.sh
```

### Opção 3: Linux nativo

```bash
chmod +x build_apk.sh
./build_apk.sh
```

## 📋 Funcionalidades da versão mobile

✅ **Implementado:**
- Carregamento de arquivos de áudio (.wav, .mp3)
- Aplicação de filtros (ruído, ganho, equalização)
- Visualização de forma de onda e espectro
- Exportação de dados (CSV)
- Salvamento de áudio processado
- Interface otimizada para mobile

⏳ **A implementar:**
- Gravação de áudio via microfone
- Equalizador avançado com 10 bandas
- Presets profissionais
- Mais filtros de áudio

## 🔧 Personalizações possíveis

### Modificar informações do APK:
Edite `buildozer.spec`:
```ini
title = Seu Título
package.name = seupacote
package.domain = com.seudominio.seuapp
version = 1.1
```

### Adicionar ícone:
1. Crie um ícone 512x512 px chamado `icon.png`
2. Adicione no `buildozer.spec`:
```ini
icon.filename = icon.png
```

### Modificar permissões:
No `buildozer.spec`:
```ini
android.permissions = WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,RECORD_AUDIO,INTERNET,CAMERA
```

## 🐛 Solução de problemas

### Erro de dependências:
```bash
# Instale dependências manualmente
pip install kivy[base] kivymd numpy scipy matplotlib soundfile pandas
```

### Erro de NDK/SDK:
O Buildozer baixa automaticamente. Se houver erro:
```bash
buildozer android clean
buildozer android debug
```

### APK muito grande:
- Remova dependências não utilizadas do `requirements-mobile.txt`
- Use `--arch armeabi-v7a` para arquitetura específica

## 📱 Testando o APK

1. **Instale no dispositivo:**
   - Ative "Fontes desconhecidas" nas configurações
   - Transfira o APK e instale

2. **Teste via USB Debug:**
```bash
buildozer android deploy run
```

## 📊 Comparação: Web vs Mobile

| Recurso | Streamlit (Web) | Kivy (Mobile) |
|---------|----------------|---------------|
| Interface | Automatizada | Personalizada |
| Performance | Servidor | Nativo |
| Distribuição | URL | APK |
| Offline | Não | Sim |
| Sensores | Limitado | Completo |

## 🎯 Próximos passos

1. **Execute o build** usando uma das opções acima
2. **Teste o APK** em um dispositivo Android
3. **Customize** a interface conforme necessário
4. **Publique** na Google Play Store (se desejar)

---

**💡 Dica:** Para desenvolvimento rápido, use GitHub Actions. Para builds frequentes, configure WSL localmente.
