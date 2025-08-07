# Script para testar a aplicação mobile localmente (sem gerar APK)

try:
    from main import AudioAnalyzerApp
    print("✅ Dependências carregadas com sucesso!")
    print("🚀 Iniciando aplicação mobile...")
    app = AudioAnalyzerApp()
    app.run()
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print("\n📦 Instalando dependências necessárias...")
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kivy", "matplotlib", "numpy", "scipy", "soundfile", "pandas"])
        print("✅ Dependências instaladas!")
        print("🔄 Tente executar novamente: python test_mobile.py")
    except Exception as install_error:
        print(f"❌ Erro na instalação: {install_error}")
        print("\n💡 Instale manualmente:")
        print("pip install kivy matplotlib numpy scipy soundfile pandas")
