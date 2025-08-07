# Script PowerShell para configurar ambiente Android no Windows

Write-Host "🔧 Configurando ambiente para gerar APK no Windows..." -ForegroundColor Green

# Verifica se o Chocolatey está instalado
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "📦 Instalando Chocolatey..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Instala dependências
Write-Host "📦 Instalando Python e Git..." -ForegroundColor Yellow
choco install python git -y

# Verifica se WSL está disponível
$wslAvailable = (Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux).State -eq "Enabled"

if (-not $wslAvailable) {
    Write-Host "⚠️  WSL não está habilitado. Para gerar APK no Windows, você tem 3 opções:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "OPÇÃO 1 - Habilitar WSL (Recomendado):" -ForegroundColor Cyan
    Write-Host "1. Execute como Administrador: wsl --install"
    Write-Host "2. Reinicie o computador"
    Write-Host "3. Configure Ubuntu no WSL"
    Write-Host "4. Execute o script build_apk.sh dentro do WSL"
    Write-Host ""
    Write-Host "OPÇÃO 2 - Docker (Alternativa):" -ForegroundColor Cyan
    Write-Host "1. Instale Docker Desktop"
    Write-Host "2. Use um container Linux para build"
    Write-Host ""
    Write-Host "OPÇÃO 3 - GitHub Actions (Mais fácil):" -ForegroundColor Cyan
    Write-Host "1. Faça push do código para GitHub"
    Write-Host "2. Configure GitHub Actions para build automático"
    Write-Host ""
    
    $choice = Read-Host "Deseja habilitar WSL agora? (s/n)"
    if ($choice -eq "s" -or $choice -eq "S") {
        Write-Host "🔧 Habilitando WSL..." -ForegroundColor Green
        dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
        dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
        
        Write-Host "✅ WSL habilitado! REINICIE o computador e execute:" -ForegroundColor Green
        Write-Host "wsl --install -d Ubuntu" -ForegroundColor White
        Write-Host "Depois, dentro do Ubuntu WSL, execute o script build_apk.sh" -ForegroundColor White
    }
} else {
    Write-Host "✅ WSL já está habilitado!" -ForegroundColor Green
    Write-Host "Execute dentro do WSL Ubuntu:" -ForegroundColor Cyan
    Write-Host "bash build_apk.sh" -ForegroundColor White
}

Write-Host ""
Write-Host "📋 Próximos passos:" -ForegroundColor Yellow
Write-Host "1. Configure WSL ou Docker"
Write-Host "2. Execute o script de build adequado"
Write-Host "3. O APK será gerado em ./bin/"
