@echo off
echo ============================================================
echo  SETUP - JurisBERTa NER - RTX 5060 Ti (Blackwell / CUDA 12.8+)
echo ============================================================

:: Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado. Instale Python 3.10+ antes de continuar.
    pause
    exit /b 1
)

echo.
echo [1/4] Criando ambiente virtual...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/4] Instalando PyTorch com suporte CUDA 12.8 (Blackwell)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo.
echo [3/4] Instalando bibliotecas de NLP e treinamento...
pip install ^
    transformers==4.51.3 ^
    datasets==3.5.0 ^
    tokenizers==0.21.1 ^
    accelerate==1.6.0 ^
    seqeval ^
    scikit-learn ^
    numpy ^
    pandas ^
    tqdm ^
    huggingface_hub

echo.
echo [4/4] Verificando instalacao da GPU...
python -c "import torch; gpu=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if gpu else 'N/A'; print(f'GPU disponivel: {gpu}'); print(f'Dispositivo: {name}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB' if gpu else '')"

echo.
echo ============================================================
echo  Setup concluido! Para ativar o ambiente no futuro:
echo  call venv\Scripts\activate.bat
echo ============================================================
pause
