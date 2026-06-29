# Запуск полного обучения STDP + CollocationMatrix
# Результаты пишутся в train_full.log
$ErrorActionPreference = "Continue"
$logFile = Join-Path $PSScriptRoot "train_full.log"
$trainScript = Join-Path $PSScriptRoot "train_full.py"

# Таймштамп
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"=== $timestamp ===" | Out-File -FilePath $logFile -Encoding utf8

# Запуск
Set-Location -LiteralPath $PSScriptRoot
python $trainScript 2>&1 | ForEach-Object { 
    $_ | Out-File -FilePath $logFile -Encoding utf8 -Append
    Write-Host $_
}

# Если завершилось
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"=== DONE $timestamp ===" | Out-File -FilePath $logFile -Encoding utf8 -Append
Read-Host "Нажмите Enter для выхода"
