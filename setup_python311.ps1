# Скрипт для настройки Python 3.11 для CogniFlex
Write-Host "Установка Python 3.11 и настройка окружения для CogniFlex" -ForegroundColor Cyan
Write-Host "==================================================="

# Проверяем, установлен ли Python 3.11
$python311 = Get-Command python3.11 -ErrorAction SilentlyContinue

if ($null -eq $python311) {
    Write-Host "Python 3.11 не найден. Пожалуйста, установите Python 3.11 с официального сайта." -ForegroundColor Red
    Write-Host "Скачайте установщик с https://www.python.org/downloads/release/python-3110/" -ForegroundColor Yellow
    Start-Process "https://www.python.org/downloads/release/python-3110/"
    exit 1
}

# Создаем новое виртуальное окружение
Write-Host "`nСоздание виртуального окружения Python 3.11..." -ForegroundColor Green
python3.11 -m venv .venv311

# Активируем виртуальное окружение
$activateScript = ".\.venv311\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    . $activateScript
} else {
    Write-Host "Ошибка: Не удалось активировать виртуальное окружение" -ForegroundColor Red
    exit 1
}

# Обновляем pip
Write-Host "`nОбновление pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Устанавливаем зависимости из основного файла requirements.txt
Write-Host "`nУстановка основных зависимостей..." -ForegroundColor Green
pip install -r requirements.txt

# Устанавливаем PyTorch для CPU
Write-Host "`nУстановка PyTorch для CPU..." -ForegroundColor Green
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем зависимости из cogniflex/requirements.txt
Write-Host "`nУстановка дополнительных зависимостей CogniFlex..." -ForegroundColor Green
pip install -r cogniflex\requirements.txt

# Устанавливаем модели для spaCy
Write-Host "`nУстановка моделей для spaCy..." -ForegroundColor Green
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm

# Завершение
Write-Host "`n===================================================" -ForegroundColor Cyan
Write-Host "Виртуальное окружение успешно настроено!" -ForegroundColor Green
Write-Host "`nЧтобы активировать окружение, выполните:" -ForegroundColor Yellow
Write-Host ".\.venv311\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "`nДля деактивации просто введите:" -ForegroundColor Yellow
Write-Host "deactivate" -ForegroundColor White

# Ожидаем нажатия клавиши
Write-Host "`nНажмите любую клавишу для выхода..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
