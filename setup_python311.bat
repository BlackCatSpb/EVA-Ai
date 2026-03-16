@echo off
REM Script to set up Python 3.11 environment for CogniFlex

echo Установка Python 3.11 и настройка окружения для CogniFlex
echo ===================================================

REM Проверяем, установлен ли Python 3.11
where python3.11 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python 3.11 не найден. Пожалуйста, установите Python 3.11 с официального сайта.
    echo Скачайте установщик с https://www.python.org/downloads/release/python-3110/
    pause
    exit /b 1
)

REM Создаем новое виртуальное окружение
echo.
echo Создание виртуального окружения Python 3.11...
python3.11 -m venv .venv311

REM Активируем виртуальное окружение
call .venv311\Scripts\activate

REM Обновляем pip
echo.
echo Обновление pip...
python -m pip install --upgrade pip

REM Устанавливаем зависимости из основного файла requirements.txt
echo.
echo Установка основных зависимостей...
pip install -r requirements.txt

REM Устанавливаем PyTorch для CPU (можно закомментировать, если нужна версия с поддержкой GPU)
echo.
echo Установка PyTorch для CPU...
pip install torch --index-url https://download.pytorch.org/whl/cpu

REM Устанавливаем зависимости из cogniflex/requirements.txt
echo.
echo Установка дополнительных зависимостей CogniFlex...
pip install -r cogniflex\requirements.txt

REM Устанавливаем модели для spaCy
echo.
echo Установка моделей для spaCy...
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm

REM Завершение
echo.
echo ===================================================
echo Виртуальное окружение успешно настроено!
echo.
echo Чтобы активировать окружение, выполните:
echo .\.venv311\Scripts\activate
echo.
echo Для деактивации просто введите:
echo deactivate

pause
