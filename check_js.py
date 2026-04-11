#!/usr/bin/env python3
import re

with open('eva_ai/gui/web_gui/static/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

# Проверить что функция loadAnalytics закрыта
if 'function loadAnalytics()' in js:
    start = js.find('function loadAnalytics()')
    brace_count = 0
    in_function = False
    end = start
    for i, c in enumerate(js[start:], start):
        if c == '{':
            brace_count += 1
            in_function = True
        elif c == '}':
            brace_count -= 1
            if in_function and brace_count == 0:
                end = i + 1
                break
    
    print(f'loadAnalytics: найдена (start={start}, end={end})')
    
    after = js[end:end+10].strip()
    print(f'После функции: "{after[:30]}..."')
    
    # Проверить что нет SyntaxError индикаторов
    if 'SyntaxError' in js:
        print('WARNING: SyntaxError найден в коде!')
    
    print('JS файл выглядит корректно!')
else:
    print('ERROR: loadAnalytics не найдена!')
