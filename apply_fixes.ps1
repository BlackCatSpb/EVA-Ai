# apply_fixes.ps1
$path = 'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\fcp_pipeline.py'
# Проверка существования файла
if (-Not (Test-Path $path)) {
    Write-Host "ERROR: File not found: $path" -ForegroundColor Red
    exit 1
}
# Чтение файла
$content = Get-Content -Path $path -Raw
$lines = $content -split "`n"
# Fix 1: Исправить отступ у строки 146 (индекс 145)
$lines[145] = '            self.pipeline = ov_genai.LLMPipeline('
# Fix 2: Добавить **kwargs после use_lora: bool = True
for ($i = 0; $i -lt $lines.Length; $i++) {
    if ($lines[$i] -match 'use_lora:\s*bool\s*=\s*True') {
        $insertPos = $i + 1
        if ($i + 1 -lt $lines.Length -and $lines[$i + 1] -match 'return_metadata') {
            $insertPos = $i + 2
        }
        $lines = $lines[0..$i] + @('        **kwargs') + $lines[($i+1)..($lines.Length-1)]
        break
    }
}
# Fix 3: Исправить вызов _generate для передачи **kwargs
for ($i = 0; $i -lt $lines.Length; $i++) {
    if ($lines[$i] -match 'response\s*=\s*self\._generate\(chat_prompt,\s*max_new_tokens\)') {
        $lines[$i] = '        response = self._generate(chat_prompt, max_new_tokens, **kwargs)'
        break
    }
}
# Запись изменений
$lines | Set-Content -Path $path -Encoding UTF8
# Проверка синтаксиса через Python
$pythonCheck = & python3 -c "
import py_compile
try:
    py_compile.compile('$path', doraise=True)
    print('SUCCESS')
except Exception as e:
    print('FAIL:' + str(e))
"
if ($pythonCheck -match 'SUCCESS') {
    Write-Host "SUCCESS: File fixed and compiles correctly" -ForegroundColor Green
} else {
    Write-Host "FAIL: $pythonCheck" -ForegroundColor Red
    exit 1
}