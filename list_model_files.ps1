# PowerShell script to list files in the model directory
$modelPath = "C:\Users\black\OneDrive\Desktop\rugpt3_large"

Write-Host "Содержимое директории модели:"
Get-ChildItem -Path $modelPath -Recurse | Format-Table FullName, Length, LastWriteTime -AutoSize

# Save to a file for better readability
$outputPath = "model_files_list.txt"
Get-ChildItem -Path $modelPath -Recurse | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize | Out-File -FilePath $outputPath -Width 1000
Write-Host "`nСписок файлов сохранен в: $outputPath"
