$WshShell = New-Object -ComObject WScript.Shell
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$MainPath = Join-Path $ProjectRoot "main.py"
$ShortcutPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Helper.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
$Shortcut.Arguments = '"' + $MainPath + '"'
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Description = "Launch Helper AI Assistant"
$Shortcut.Save()
Write-Host "Shortcut created at $ShortcutPath"
