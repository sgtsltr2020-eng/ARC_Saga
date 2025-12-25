$ErrorActionPreference = "Stop"
git add .
git commit -m "wip: snapshot $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
git push origin HEAD
Write-Host "Work saved to remote."
