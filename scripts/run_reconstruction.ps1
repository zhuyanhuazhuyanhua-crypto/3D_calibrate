param(
    [string]$Root = "$(Get-Location)"
)

python -m src.main demo --root $Root
