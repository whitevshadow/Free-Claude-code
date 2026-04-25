# Free Claude Code - Model Picker
# Shows all available real NVIDIA NIM models and starts Claude with your selection

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "     Free Claude Code - Model Selection Menu              " -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# Fetch models from proxy
Write-Host "Fetching available models..." -ForegroundColor Gray
try {
    $models = Invoke-RestMethod -Uri "http://localhost:8082/v1/models" -Headers @{"x-api-key" = "freecc"} -ErrorAction Stop
} catch {
    Write-Host "[ERROR] Could not connect to proxy at http://localhost:8082" -ForegroundColor Red
    Write-Host "        Make sure Docker is running: docker-compose up -d" -ForegroundColor Yellow
    exit 1
}

Write-Host "Found $($models.data.Count) models`n" -ForegroundColor Green

# Display models
$index = 1
$modelMap = @{}
foreach ($model in $models.data) {
    $modelMap[$index] = $model
    
    # Color code by provider
    $color = "White"
    if ($model.id -match "deepseek") { $color = "Magenta" }
    elseif ($model.id -match "qwen") { $color = "Cyan" }
    elseif ($model.id -match "mistral") { $color = "Yellow" }
    elseif ($model.id -match "kimi") { $color = "Blue" }
    elseif ($model.id -match "glm") { $color = "Green" }
    
    Write-Host "  $index. " -NoNewline -ForegroundColor Gray
    Write-Host "$($model.display_name)" -ForegroundColor $color
    Write-Host "     $($model.id)" -ForegroundColor DarkGray
    Write-Host ""
    $index++
}

# Get user selection
Write-Host "------------------------------------------------------------" -ForegroundColor Gray
$selection = Read-Host "`nSelect a model (1-$($models.data.Count)) or press Enter for #1"

if ([string]::IsNullOrWhiteSpace($selection)) {
    $selection = "1"
}

$selectionNum = 0
if (-not [int]::TryParse($selection, [ref]$selectionNum) -or $selectionNum -lt 1 -or $selectionNum -gt $models.data.Count) {
    Write-Host "`n[ERROR] Invalid selection. Please enter a number between 1 and $($models.data.Count)" -ForegroundColor Red
    exit 1
}

$selectedModel = $modelMap[$selectionNum]

Write-Host "`n[OK] Selected: " -NoNewline -ForegroundColor Green
Write-Host "$($selectedModel.display_name)" -ForegroundColor Cyan
Write-Host "     Model ID: $($selectedModel.id)`n" -ForegroundColor Gray

# Set environment variables
$env:ANTHROPIC_BASE_URL = "http://localhost:8082"
$env:ANTHROPIC_AUTH_TOKEN = "freecc"
$env:ANTHROPIC_MODEL = $selectedModel.id

Write-Host "[*] Starting Claude Code with $($selectedModel.display_name)...`n" -ForegroundColor Green
Write-Host "------------------------------------------------------------`n" -ForegroundColor Gray

# Start Claude
claude
