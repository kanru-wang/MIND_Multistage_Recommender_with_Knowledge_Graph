param(
    [ValidateSet("phase1", "phase2", "phase3", "all")]
    [string]$Phase = "all"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$TemporalConfig = Join-Path $RepoRoot "configs\mind_large_temporal_text_adapt.yaml"
$ContinuationConfig = Join-Path $RepoRoot "configs\mind_large_submission_text_continue.yaml"
$SubmissionConfig = Join-Path $RepoRoot "configs\mind_large_submission.yaml"
$RecencyConfig = Join-Path $RepoRoot "configs\mind_large_submission_recency_alpha_002.yaml"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual-environment Python was not found at $Python"
}

function Invoke-Mindrec {
    param([string]$Command, [string]$Config)
    Write-Host "`n>>> mindrec $Command --config $Config" -ForegroundColor Cyan
    & $Python -m mindrec.cli $Command --config $Config
    if ($LASTEXITCODE -ne 0) {
        throw "mindrec $Command failed with exit code $LASTEXITCODE"
    }
}

function Test-ProcessedDataset {
    param([string]$DatasetRoot, [string[]]$RequiredArtifacts)
    foreach ($Artifact in $RequiredArtifacts) {
        if (-not (Test-Path -LiteralPath (Join-Path $DatasetRoot $Artifact))) {
            return $false
        }
    }
    return $true
}

Push-Location $RepoRoot
try {
    if ($Phase -in @("phase1", "all")) {
        Write-Host "PHASE 1: Select the MiniLM optimizer-update count" -ForegroundColor Green
        $TemporalData = Join-Path $RepoRoot "data\processed\MINDlarge_temporal_tune"
        $TemporalArtifacts = @(
            "id_maps.json",
            "news.parquet",
            "train_behaviors.parquet",
            "val_behaviors.parquet",
            "val_pairs.parquet",
            "val_impressions.parquet",
            "preprocess_meta.json"
        )
        if (Test-ProcessedDataset $TemporalData $TemporalArtifacts) {
            Write-Host "Reusing existing Large Temporal Train and Large Temporal Val artifacts." -ForegroundColor Yellow
        }
        else {
            Write-Host "Large Temporal Train/Val artifacts are incomplete; rebuilding them." -ForegroundColor Yellow
            Invoke-Mindrec "preprocess" $TemporalConfig
        }
        Invoke-Mindrec "adapt_text_encoder" $TemporalConfig
    }

    if ($Phase -in @("phase2", "all")) {
        Write-Host "PHASE 2: Confirm the selected encoder in the temporal pipeline" -ForegroundColor Green
        $SelectedEncoder = Join-Path $RepoRoot "runs\mind_large_temporal_text_adapt_v1\text_encoder\model"
        if (-not (Test-Path -LiteralPath $SelectedEncoder)) {
            throw "Phase 1 encoder not found at $SelectedEncoder"
        }
        Invoke-Mindrec "train_teacher" $TemporalConfig
        Invoke-Mindrec "train_ranker" $TemporalConfig
        Invoke-Mindrec "evaluate" $TemporalConfig
    }

    if ($Phase -in @("phase3", "all")) {
        Write-Host "PHASE 3: Maximum-data final fit and submission" -ForegroundColor Green
        $SelectionMeta = Join-Path $RepoRoot "runs\mind_large_temporal_text_adapt_v1\text_encoder\meta.json"
        if (-not (Test-Path -LiteralPath $SelectionMeta)) {
            throw "Phase 1 selection metadata not found at $SelectionMeta"
        }
        $Phase2Evaluation = Join-Path $RepoRoot "runs\mind_large_temporal_text_adapt_v1\eval\ranker_eval_val.json"
        if (-not (Test-Path -LiteralPath $Phase2Evaluation)) {
            throw "Phase 2 evaluation not found at $Phase2Evaluation"
        }
        $SubmissionData = Join-Path $RepoRoot "data\processed\MINDlarge_submission"
        $SubmissionArtifacts = @(
            "id_maps.json",
            "news.parquet",
            "train_behaviors.parquet",
            "preprocess_meta.json"
        )
        if (Test-ProcessedDataset $SubmissionData $SubmissionArtifacts) {
            Write-Host "Reusing existing maximum-data Train + Dev artifacts." -ForegroundColor Yellow
        }
        else {
            Write-Host "Maximum-data artifacts are incomplete; rebuilding them." -ForegroundColor Yellow
            Invoke-Mindrec "preprocess" $SubmissionConfig
        }
        Invoke-Mindrec "adapt_text_encoder" $ContinuationConfig
        Invoke-Mindrec "train_teacher" $SubmissionConfig
        Invoke-Mindrec "train_ranker" $SubmissionConfig
        $ItemAge = Join-Path $SubmissionData "item_age_index.npz"
        if (Test-Path -LiteralPath $ItemAge) {
            Write-Host "Reusing existing item-age index." -ForegroundColor Yellow
        }
        else {
            Invoke-Mindrec "build_item_age" $RecencyConfig
        }
        Invoke-Mindrec "write_submission" $RecencyConfig
    }
}
finally {
    Pop-Location
}
