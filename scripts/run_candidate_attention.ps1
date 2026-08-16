param(
    [ValidateSet("phase2", "phase3", "all")]
    [string]$Phase = "phase2"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$TemporalConfig = Join-Path $RepoRoot "configs\mind_large_temporal_candidate_attention.yaml"
$SubmissionConfig = Join-Path $RepoRoot "configs\mind_large_submission_candidate_attention.yaml"
$RecencyConfig = Join-Path $RepoRoot "configs\mind_large_submission_candidate_attention_recency_alpha_002.yaml"

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

function Assert-Artifacts {
    param([string[]]$Paths, [string]$Hint)
    $Missing = @(
        $Paths | Where-Object { -not (Test-Path -LiteralPath $_) }
    )
    if ($Missing.Count -gt 0) {
        $MissingList = ($Missing | ForEach-Object { "  - $_" }) -join "`n"
        throw "Required artifacts were not found:`n$MissingList`n$Hint"
    }
}

Push-Location $RepoRoot
try {
    if ($Phase -in @("phase2", "all")) {
        Write-Host "PHASE 2: Validate candidate-aware history attention" -ForegroundColor Green
        $TemporalData = Join-Path $RepoRoot "data\processed\MINDlarge_temporal_tune"
        $TemporalTeacher = Join-Path $RepoRoot "runs\mind_large_temporal_text_adapt_v1\teacher"
        Assert-Artifacts @(
            (Join-Path $TemporalData "id_maps.json"),
            (Join-Path $TemporalData "news.parquet"),
            (Join-Path $TemporalData "train_behaviors.parquet"),
            (Join-Path $TemporalData "val_pairs.parquet"),
            (Join-Path $TemporalData "val_impressions.parquet"),
            (Join-Path $TemporalData "val_behaviors.parquet"),
            (Join-Path $TemporalData "item_click_counts.json"),
            (Join-Path $TemporalTeacher "model.pt"),
            (Join-Path $TemporalTeacher "item_ranker_base_emb.npy"),
            (Join-Path $TemporalTeacher "item_teacher_emb.npy")
        ) "Run .\scripts\run_text_adaptation.ps1 -Phase phase2 first."
        Invoke-Mindrec "train_ranker" $TemporalConfig
        Invoke-Mindrec "evaluate" $TemporalConfig
    }

    if ($Phase -in @("phase3", "all")) {
        Write-Host "PHASE 3: Fit the selected low-LR two-epoch candidate-aware ranker and write submission" -ForegroundColor Green
        $TemporalEvaluation = Join-Path $RepoRoot "runs\mind_large_temporal_text_adapt_candidate_attention_v1\eval\ranker_eval_val.json"
        $SubmissionData = Join-Path $RepoRoot "data\processed\MINDlarge_submission"
        $SubmissionTeacher = Join-Path $RepoRoot "runs\mind_large_submission_text_adapt_v1\teacher"
        $HiddenTestBehaviors = Join-Path $RepoRoot "data\raw\MINDlarge_test\behaviors.tsv"
        Assert-Artifacts @(
            $TemporalEvaluation,
            (Join-Path $SubmissionData "id_maps.json"),
            (Join-Path $SubmissionData "news.parquet"),
            (Join-Path $SubmissionData "train_behaviors.parquet"),
            (Join-Path $SubmissionData "item_click_counts.json"),
            (Join-Path $SubmissionTeacher "model.pt"),
            (Join-Path $SubmissionTeacher "item_ranker_base_emb.npy"),
            (Join-Path $SubmissionTeacher "item_teacher_emb.npy"),
            $HiddenTestBehaviors
        ) "Run Phase 2 and review its validation result, then run .\scripts\run_text_adaptation.ps1 -Phase phase3 if the maximum-data teacher is missing."

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
