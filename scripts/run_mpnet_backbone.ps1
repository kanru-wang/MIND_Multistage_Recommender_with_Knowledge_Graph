param(
    [ValidateSet("phase1", "phase2", "phase3", "all")]
    [string]$Phase = "phase1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$TemporalConfig = Join-Path $RepoRoot "configs\mind_large_temporal_mpnet.yaml"
$ContinuationConfig = Join-Path $RepoRoot "configs\mind_large_submission_mpnet_text_continue.yaml"
$SubmissionConfig = Join-Path $RepoRoot "configs\mind_large_submission_mpnet.yaml"
$CandidateConfig = Join-Path $RepoRoot "configs\mind_large_submission_mpnet_candidate_attention.yaml"
$RecencyConfig = Join-Path $RepoRoot "configs\mind_large_submission_mpnet_candidate_attention_recency_alpha_002.yaml"
$TemporalRun = "mind_large_temporal_mpnet_candidate_attention_v1"
$SubmissionTeacherRun = "mind_large_submission_mpnet_v1"
$SubmissionRankerRun = "mind_large_submission_mpnet_candidate_attention_low_lr_2ep_v1"
$SubmissionOutputRun = "mind_large_submission_mpnet_candidate_attention_low_lr_2ep_recency_alpha_002_v1"
$MpnetModelName = "sentence-transformers/all-mpnet-base-v2"
$SelectedUpdate = 9000
$ContinuationUpdates = 2000

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual-environment Python was not found at $Python"
}

function Invoke-Mindrec {
    param([string]$Command, [string]$Config = $TemporalConfig)
    Write-Host "`n>>> mindrec $Command --config $Config" -ForegroundColor Cyan
    & $Python -m mindrec.cli $Command --config $Config
    if ($LASTEXITCODE -ne 0) {
        throw "mindrec $Command failed with exit code $LASTEXITCODE"
    }
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required metadata was not found at $Path"
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Test-AllPaths {
    param([string[]]$Paths)
    foreach ($Path in $Paths) {
        if (-not (Test-Path -LiteralPath $Path)) {
            return $false
        }
    }
    return $true
}

function Assert-FreeDiskSpace {
    param([double]$RequiredGB, [string]$Stage)
    $WorkspaceDrive = (Get-Item -LiteralPath $RepoRoot).PSDrive.Name
    $FreeBytes = (Get-PSDrive -Name $WorkspaceDrive).Free
    $FreeGB = [math]::Round($FreeBytes / 1GB, 2)
    if ($FreeBytes -lt ($RequiredGB * 1GB)) {
        throw "$Stage requires at least $RequiredGB GB free on drive $WorkspaceDrive before it starts; only $FreeGB GB is available. Free disk space and rerun -Phase phase3. Compatible completed stages will be reused."
    }
    Write-Host "Disk-space preflight passed for $Stage`: $FreeGB GB free (minimum $RequiredGB GB)." -ForegroundColor Green
}

function Assert-MpnetTemporalSelection {
    $EncoderRoot = Join-Path $RepoRoot "runs\$TemporalRun\text_encoder"
    $EncoderMeta = Read-JsonFile (Join-Path $EncoderRoot "meta.json")
    if ($EncoderMeta.base_model_name -ne $MpnetModelName) {
        throw "Phase 3 requires $MpnetModelName, but Phase 1 metadata reports $($EncoderMeta.base_model_name)."
    }
    if ([int]$EncoderMeta.best_update -ne $SelectedUpdate) {
        throw "Phase 3 is locked to the selected $SelectedUpdate-update checkpoint; metadata reports $($EncoderMeta.best_update)."
    }
    if ([int]$EncoderMeta.batch_size -ne 16 -or
        [int]$EncoderMeta.gradient_accumulation_steps -ne 4 -or
        [int]$EncoderMeta.max_optimizer_updates -ne 10000 -or
        [int]$EncoderMeta.completed_optimizer_updates -ne 10000) {
        throw "Phase 1 did not use the locked batch-16, accumulation-4, 10,000-update adaptation schedule."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $EncoderRoot "model\modules.json"))) {
        throw "The selected Phase 1 encoder model is missing from $EncoderRoot."
    }

    $EvaluationPath = Join-Path $RepoRoot "runs\$TemporalRun\eval\ranker_eval_val.json"
    $Evaluation = Read-JsonFile $EvaluationPath
    if ([int]$Evaluation.n_impressions -ne 807988) {
        throw "Phase 2 evaluation used $($Evaluation.n_impressions) impressions instead of the expected 807988."
    }
    if ([double]$Evaluation.ranking.auc -lt 0.671592574) {
        throw "MPNet temporal AUC $($Evaluation.ranking.auc) does not beat the controlled MiniLM reference; Phase 3 is blocked."
    }
    Write-Host "Preflight passed: MPNet update $SelectedUpdate, temporal AUC $([math]::Round([double]$Evaluation.ranking.auc, 6))." -ForegroundColor Green
}

function Test-CompatibleContinuation {
    $MetaPath = Join-Path $RepoRoot "runs\$SubmissionTeacherRun\text_encoder\meta.json"
    $ModelPath = Join-Path $RepoRoot "runs\$SubmissionTeacherRun\text_encoder\model\modules.json"
    if (-not (Test-Path -LiteralPath $MetaPath)) {
        return $false
    }
    $Meta = Read-JsonFile $MetaPath
    if ($Meta.base_model_name -ne $MpnetModelName -or
        [int]$Meta.initial_model_update -ne $SelectedUpdate -or
        [int]$Meta.completed_optimizer_updates -ne $ContinuationUpdates -or
        [int]$Meta.cumulative_optimizer_updates -ne ($SelectedUpdate + $ContinuationUpdates) -or
        [int]$Meta.batch_size -ne 16 -or
        [int]$Meta.gradient_accumulation_steps -ne 4 -or
        $Meta.training_split -ne "val") {
        throw "Existing Phase 3 encoder metadata is incompatible. Refusing to overwrite it automatically: $MetaPath"
    }
    return (Test-Path -LiteralPath $ModelPath)
}

function Test-CompatibleTeacher {
    $TeacherRoot = Join-Path $RepoRoot "runs\$SubmissionTeacherRun\teacher"
    $Required = @(
        (Join-Path $TeacherRoot "model.pt"),
        (Join-Path $TeacherRoot "item_base_emb.npy"),
        (Join-Path $TeacherRoot "item_ranker_base_emb.npy"),
        (Join-Path $TeacherRoot "item_teacher_emb.npy"),
        (Join-Path $TeacherRoot "user_teacher_emb.npy"),
        (Join-Path $TeacherRoot "meta.json")
    )
    if (-not (Test-AllPaths $Required)) {
        return $false
    }
    $Meta = Read-JsonFile (Join-Path $TeacherRoot "meta.json")
    if ($Meta.model_name -ne $MpnetModelName -or
        [int]$Meta.item_base_dim -ne 768 -or
        [int]$Meta.epochs -ne 4 -or
        [int]$Meta.best_epoch -ne 4 -or
        $Meta.selection_mode -ne "fixed_epoch") {
        throw "Existing MPNet submission teacher is incompatible: $(Join-Path $TeacherRoot 'meta.json')"
    }
    return $true
}

function Test-CompatibleRanker {
    $RankerRoot = Join-Path $RepoRoot "runs\$SubmissionRankerRun\ranker"
    $Required = @(
        (Join-Path $RankerRoot "best.pt"),
        (Join-Path $RankerRoot "train_summary.json")
    )
    if (-not (Test-AllPaths $Required)) {
        return $false
    }
    $Summary = Read-JsonFile (Join-Path $RankerRoot "train_summary.json")
    if ($Summary.teacher_artifact_run_name -ne $SubmissionTeacherRun -or
        $Summary.history_pooling -ne "candidate_attention" -or
        [int]$Summary.best_epoch -ne 2 -or
        $Summary.selection_mode -ne "fixed_epoch") {
        throw "Existing MPNet submission ranker is incompatible: $(Join-Path $RankerRoot 'train_summary.json')"
    }
    return $true
}

function Test-CompatibleSubmission {
    $SubmissionRoot = Join-Path $RepoRoot "runs\$SubmissionOutputRun\submission"
    $SubmissionZip = Join-Path $SubmissionRoot "prediction.zip"
    $SubmissionMeta = Join-Path $SubmissionRoot "submission_meta.json"
    if (-not (Test-AllPaths @($SubmissionZip, $SubmissionMeta))) {
        return $false
    }
    $Meta = Read-JsonFile $SubmissionMeta
    if ([int]$Meta.n_impressions -ne 2370727 -or
        $Meta.history_pooling -ne "candidate_attention" -or
        -not [bool]$Meta.posthoc_recency.enabled -or
        [double]$Meta.posthoc_recency.alpha -ne 0.02) {
        throw "Existing MPNet submission output is incompatible: $SubmissionMeta"
    }
    return $true
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
        Write-Host "PHASE 1: Select the MPNet optimizer-update count on Large Temporal Val" -ForegroundColor Green
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
            Write-Host "Reusing the proven Large Temporal Train/Val distribution." -ForegroundColor Yellow
        }
        else {
            Write-Host "Large Temporal Train/Val artifacts are incomplete; rebuilding them." -ForegroundColor Yellow
            Invoke-Mindrec "preprocess"
        }
        Invoke-Mindrec "adapt_text_encoder"
    }

    if ($Phase -in @("phase2", "all")) {
        Write-Host "PHASE 2: Train and evaluate the selected pipeline with MPNet" -ForegroundColor Green
        $SelectedEncoder = Join-Path $RepoRoot "runs\mind_large_temporal_mpnet_candidate_attention_v1\text_encoder\model"
        if (-not (Test-Path -LiteralPath $SelectedEncoder)) {
            throw "Phase 1 encoder not found at $SelectedEncoder. Run this script with -Phase phase1 first."
        }
        Invoke-Mindrec "train_teacher"
        Invoke-Mindrec "train_ranker"
        Invoke-Mindrec "evaluate"

        $Evaluation = Join-Path $RepoRoot "runs\mind_large_temporal_mpnet_candidate_attention_v1\eval\ranker_eval_val.json"
        Write-Host "Temporal evaluation complete: $Evaluation" -ForegroundColor Green
        Write-Host "Review it against candidate-attention MiniLM AUC 0.671593 before implementing any maximum-data submission." -ForegroundColor Yellow
    }

    if ($Phase -in @("phase3", "all")) {
        Write-Host "PHASE 3: Continue selected MPNet on maximum data and write the hidden-test submission" -ForegroundColor Green
        Assert-MpnetTemporalSelection

        $SubmissionData = Join-Path $RepoRoot "data\processed\MINDlarge_submission"
        $SubmissionArtifacts = @(
            "id_maps.json",
            "news.parquet",
            "train_behaviors.parquet",
            "item_click_counts.json",
            "preprocess_meta.json"
        )
        if (Test-ProcessedDataset $SubmissionData $SubmissionArtifacts) {
            Write-Host "Reusing complete MINDlarge_submission preprocessing artifacts." -ForegroundColor Yellow
        }
        else {
            Write-Host "Submission preprocessing artifacts are incomplete; rebuilding them without reading labels from the hidden test set." -ForegroundColor Yellow
            Invoke-Mindrec "preprocess" $SubmissionConfig
        }
        $PreprocessMetaPath = Join-Path $SubmissionData "preprocess_meta.json"
        $PreprocessMeta = Read-JsonFile $PreprocessMetaPath
        if ($PreprocessMeta.mode -ne "leaderboard_submission" -or
            [int]$PreprocessMeta.n_submission_impressions -ne 2370727) {
            throw "Submission preprocessing metadata is incompatible: $PreprocessMetaPath"
        }
        $HiddenTestBehaviors = Join-Path $RepoRoot "data\raw\MINDlarge_test\behaviors.tsv"
        if (-not (Test-Path -LiteralPath $HiddenTestBehaviors)) {
            throw "MINDlarge hidden-test behaviors were not found at $HiddenTestBehaviors"
        }

        if (Test-CompatibleContinuation) {
            Write-Host "Reusing compatible MPNet continuation: exactly $ContinuationUpdates updates after selected update $SelectedUpdate." -ForegroundColor Yellow
        }
        else {
            # The continuation model plus subsequent 768-dimensional teacher
            # arrays require several GB. Fail before a multi-hour training run,
            # not while serializing its final weights.
            Assert-FreeDiskSpace 6.0 "MPNet continuation and downstream artifacts"
            Invoke-Mindrec "adapt_text_encoder" $ContinuationConfig
            if (-not (Test-CompatibleContinuation)) {
                throw "MPNet continuation completed without producing the expected fixed-update artifacts."
            }
        }

        if (Test-CompatibleTeacher) {
            Write-Host "Reusing compatible fixed four-epoch MPNet teacher." -ForegroundColor Yellow
        }
        else {
            Assert-FreeDiskSpace 3.0 "MPNet teacher artifacts"
            Invoke-Mindrec "train_teacher" $SubmissionConfig
            if (-not (Test-CompatibleTeacher)) {
                throw "MPNet teacher training completed without the expected fixed four-epoch artifacts."
            }
        }

        if (Test-CompatibleRanker) {
            Write-Host "Reusing compatible fixed two-epoch candidate-attention ranker." -ForegroundColor Yellow
        }
        else {
            Assert-FreeDiskSpace 1.0 "MPNet ranker artifacts"
            Invoke-Mindrec "train_ranker" $CandidateConfig
            if (-not (Test-CompatibleRanker)) {
                throw "MPNet ranker training completed without the expected fixed two-epoch artifacts."
            }
        }

        $ItemAgePath = Join-Path $SubmissionData "item_age_index.npz"
        if (Test-Path -LiteralPath $ItemAgePath) {
            Write-Host "Reusing submission item-age artifact." -ForegroundColor Yellow
        }
        else {
            Invoke-Mindrec "build_item_age" $RecencyConfig
        }

        $SubmissionRoot = Join-Path $RepoRoot "runs\$SubmissionOutputRun\submission"
        $SubmissionZip = Join-Path $SubmissionRoot "prediction.zip"
        if (Test-CompatibleSubmission) {
            Write-Host "Reusing complete MPNet Phase 3 submission: $SubmissionZip" -ForegroundColor Green
        }
        else {
            Assert-FreeDiskSpace 0.5 "MPNet submission output"
            Invoke-Mindrec "write_submission" $RecencyConfig
            if (-not (Test-CompatibleSubmission)) {
                throw "Submission scoring completed without a compatible 2,370,727-impression archive."
            }
            Write-Host "MPNet Phase 3 submission complete: $SubmissionZip" -ForegroundColor Green
        }
    }
}
finally {
    Pop-Location
}
