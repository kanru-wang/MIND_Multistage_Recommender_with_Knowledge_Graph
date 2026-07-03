param(
  [string]$Config="configs/mind_small.yaml"
)

$ErrorActionPreference = "Stop"

function Invoke-MindRec {
  param([string]$Command)

  & python -m mindrec.cli $Command --config $Config
  if ($LASTEXITCODE -ne 0) {
    throw "mindrec command failed: $Command"
  }
}

$Mode = & python -c "import sys; from mindrec.config import load_config; print(load_config(sys.argv[1]).get('data', {}).get('mode', 'standard'))" $Config
if ($LASTEXITCODE -ne 0) {
  throw "Could not read config mode from $Config"
}
$Mode = $Mode.Trim()
$TeacherArtifactRun = & python -c "import sys; from mindrec.config import load_config; c=load_config(sys.argv[1]); print(c.get('artifacts', {}).get('teacher_run_name', ''))" $Config
if ($LASTEXITCODE -ne 0) {
  throw "Could not read teacher artifact settings from $Config"
}
$TeacherArtifactRun = $TeacherArtifactRun.Trim()

Invoke-MindRec "preprocess"
if ($TeacherArtifactRun) {
  Write-Host "Reusing teacher artifacts from run: $TeacherArtifactRun"
} else {
  Invoke-MindRec "train_teacher"
}

if ($Mode -eq "leaderboard_submission") {
  Invoke-MindRec "train_ranker"
  Invoke-MindRec "write_submission"
  return
}

Invoke-MindRec "build_index"
Invoke-MindRec "eval_retrieval"
Invoke-MindRec "eval_retrieval_sweep"
Invoke-MindRec "train_ranker"
Invoke-MindRec "evaluate"
Invoke-MindRec "rerank_search"
Invoke-MindRec "rerank_eval"
