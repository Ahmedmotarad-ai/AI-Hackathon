$chunks = Get-Content data/chunks/chunks.jsonl | ForEach-Object { $_ | ConvertFrom-Json }

function Show-Chunk($c) {
    $textPreview = if ($c.text.Length -gt 500) { $c.text.Substring(0, 500) + "..." } else { $c.text }
    Write-Output "=== chunk_id: $($c.chunk_id) | document: $($c.document) | section: $($c.section) | page: $($c.page) ==="
    Write-Output $textPreview
    Write-Output ""
}

Write-Output "============================================"
Write-Output "FROM ESC_2023: HFmrEF sections"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -like "esc_hf_2023_focused_update_chunk_*" -and $_.section -like "*HFmrEF*" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM ESC_2023: HFpEF sections"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -like "esc_hf_2023_focused_update_chunk_*" -and $_.section -like "*HFpEF*" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM ESC_2023: Recommendation Table"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -like "esc_hf_2023_focused_update_chunk_*" -and $_.section -like "*Recommendation Table*" } | ForEach-Object {
    $preview = if ($_.text.Length -le 800) { $_.text } else { $_.text.Substring(0, 800) + "..." }
    Write-Output "=== chunk_id: $($_.chunk_id) | document: $($_.document) | section: $($_.section) | page: $($_.page) ==="
    Write-Output $preview
    Write-Output ""
}

Write-Output "============================================"
Write-Output "FROM ESC_2023: Treatment mentioning SGLT2/dapa/empa (first 3)"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -like "esc_hf_2023_focused_update_chunk_*" -and $_.section -like "*Treatment*" -and ($_.text -match "SGLT2|dapagliflozin|empagliflozin") } | Select-Object -First 3 | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM ESC_2023: Diabetes (first 2)"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -like "esc_hf_2023_focused_update_chunk_*" -and $_.section -like "*Diabetes*" } | Select-Object -First 2 | ForEach-Object {
    $preview = if ($_.text.Length -gt 400) { $_.text.Substring(0, 400) + "..." } else { $_.text }
    Write-Output "=== chunk_id: $($_.chunk_id) | document: $($_.document) | section: $($_.section) | page: $($_.page) ==="
    Write-Output $preview
    Write-Output ""
}

Write-Output "============================================"
Write-Output "FROM ESC_2021: chunks 0010-0015"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "esc_hf_2021_chunk_001[0-5]" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM ESC_2021: chunks 0145-0155"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "esc_hf_2021_chunk_014[5-9]|esc_hf_2021_chunk_015[0-5]" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM ESC_2021: chunks 0180-0190"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "esc_hf_2021_chunk_018[0-9]|esc_hf_2021_chunk_0190" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM ESC_2021: chunks 0298-0310"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "esc_hf_2021_chunk_029[89]|esc_hf_2021_chunk_030[0-9]|esc_hf_2021_chunk_0310" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM ESC_2021: chunks 0340-0355"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "esc_hf_2021_chunk_034[0-9]|esc_hf_2021_chunk_035[0-5]" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM ESC_2021: chunks 0375-0385"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "esc_hf_2021_chunk_037[5-9]|esc_hf_2021_chunk_038[0-5]" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM ESC_2021: chunks 0475-0485"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "esc_hf_2021_chunk_047[5-9]|esc_hf_2021_chunk_048[0-5]" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM ESC_2021: chunks 0498-0510"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "esc_hf_2021_chunk_049[89]|esc_hf_2021_chunk_050[0-9]|esc_hf_2021_chunk_0510" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM NICE_2018: chunks 0006-0008"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "nice_hf_2018_chunk_000[6-8]" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM NICE_2018: chunks 0010-0014"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "nice_hf_2018_chunk_001[0-4]" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM NICE_2018: chunks 0016-0020"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "nice_hf_2018_chunk_001[6-9]|nice_hf_2018_chunk_0020" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM NICE_2018: chunks 0021-0028"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "nice_hf_2018_chunk_002[1-8]" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM NICE_2018: chunks 0033-0038"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "nice_hf_2018_chunk_003[3-8]" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM NICE_2018: chunks 0039-0043"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "nice_hf_2018_chunk_0039|nice_hf_2018_chunk_004[0-3]" } | ForEach-Object { Show-Chunk $_ }

Write-Output "============================================"
Write-Output "FROM NICE_2018: chunks 0046-0050"
Write-Output "============================================"
$chunks | Where-Object { $_.chunk_id -match "nice_hf_2018_chunk_004[6-9]|nice_hf_2018_chunk_0050" } | ForEach-Object { Show-Chunk $_ }
