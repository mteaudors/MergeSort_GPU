
$out = "temps_mickael_P2.txt"

if(Test-Path -Path $out) {
    Remove-Item $out
}

Set-Location -Path C:\Users\mteau\source\repos\MergeSort_GPU

if(Test-Path -Path $out) {
    Remove-Item $out
}

Copy-Item cuda_header.h -Destination cuda_header.h.copy

$a = 2
$N = 2048

echo "Working ..."

while($a -le 512) {
    (Get-Content cuda_header.h).replace('#define SIZE_A', "#define SIZE_A $a //") | Set-Content cuda_header.h
    (Get-Content cuda_header.h).replace('#define SIZE_B', "#define SIZE_B $a //") | Set-Content cuda_header.h
    make merge_batch > $null
    $(.\merge_batch) >> $out
    echo "" >> $out
    $a = $a * 2
}

while($N -le 131072) {
    (Get-Content cuda_header.h).replace('#define N', "#define N $N //") | Set-Content cuda_header.h
    make merge_batch > $null
    $(.\merge_batch) >> $out
    echo "" >> $out
    $N = $N * 2
}


Copy-Item cuda_header.h.copy -Destination cuda_header.h
Remove-Item cuda_header.h.copy

Move-Item $out -Destination perfs\
Set-Location -Path C:\Users\mteau\source\repos\MergeSort_GPU\perfs