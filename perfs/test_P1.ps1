
<#
$out = "temps_mickael_P1_no_batch.txt"

if(Test-Path -Path $out) {
    Remove-Item $out
}



if(Test-Path -Path $out) {
    Remove-Item $out
}
#>

Set-Location -Path C:\Users\mteau\source\repos\MergeSort_GPU
Copy-Item cuda_header.h -Destination cuda_header.h.copy
$a = 2
$lim = 8192

echo "Working ..."
while($a -le $lim) {
    (Get-Content cuda_header.h).replace('#define D', "#define D $a //") | Set-Content cuda_header.h
    make merge_sort > $null
    for($i=0 ; $i -lt 5 ; $i++) {
        $(make exec) #>> $out
    }
    
    #echo "" >> $out
    $a = $a * 2
}

Copy-Item cuda_header.h.copy -Destination cuda_header.h
Remove-Item cuda_header.h.copy

#Move-Item $out -Destination perfs\
Set-Location -Path C:\Users\mteau\source\repos\MergeSort_GPU\perfs
