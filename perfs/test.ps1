
$out = "temps_mickael_P1.txt"
Remove-Item $out
Copy-Item cuda_header.h -Destination cuda_header.h.copy

$a = 2
$lim = 67108864

echo "Working ..."
while($a -le $lim) {
    (Get-Content cuda_header.h).replace('#define D', "#define D $a //") | Set-Content cuda_header.h
    make merge_sort > $null
    $(make exec) >> $out
    echo "" >> $out
    $a = $a * 2
}

Copy-Item cuda_header.h.copy -Destination cuda_header.h
Remove-Item cuda_header.h.copy
