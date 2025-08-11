param(
    [int]$NUM,  # 第一个参数：启动数量
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args # 剩余参数传给 client.py
)

$processes = @()

for ($i = 1; $i -le $NUM; $i++) {
    # 构造参数数组，过滤空值
    $argsList = @("client.py")
    if ($Args) {
        $argsList += $Args | Where-Object { $_ -ne $null -and $_.Trim() -ne "" }
    }

    # 启动进程并返回进程对象
    $p = Start-Process python -ArgumentList $argsList -PassThru
    if ($p) {
        $processes += $p
    } else {
        Write-Host "启动进程失败：python $argsList"
    }
}

# 等待所有进程结束
foreach ($p in $processes) {
    $p.WaitForExit()
}
