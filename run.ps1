param(
    [int]$NUM,      # 第一个参数：启动数量
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args # 剩余参数传给 client.py
)

$processes = @()

for ($i = 1; $i -le $NUM; $i++) {
    # 启动 python 客户端并传递参数
    $p = Start-Process python -ArgumentList @("client.py") + $Args -PassThru
    $processes += $p
}

# 等待全部进程结束
foreach ($p in $processes) {
    $p.WaitForExit()
}