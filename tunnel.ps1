# 建立 h3_sm75_server 双向 SSH 隧道
# -L 8190: 映射远程 dit server / WebUI 到本机 127.0.0.1:8190
# -R 9310: 映射远程回调端口到本机 cond server (127.0.0.1:8189)
# 注意：必须显式 127.0.0.1，勿写 localhost（防止 sshd 优先解析 IPv6 ::1 导致断连）

$HostName = "bwg.aki.pm"
$Port     = 56622
$User     = "sheepserver"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " 启动 h3_sm75_server 双向 SSH 隧道" -ForegroundColor Cyan
Write-Host " 本机 WebUI : http://127.0.0.1:8190" -ForegroundColor Green
Write-Host " 远程回调   : 127.0.0.1:9310 -> 本机 :8189" -ForegroundColor Green
Write-Host " 按 Ctrl+C 即可终止隧道" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

ssh -N -p $Port `
    -L 8190:127.0.0.1:8190 `
    -R 9310:127.0.0.1:8189 `
    "${User}@${HostName}"
