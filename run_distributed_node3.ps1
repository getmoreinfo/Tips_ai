# 노드 3 실행 스크립트
# 사용법: PowerShell에서 .\run_distributed_node3.ps1 실행

# 메인 노드의 IP 주소를 여기에 설정하세요
$MASTER_ADDR = "192.168.1.100"  # 메인 노드의 IP 주소로 변경
$MASTER_PORT = "29500"
$NODE_RANK = "2"
$NUM_NODES = "3"

Write-Host "=========================================="
Write-Host "멀티 노드 분산 학습 시작 (노드 3)"
Write-Host "=========================================="
Write-Host "Master Address: $MASTER_ADDR"
Write-Host "Master Port: $MASTER_PORT"
Write-Host "Node Rank: $NODE_RANK"
Write-Host "Total Nodes: $NUM_NODES"
Write-Host "=========================================="
Write-Host ""

python -m torch.distributed.launch `
    --nproc_per_node=1 `
    --nnodes=$NUM_NODES `
    --node_rank=$NODE_RANK `
    --master_addr=$MASTER_ADDR `
    --master_port=$MASTER_PORT `
    02_finetune_distributed.py
