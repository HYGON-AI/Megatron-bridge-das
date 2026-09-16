# usage

```Shell

# Multi-Node
#     list one hostname/IP per line
#     cat > hostfile <<'EOF'
#     node01
#     node02
#     EOF
cd examples/deepseek
bash train_deepseekv4_pro_pretrain.sh deepseek_v4_pro_pretrain_256gpu_bw1000_bf16_test_config train hostfile 32
# ==============================================================================
```
