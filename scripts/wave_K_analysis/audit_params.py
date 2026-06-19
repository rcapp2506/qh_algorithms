"""Audit parameter count delle due architetture in Wave-K R=10.

Riproduco fedelmente l'__init__ di HybridConvNet (run_qcnn_multiseed.py)
e di ClassicalConvNet (classical_cnn_multiseed_stats_v1.ipynb) e conto:
- param totali
- param trainable
- param per blocco (trunk vs head)

Per Linear(N->M): N*M + M (bias)
Per Conv2d(C_in->C_out, ks=k, padding=p): C_out*C_in*k*k + C_out (bias)
Per BatchNorm2d(C): 2*C (gamma, beta) + 2*C buffer (running mean/var, non trainable)
Per nn.Dropout / MaxPool / ReLU: 0 trainable param
"""

# ---------- QCNN ----------
# Da run_qcnn_multiseed.py HybridConvNet (config default Wave-K):
# config.num_conv_channels = 6
# config.conv_kernel_size = 5
# config.conv_padding = 2
# config.dropout_rate = 0.0
# Input 64x64x3
# Quanv layer: 9 qubit, kernel 3, output channels 6, num_weights = 9
# input_scale = nn.Parameter(torch.ones(9)) — 9 param trainable

CH = 6
KS = 5
PAD = 2
NUM_QUBITS = 9
NUM_WEIGHTS = 9
IN_CH = 3

# Conv1: 3 -> 6, ks=5, pad=2 (Sequential: Conv2d + BN + ReLU + MaxPool)
qcnn_conv1 = CH * IN_CH * KS * KS + CH            # 6*3*25 + 6 = 456
qcnn_bn1   = 2 * CH                                # 12
qcnn_conv2 = CH * CH * KS * KS + CH                # 6*6*25 + 6 = 906
qcnn_bn2   = 2 * CH                                # 12

# QuantumConvLayer:
#   input_scale: nn.Parameter(torch.ones(9))     -> 9
#   quantum_weights: nn.Parameter(num_weights=9) -> 9
qcnn_quanv_input_scale = NUM_QUBITS                # 9
qcnn_quanv_weights     = NUM_WEIGHTS               # 9

# Flatten size = num_conv_channels * quanv_output_size^2
# img_size=64 -> feature_map_size = 64//4 = 16 -> quanv_output_size = (16-3)/1 + 1 = 14
# flatten_size = 6 * 14 * 14 = 1176
flat = CH * 14 * 14
fc1_out = max(flat // 3, 64)                        # max(392, 64) = 392
qcnn_fc1 = flat * fc1_out + fc1_out                 # 1176*392 + 392 = 461,384
qcnn_fc2 = fc1_out * 2 + 2                          # 392*2 + 2 = 786

qcnn_total = (qcnn_conv1 + qcnn_bn1 + qcnn_conv2 + qcnn_bn2
              + qcnn_quanv_input_scale + qcnn_quanv_weights
              + qcnn_fc1 + qcnn_fc2)

print("=" * 60)
print("QCNN — HybridConvNet (Wave-K defaults, num_qubits=9, ch=6)")
print("=" * 60)
print(f"  Conv1 (3->6, ks=5, pad=2):        {qcnn_conv1:>10,}")
print(f"  BatchNorm1 (6):                  {qcnn_bn1:>10,}")
print(f"  Conv2 (6->6, ks=5, pad=2):        {qcnn_conv2:>10,}")
print(f"  BatchNorm2 (6):                  {qcnn_bn2:>10,}")
print(f"  Quanv input_scale (9):           {qcnn_quanv_input_scale:>10,}")
print(f"  Quanv quantum_weights (9):       {qcnn_quanv_weights:>10,}")
print(f"  fc1 Linear({flat}->{fc1_out}):           {qcnn_fc1:>10,}")
print(f"  fc2 Linear({fc1_out}->2):              {qcnn_fc2:>10,}")
print(f"  {'TOTAL':>32}: {qcnn_total:>10,}")
print(f"  (Lightning reports 463K -> match? {abs(qcnn_total - 463000) < 1000})")
print()

# ---------- CCNN matched-capacity ----------
# Da classical_cnn_multiseed_stats_v1.ipynb ClassicalConvNet
# Conv(3->16, ks=3, pad=1) + Pool + Dropout
# Conv(16->32, ks=3, pad=1) + Pool + Dropout
# Conv(32->64, ks=3, pad=1) + Dropout                    ← NO POOL
# Conv(64->64, ks=3, pad=1) + Dropout                    ← sostituto quanv
# Flatten + LazyLinear(?->5221) + Linear(5221->2)

ccnn_conv1 = 16 * 3 * 9 + 16              # 448
ccnn_conv2 = 32 * 16 * 9 + 32             # 4640
ccnn_conv3 = 64 * 32 * 9 + 64             # 18496
ccnn_conv4 = 64 * 64 * 9 + 64             # 36928

# Forward shape computation
# 64x64x3 -> Conv1 ks=3 pad=1 -> 64x64x16 -> Pool -> 32x32x16
# -> Conv2 -> 32x32x32 -> Pool -> 16x16x32
# -> Conv3 -> 16x16x64 (NO pool, NO pool elsewhere)
# -> Conv4 -> 16x16x64
# -> Flatten -> 64*16*16 = 16384 features
ccnn_flat = 64 * 16 * 16
ccnn_fc1  = ccnn_flat * 5221 + 5221       # 16384*5221 + 5221 = 85,544,485
ccnn_fc2  = 5221 * 2 + 2                  # 10,444

ccnn_total = ccnn_conv1 + ccnn_conv2 + ccnn_conv3 + ccnn_conv4 + ccnn_fc1 + ccnn_fc2

print("=" * 60)
print("CCNN — ClassicalConvNet (matched-capacity to C16-Q64)")
print("=" * 60)
print(f"  Conv1 (3->16, ks=3, pad=1):       {ccnn_conv1:>12,}")
print(f"  Conv2 (16->32, ks=3, pad=1):      {ccnn_conv2:>12,}")
print(f"  Conv3 (32->64, ks=3, pad=1):      {ccnn_conv3:>12,}")
print(f"  Conv4 (64->64, ks=3, pad=1):      {ccnn_conv4:>12,}  [replacement quanv]")
print(f"  fc1 LazyLinear({ccnn_flat}->5221):     {ccnn_fc1:>12,}")
print(f"  fc2 Linear(5221->2):              {ccnn_fc2:>12,}")
print(f"  {'TOTAL':>32}: {ccnn_total:>12,}")
print()

# ---------- Confronto ----------
print("=" * 60)
print("CONFRONTO")
print("=" * 60)
print(f"  QCNN total params:  {qcnn_total:>12,}")
print(f"  CCNN total params:  {ccnn_total:>12,}")
print(f"  Ratio CCNN/QCNN:    {ccnn_total / qcnn_total:>12.1f}x")
print()
print(f"  QCNN trunk (conv+BN+quanv):    {qcnn_conv1+qcnn_bn1+qcnn_conv2+qcnn_bn2+qcnn_quanv_input_scale+qcnn_quanv_weights:>10,}")
print(f"  CCNN trunk (4 conv):           {ccnn_conv1+ccnn_conv2+ccnn_conv3+ccnn_conv4:>10,}")
print(f"  -> CCNN trunk has ~{(ccnn_conv1+ccnn_conv2+ccnn_conv3+ccnn_conv4) / (qcnn_conv1+qcnn_bn1+qcnn_conv2+qcnn_bn2+qcnn_quanv_input_scale+qcnn_quanv_weights):.0f}x more trunk params than QCNN")
print()
print(f"  QCNN head (fc1+fc2):           {qcnn_fc1+qcnn_fc2:>10,}")
print(f"  CCNN head (fc1+fc2):           {ccnn_fc1+ccnn_fc2:>10,}")
print(f"  -> CCNN head has ~{(ccnn_fc1+ccnn_fc2) / (qcnn_fc1+qcnn_fc2):.0f}x more head params than QCNN")
print()
print(f"  QCNN flatten input dim:        {flat:>10,}  (6 channels x 14x14)")
print(f"  CCNN flatten input dim:        {ccnn_flat:>10,}  (64 channels x 16x16)")
print(f"  -> CCNN sees {ccnn_flat / flat:.1f}x more features at the linear head")
print()
print("Quantum trainable parameters in QCNN: %d (= 9 RZ angles + 9 input scales)" %
      (qcnn_quanv_input_scale + qcnn_quanv_weights))
print()
print("CONCLUSION: 'matched-capacity' is severely violated.")
print("  The CCNN has ~%dx more parameters than the QCNN." % round(ccnn_total / qcnn_total))
print("  The mismatch is dominated by the head (Linear flatten input).")
