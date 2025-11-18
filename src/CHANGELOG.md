All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.2.1 - 18/11/2025

### Added
- Added mixed precision training support (PyTorch AMP)
- Added automatic dtype detection (bf16 or fp16)
- Added gradient scaling for numerical stability

### Fixed 

**Critical** - FlashAttention dtype Error
**Root Cause**
- FlashAttention library requires tensors in fp16 or bf16 precision. Model was running in fp32 (PyTorch default)
- No automatic dtype conversion in attention block

### Changed

1. Updated src/model/attention_block.py
- Added automatic dtype detection and conversion
- Converts fp32 → bf16/fp16 before FlashAttention
- Converts back to fp32 after attention
- Added fallback to standard attention if FlashAttention fails
- Added import guard for FlashAttention availability

2. Updated src/train.py 

#### API Changes

 AttentionBlock(d_model, n_heads=8, dropout=0.1, use_flash_attn=True)
- use_flash_attn: Enable/disable FlashAttention (default: True)

- config.use_mixed_precision = True  # Default: True
