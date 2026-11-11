#!/bin/bash
B=build/cuda-release/tilestream_codec_benchmark
for t in uint16 float32; do for c in 0 16384 65536 262144 1048576; do for ty in "char 0" "typed 0" "char 1" "typed 1"; do set -- $ty; for s in lz4_gpu_decompress lz4_gpu_compress; do
$B --scenario $s --type $t --chunk $c --lz4-type $1 --bitshuffle $2 --samples 10 --warmup 3 2>&1 | python3 -c "import sys,json
t=sys.stdin.read()
try:
  d=json.loads(t); m=d['method']; print('$t', m['chunk_bytes'], m['data_type'], m['bitshuffle'], '$s', round(d['bytes']['ratio_payload'],3), round(d['throughput_gbps']['device'],1), 'GB/s', d['memory']['workspace_bytes'])
except Exception: print('$t $c $1 $2 $s FAIL', t[-200:])"
done; done; done; done
