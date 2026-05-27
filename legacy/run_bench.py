import sys
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    from Zeta.runtime import ZetaRuntime
    from Zeta.axioms import AxiomVerifier
    
    print('--- AXIOMS VERIFICATION ---')
    AxiomVerifier.print_report()
    
    print('\n--- INIT ---')
    rt = ZetaRuntime.init(device='cpu', V=256, D=54, N=11, ctx=256)
    
    print('\n--- BENCHMARK ---')
    rt.print_benchmark(B=4, L=64)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
