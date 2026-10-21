import sys, torch
def rb(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    sd = d["model_state_dict"]
    W = {k: v.float() for k, v in sd.items() if "layers_rb" in k}
    return d["epoch"], W
e0, A = rb(sys.argv[1]); e1, B = rb(sys.argv[2])
print(f"A=epoch {e0}  B=epoch {e1}")
for k in sorted(A):
    a, b = A[k], B[k]
    if a.dim() == 2:
        # rows of the RB layer should stay ~orthonormal (they are the SVD basis V^T)
        def orth(M):
            G = M @ M.T
            I = torch.eye(G.shape[0])
            return (G - I).abs().max().item(), G.diag().mean().item()
        oa, da = orth(a); ob, db = orth(b)
        print(f"\n{k}  shape {tuple(a.shape)}")
        print(f"   |W|      A {a.norm():.4f}   B {b.norm():.4f}")
        print(f"   |B-A|/|A|            {(b-a).norm()/a.norm():.4f}")
        print(f"   max|WW^T - I|   A {oa:.4g}   B {ob:.4g}")
        print(f"   mean diag(WW^T) A {da:.4g}   B {db:.4g}")
    else:
        print(f"\n{k} (bias) shape {tuple(a.shape)}")
        print(f"   |b|      A {a.norm():.4g}   B {b.norm():.4g}   max|B| {b.abs().max():.4g}")
