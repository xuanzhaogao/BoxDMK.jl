using BoxDMK
using LinearAlgebra
using Test

function _apply_dim_reference(src, mat, n, nd, dim)
    out = similar(src)

    for iv in 1:nd
        src_3d = reshape(@view(src[iv, :]), n, n, n)
        out_3d = reshape(@view(out[iv, :]), n, n, n)

        @inbounds for i3 in 1:n, i2 in 1:n, i1 in 1:n
            acc = zero(eltype(out))
            if dim == 1
                for k1 in 1:n
                    acc += mat[i1, k1] * src_3d[k1, i2, i3]
                end
                out_3d[i1, i2, i3] = acc
            elseif dim == 2
                for k2 in 1:n
                    acc += mat[i2, k2] * src_3d[i1, k2, i3]
                end
                out_3d[i1, i2, i3] = acc
            else
                for k3 in 1:n
                    acc += mat[i3, k3] * src_3d[i1, i2, k3]
                end
                out_3d[i1, i2, i3] = acc
            end
        end
    end

    return out
end

@testset "Tensor Products 3D" begin
    n = 4
    ndim = 3
    nd = 2
    I_mat = Matrix{Float64}(I, n, n)
    vals = rand(nd, n^3)
    out = similar(vals)

    BoxDMK.tensor_product_apply!(out, (I_mat, I_mat, I_mat), vals, n, ndim, nd)
    @test out ≈ vals

    A = rand(n, n)
    B = rand(n, n)
    C = rand(n, n)
    K = kron(C, kron(B, A))

    BoxDMK.tensor_product_apply!(out, (A, B, C), vals, n, ndim, nd)
    @test out ≈ vals * K' atol = 1e-11

    workspace = BoxDMK._tensor_apply_workspace(Float64, nd, n, ndim)
    dim2_out = similar(vals)
    dim2_expected = _apply_dim_reference(vals, B, n, nd, 2)
    BoxDMK._tensor_product_apply_dim_3d!(dim2_out, vals, B, n, nd, 2, workspace)
    @test dim2_out ≈ dim2_expected atol = 1e-11
end

@testset "P2C/C2P Transforms" begin
    basis = LegendreBasis()
    p2c = BoxDMK.p2c_transform(basis, 6, 3)
    c2p = BoxDMK.c2p_transform(basis, 6, 3)

    @test size(p2c) == (6, 6, 3, 8)
    @test size(c2p) == (6, 6, 3, 8)

    for ic in 1:8, d in 1:3
        @test c2p[:, :, d, ic] ≈ transpose(p2c[:, :, d, ic])
    end
end

@testset "arena pool stays bounded under varying buffer sizes" begin
    BoxDMK.release_memory!()
    big = 0
    for k in 1:200
        ar = BoxDMK.Arena()
        sz = [1000 + 37 * ((k * 7919 + j) % 101) for j in 1:5]     # sizes vary from solve to solve
        for n in sz
            BoxDMK.alloc(ar, n)
        end
        big = max(big, sum(sz))
        BoxDMK.release!(ar)
        @test BoxDMK._POOL_WORDS[] <= 2 * BoxDMK._ARENA_MAX[]
    end
    @test BoxDMK._POOL_WORDS[] == sum(n * length(v) for (n, v) in BoxDMK._POOL; init = 0)
    BoxDMK.release_memory!()
    @test BoxDMK._POOL_WORDS[] == 0 && isempty(BoxDMK._POOL)
end
