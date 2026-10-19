# API reference

## Free space

```@docs
build_tree
bdmk
PotentialField
evaluate_potential
evaluate_gradient
evaluate_hessian
```

## Periodic

```@docs
bdmk_periodic
PeriodicResult
evaluate_periodic
periodic_plan
PeriodicPlan
periodic_solve!
```

## Quasi-2D

```@docs
bdmk_q2d
q2d_plan
```

## Kernels and bases

```@docs
LaplaceKernel
YukawaKernel
SqrtLaplaceKernel
LegendreBasis
ChebyshevBasis
```

## Fortran reference and utilities

```@docs
bdmk_fortran
build_tree_fortran
bdmk_native
BoxDMK.release_memory!
```
