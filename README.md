# EpiDist.jl

Computational kernels for the epidemic-distancing models described in the accompanying manuscript.

## Installation

The code targets Julia 1.11.7. From the repository root:

```julia
using Pkg
Pkg.instantiate()
Pkg.test()
```

Both `Project.toml` and `Manifest.toml` are included. The project file declares the direct dependencies; the manifest locks the complete computational environment.

## Models

Loading the package exposes seven modules:

- `Commons`: shared analytical expressions and callback utilities
- `Null`: null model without distancing
- `Nash`: decentralized laissez-faire model (LF)
- `Ctrl`: controlled optimum model (CO), i.e., the utilitarian optimum without an endogenous population response
- `Adpt`: adaptive optimum model (AO)
- `DevC`: deviator model under the CO specification
- `DevL`: deviator model under the LF specification

## State-variable normalization

The epidemiological state variables `S`, `I`, and `R` in the code are normalized by the herd-immunity threshold, $1/\mathcal{R}_0$. Therefore, their relationship to the population fractions used in the main text is

\[
S_{\mathrm{code}} = \mathcal{R}_0 S_{\mathrm{text}}, \qquad
I_{\mathrm{code}} = \mathcal{R}_0 I_{\mathrm{text}}, \qquad
R_{\mathrm{code}} = \mathcal{R}_0 R_{\mathrm{text}}.
\]

For example, `S == 1` in the code corresponds to $S_{\mathrm{text}} = 1/\mathcal{R}_0$, the herd-immunity threshold.

Representative single-case calls are:

```julia
using EpiDist

null_solution = Null.shoot(3.0)
nash_result = Nash.aimed_shot(3.0, 20.0)
control_result = Ctrl.aimed_shot(3.0, 20.0)

# The adaptive search exposes its three branches separately.
adaptive_active = Adpt.hunt_active(1.5, 40.0)
adaptive_inactive = Adpt.hunt_inactive(1.5, 40.0)
adaptive_doormat = Adpt.hunt_doormat(1.5, 40.0)

# The deviator solvers take (R0, deviator shooting argument,
# background shooting argument).
dev_c_solution = DevC.shoot(7.5, 6.0554544523933395e-6,
                            0.0005577161565235696)
dev_l_solution = DevL.shoot(7.5, 6.0554544523933395e-6,
                            0.0005577161565235696)
```
