# Leverage and Contagion Effects in Implied Volatility Surfaces: A Mixed Neural Vine Copula Approach
_Project by Luca Leimbeck Del Duca | **Partner Firm:** Optiver_

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![uv](https://img.shields.io/badge/uv-fast_package_manager-black.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research--complete-success.svg)

## Abstract 
Forecasting the joint evolution of implied volatility surfaces across a high-dimensional universe of assets is a critical challenge for market makers, particularly during periods of systemic stress. Traditional econometric models often struggle to reconcile the arbitrage-free geometry of the surface with the complex, non-linear dependence structure of asset returns. This paper proposes a unified deep generative framework that decomposes the joint forecasting problem into three sequential stages: geometry, dynamics, and topology. 

First, the surface stochastic volatility inspired (SSVI) parameterization and beta-adjusted multilevel functional principal component analysis ($\beta^{adj}$-mfPCA) are employed to construct an arbitrage-free, low-dimensional market representation. Second, the temporal evolution of the extracted factors is modeled using neural stochastic differential equations (NSDE), which capture path-dependent memory structures without imposing the rigid parametric constraints of standard linear benchmarks. Finally, the high-dimensional, asymmetric dependence structure is estimated via a novel differentiable mixed regular vine copula. Evaluating this framework across a diverse panel of 14 assets reveals that the dynamic neural topology significantly outperforms benchmarks in capturing asymmetric tail dependence and passing strict conditional coverage criteria. Furthermore, ex-ante delta-hedging simulations accurately isolate the systematic vanna bleed, proving the economic value of correctly specifying high-dimensional market contagion.

## Repository Architecture

The codebase is modularized to strictly mirror the methodological stages detailed in the research framework.

```text
├── config/                         # Global project configuration
│   ├── assets.json                 # Panel definitions (14 CBOE assets)
│   └── settings.py                 # Hyperparameters and path management
├── data/                           # Data storage (git-ignored)
│   └── results/                    # Persisted artifacts (backtests, copulas, factors)
├── src/                            # Core model pipeline
│   ├── preprocessing/              # Option chain filtering, arbitrage checks, and rate construction
│   ├── fitting/                    # SSVI calibration engine via Differential Evolution & SLSQP
│   ├── compression/                # Beta-adjusted multilevel functional PCA (β^adj-mfPCA)
│   ├── dynamics/                   # Marginal filtration (NGARCH-t, HAR-GARCH, NSDE, EVT tail modeling)
│   ├── dependence/                 # Topological networks (Static, GAS, and Neural Dynamic Vines)
│   ├── backtesting/                # Ex-post VaR/ES and ex-ante vanna bleed hedging simulations
│   └── intermediate_diagnostics/   # Pipeline validation and diagnostic logging
├── main.py                         # Primary execution entry point
├── pyproject.toml                  # Project metadata and dependencies
└── uv.lock                         # Strict deterministic dependency locking
```

## Methodology Flowchart

```mermaid
flowchart TD
    %% Data Sources
    subgraph "Data Sources"
        RawData["Raw Option Chains (14 Assets, CBOE)"]:::data
        Rates["Risk-Free Rates & Corp. Actions"]:::data
    end

    %% Stage 1: Geometry
    subgraph "Stage 1: Geometry"
        Preproc["Preprocessing & Splitting"]:::process
        SSVI["SSVI Surface Calibration"]:::model
        PCA["Beta-Adjusted mfPCA"]:::model
    end

    %% Stage 2: Dynamics
    subgraph "Stage 2: Dynamics (Marginal Whitening)"
        Spot["Spot Returns: NGARCH-t"]:::model
        VolHAR["Vol Factors: HAR-GARCH"]:::model
        VolNSDE["Vol Factors: Neural SDE"]:::model
        EVT["EVT Tail Calibration"]:::process
    end

    %% Stage 3: Topology
    subgraph "Stage 3: Topology (Dependence)"
        Static["Static Mixed Vine"]:::model
        GAS["GAS Mixed Vine"]:::model
        Neural["Neural Dynamic Vine"]:::model
    end

    %% Evaluation
    subgraph "Risk Simulation & Evaluation"
        ExPost["Ex-Post Backtesting (VaR, ES)"]:::decision
        ExAnte["Ex-Ante Scenario Analysis (Vanna Bleed)"]:::process
    end

    %% Flow
    RawData --> Preproc
    Rates --> Preproc
    Preproc --> SSVI
    SSVI --> PCA

    PCA --> VolHAR
    PCA --> VolNSDE

    VolHAR --> EVT
    VolNSDE --> EVT

    Spot --> Static
    EVT --> Static
    
    Spot --> GAS
    EVT --> GAS
    
    Spot --> Neural
    EVT --> Neural

    Static --> ExPost
    GAS --> ExPost
    Neural --> ExPost

    ExPost --> ExAnte

    %% Styles
    classDef data fill:#AED6F1,stroke:#1B4F72,color:#1B2631;
    classDef process fill:#ABEBC6,stroke:#1D8348,color:#145A32;
    classDef model fill:#F9E79F,stroke:#B7950B,color:#7D6608;
    classDef decision fill:#F5B7B1,stroke:#943126,color:#641E16;
```
