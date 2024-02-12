# 1. ReLife2

![](../img/relife_modules.png)

ReLife2 is composed of three modules:

- `data` : it contains all necessary objects to load data used in ReLife2
- `survival`: it contains all objects used for survival analysis workflow. This module is composed of three submodules `parametric`, `nonparametric` and `semiparametric`  
- `policy`: it contains all objects used for reliability theory and renewal theory


# 2. Modules

## 2.1. Module `data` 

![](../img/relife_data.png)

The `data` module is composed of several objects. For a basic usage can only focus on the `SurvivalData` object. For advanced contributors, please refer to [Dive into `data`](.how_to_contribute.md). You will find further details about ReLife2 data objects. For now, consider using `SurvivalData` object. `SurvivalData` is composed of three main methods :

- `observed()`
- `censored()`
- `truncated()`
