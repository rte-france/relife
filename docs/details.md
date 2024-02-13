# 1. ReLife2

![](../img/relife_modules.png)

ReLife2 is composed of three modules:

- `data` : it contains all necessary objects to load data used in ReLife2
- `survival`: it contains all objects used for survival analysis workflow. This module is composed of three submodules `parametric`, `nonparametric` and `semiparametric`  
- `policy`: it contains all objects used for reliability theory and renewal theory


# 2. `data` module

![](../img/relife_data.png)

The `data` module is composed of several objects. For a basic usage, one can only focus on the `SurvivalData` dataclass and got to its dedicated section. For contributors, please take a look at the next section to understand our conception.

## `LifetimeDecoder` family

To understand how `data` module works, it is necessary to know the `decoder` submodule. It contains objects that decodes lifetime data. Why ? Survival analysis carries on lifetime data whose format is not always constant. Our `LifetimeDecoder` objects share the same setup and their role is to extract all necessary information from given lifetime data. For now, there are three decoder implementations :

- `BaseCensoredLifetime`
- `AdvancedCensoredLifetime`
- `Truncation`

![](../img/data_decoder.png)

As you can see, every `LifetimeDecoder` implements `get_*_values()` and `get_*_index()` methods. Here `*` means either left, right, interval or regular. It is just getter methods returning either lifetime values or index. The following table describes more precisely what it means for each implemented decoders

| |`get_*_values`()|`get_*_index`()|
|-|-|-|
|`BaseCensoredLifetime`|return left-right-interval censored or regular (observed) lifetime values|return left-right-interval censored or regular (observed) lifetimes index|
|`AdvancedCensoredLifetime`|return left-right-interval censored or regular (observed) lifetime values|return left-right-interval censored or regular (observed) lifetimes index|
|`Truncation`|return left-right-interval truncation values or regular (not truncated) lifetime values|return left-right-interval truncated or regular (not truncated) lifetimes index|

Because one user doesn't want to know all the decoders, their instanciations are handled by two factories : `censoredlifetimes_decoder` and `truncations_decoder`.

**Want to contribute ? :** at this step, you may noticed that this code structure allows every one to add its own `LifetimeDecoder`. To do so, please make sure that your decoder object inherits from `LifetimeDecoder`. It is necessary that every decoder shares the same structure to be used in our dataclass object (see next section). You might also extend current factories to return your decoder. 


## `SurvivalData` dataclass

Previous decoder objects serve `SurvivalData` dataclass. They fuel this dataclass attributes. As all decoders share the same structure, the initialization of the dataclass must not change. Only extensions are allowed.
`SurvivalData` dataclass proposes three main methods allowing to access rapidly to lifetime information in a more intelligibly manner :

- `observed()`
- `censored()`
- `truncated()`

