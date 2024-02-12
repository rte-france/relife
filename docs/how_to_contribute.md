# Dive into `data`

`data` module handles survival data loading. Basically, survival data are composed of :
- lifetime data
- other data like covariates

In Relife2, all lifetime data are objects derived from `LifetimeFormat`. This object expects methods that define left-right-interval-regular values and index. What are they ? It depends on the type of lifetime data you are considering :

- when using `BaseCensoredLifetime` and `AdvancedCensoredLifetime`, left-right-interval-regular values mean censored or regular lifetimes.   Left-right-interval-regular indexes mean censored or regular lifetimes index
- when using `Truncation`, left-right-interval-regular values mean truncation values or regular lifetimes. Left-right-interval-regular indexes mean truncated or regular lifetimes index

Why it has to be like that ? `BaseCensoredLifetime`, `AdvancedCensoredLifetime` and `Truncation` respect the same structure as they are all lifetimes data. Thus, we found it more convenient to make them derived from the same object.

For more convenience, their instanciations are handled by two factories : `lifetimes` and `truncations`.

**Want to contribute ? :**  at this step, if you want to contribute, please verify that either `BaseCensoredLifetime`, `AdvancedCensoredLifetime` or `Truncation` can handle your needs. If not, make a new object derived from `LifetimeFormat` and add it to the good factory.