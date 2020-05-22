# Troubleshoot

## Warnings

* If you spot runtime warnings like `XXX not implemented`, it means that the computation graph contains operations like `XXX` that does not affect the data flow values, so it is safe that we do not implement its abstraction semantics.
* If you spot runtime warnings like `fail to analysis YYY due to NotImplemented`, it means that the computation graph contains operations like `YYY` whose abstraction semantics is not implemented. DEBAR handles `YYY` in a sound manner, it treats the output of `YYY` as unbounded, i.e., in the range `[-inf,+inf]`. For better analysis results, we encourage other developers to contribute more implementations of abstraction semantics. Please see [Analysis](./analysis.md) for how to implement abstraction semantics.
  The unhandled operations will be prompt into console before the static analysis.

## Runtime Errors

Please open an issue if you spot any runtime errors.

