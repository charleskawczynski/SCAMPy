
"""
    solve_tridiag(x, f, a, b, c, n, xtemp, γ, β)

Solves for `x` in the equation
          `Ax = f`
where `A` is a tridiagonal matrix:
```
           _                                           _ -1
          |  b[0] c[0]                                 |
          |  a[0] b[1]  c[1]                           |
          |        a[1]  b[2]  c[2]                    |
 x    =   |           *     *     *                    |   f
          |                  *     *     *             |
          |                    a[n-3] b[n-1]  c[n-1]   |
          |_                           a[n-1]  b[n]   _|

          |____________________________________________|
                                 A
```
and given arguments:
--------------------------------------------
| x[1:n]      | the result                |
| f[1:n]      | right hand side           |
| a[1:n-1]    | sub-diagonal              |
| b[1:n]      | main diagonal             |
| c[1:n-1]    | super-diagonal            |
| n           | system size               |
| xtemp[1:n]  | temporary                 |
| γ[1:n-1]    | temporary                 |
| β[1:n]      | temporary                 |
--------------------------------------------
"""
def solve_tridiag(x, f, a, b, c, n, xtemp, γ, β):
  # Define coefficients:
  β[0] = b[0]
  γ[0] = c[0]/β[0]
  for i in range(1, n-1):
    β[i] = b[i]-a[i-1]*γ[i-1]
    γ[i] = c[i]/β[i]
  β[n-1] = b[n-1]-a[n-2]*γ[n-2]

  # Forward substitution:
  xtemp[0] = f[0]/β[0]
  for i in range(1, n):
    m = f[i] - a[i-1]*xtemp[i-1]
    xtemp[i] = m/β[i]

  # Backward substitution:
  x[n-1] = xtemp[n-1]
  for i in range(n-2,-1,-1):
    x[i] = xtemp[i]-γ[i]*x[i+1]


"""
    solve_tridiag_stored(x, f, a, β, γ, n, xtemp)

Solves for `x` in the equation
          `Ax = f`
where `A` is a tridiagonal matrix.

Coefficients in solve_tridiag! can be pre-computed,
by applying LU factorization to A (shown below).
The coefficients, β and γ, can be computed in init_β_γ!.
```
 _                                           _
|  b[0]  c[0]                                 |
|  a[0]  b[1]  c[1]                           |
|         a[1]  b[2]  c[2]                    |
|           *     *     *                     |
|                 *     *     *               |
|                    a[n-3]  b[n-1]  c[n-1]   |
|_                            a[n-1]  b[n]   _|

=
 _                                        _   _                                      _ -1
|  β[0]                                    | |  1  γ[0]                               |
|  α[0]  β[1]                              | |        1  γ[1]                         |
|        α[1]  β[2]                        | |              1  γ[2]                   |
|           *     *     *                  | |                 *     *                |
|                 *     *                  | |                       *     *          |
|                    α[n-3]  β[n-1]        | |                             1   γ[n-1] |
|_                           α[n-1]  β[n] _| |_                                1     _|
```

and given arguments:
--------------------------------------------
| x[1:n]       | the result                |
| f[1:n]       | right hand side           |
| a[1:n-1]     | sub-diagonal              |
| β[1:n]       | temporary                 |
| γ[1:n-1]     | temporary                 |
| n            | system size               |
| xtemp[1:n]   | temporary                 |
--------------------------------------------
"""
def solve_tridiag_stored(x, f, a, β, γ, n, xtemp):
  # Forward substitution:
  xtemp[0] = f[0]/β[0]
  for i in range(1, n):
    m = f[i] - a[i-1]*xtemp[i-1]
    xtemp[i] = m/β[i]

  # Backward substitution:
  x[n-1] = xtemp[n-1]
  for i in range(n-2,-1,-1):
    x[i] = xtemp[i]-γ[i]*x[i+1]

"""
    init_β_γ(β, γ, a, b, c, n)

Returns the pre-computed coefficients, from applying
LU factorization, for the tridiagonal system. These
coefficients can be passed as arguments to solve_tridiag_stored!.
-----------------------------------------
| β[1:n]    | temporary                 |
| γ[1:n-1]  | temporary                 |
| a[1:n-1]  | sub-diagonal              |
| b[1:n]    | main diagonal             |
| c[1:n-1]  | super-diagonal            |
| n         | system size               |
-----------------------------------------
"""
def init_β_γ(β, γ, a, b, c, n):
  # Define coefficients:
  β[0] = b[0]
  γ[0] = c[0]/β[0]
  for i in range(1, n-1):
    β[i] = b[i]-a[i-1]*γ[i-1]
    γ[i] = c[i]/β[i]
  β[n-1] = b[n-1]-a[n-2]*γ[n-2]

