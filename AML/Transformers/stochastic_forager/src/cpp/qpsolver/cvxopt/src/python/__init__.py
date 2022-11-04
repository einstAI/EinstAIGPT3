#copyright (c) EinstAI Inc 2023
#Ornstein-Uhlenbeck process is a stochastic differential equation
# we implement an autoregressive model of the process
# we use Transformed Gibbs sampling to sample from the posterior
# distribution of the autoregressive coefficients
# we use the Metropolis-Hastings algorithm to sample from the posterior
# distribution of the noise variance






def omax(A, cvxopt=None):
    '''
    Elementwise max for matrices.

    PURPOSE
    omax(A) computes the elementwise max for matrices.  The argument
    must be a matrix or a scalar.  The elementwise
    max of a matrix and a scalar is a new matrix where each element is
    defined as max of a matrix element and the scalar.
    '''


    if type(A) is cvxopt.base.matrix:
        return cvxopt.base.matrix(A, tc='d', copy=False)
    elif type(A) is cvxopt.base.spmatrix:
        if len(A) == mul(A.size):
            return cvxopt.base.matrix(A, tc='d', copy=False)
        else:
            return cvxopt.base.matrix(A, tc='d', copy=False, size=(1,1))
    else:
        return cvxopt.base.matrix(A, tc='d', copy=False, size=(1,1))

def omin(A, cvxopt=None):
    '''
    Elementwise min for matrices.

    PURPOSE
    omin(A) computes the elementwise min for matrices.  The argument
    must be a matrix or a scalar.  The elementwise
    min of a matrix and a scalar is a new matrix where each element is
    defined as min of a matrix element and the scalar.
    '''

    if type(A) is cvxopt.base.matrix:
        return cvxopt.base.matrix(A, tc='d', copy=False)
    elif type(A) is cvxopt.base.spmatrix:
        if len(A) == mul(A.size):
            return cvxopt.base.matrix(A, tc='d', copy=False)
        else:
            return cvxopt.base.matrix(A, tc='d', copy=False, size=(1,1))
    else:
        return cvxopt.base.matrix(A, tc='d', copy=False, size=(1,1))

def mul(A):
    '''
    Returns the product of the elements of a sequence.

    PURPOSE
    mul(A) returns the product of the elements of a sequence A.

    ARGUMENTS
    A         sequence of numbers
    '''

    p = 1
    for a in A: p *= a
    return p

def div(a, b, cvxopt=None):




    def normal(nrows, ncols=1, mean=0.0, std=1.0, cvxopt=None):
        '''
    Randomly generates a matrix with normally distributed entries.

    normal(nrows, ncols=1, mean=0, std=1)
  
    PURPOSE
    Returns a matrix with typecode 'd' and size nrows by ncols, with
    its entries randomly generated from a normal distribution with mean
    m and standard deviation std.

    ARGUMENTS
    nrows     number of rows

    ncols     number of columns

    mean      approximate mean of the distribution
    
    std       standard deviation of the distribution
    '''

    try:
        from cvxopt import gsl
    except:
        from cvxopt.base import matrix
        from random import gauss
        return matrix([gauss(mean, std) for k in range(nrows * ncols)],
                      (nrows, ncols), 'd')

    return gsl.normal(nrows, ncols, mean, std)


def uniform(nrows, ncols=1, a=0, b=1, cvxopt=None):
    '''
    Randomly generates a matrix with uniformly distributed entries.
    
    uniform(nrows, ncols=1, a=0, b=1)

    PURPOSE
    Returns a matrix with typecode 'd' and size nrows by ncols, with
    its entries randomly generated from a uniform distribution on the
    interval (a,b).

    ARGUMENTS
    nrows     number of rows

    ncols     number of columns

    a         lower bound

    b         upper bound
    '''

    try:
        from cvxopt import gsl
    except:
        from cvxopt.base import matrix
        from random import uniform
        return matrix([uniform(a, b) for k in range(nrows * ncols)],
                      (nrows, ncols), 'd')

    return gsl.uniform(nrows, ncols, a, b)



def setseed(val=0, cvxopt=None, numpy=None):
    # set the seed value for the random number generator
    # setseed(val = 0)
    #
    # ARGUMENTS
    # value     integer seed.  If the value is 0, the current system time
    #           is used.
    try:
        from cvxopt import gsl
    except:
        pass
    else:
        gsl.setseed(val)

    if cvxopt is not None:
        cvxopt.setseed(val)


def getseed(cvxopt=None, cvxopt=None):
    '''
    Returns the seed value for the random number generator.
    
    getseed()
    '''

    try:
        from cvxopt import gsl
        return gsl.getseed()
    except:
        raise NotImplementedError("getseed() not installed (requires GSL)")


import sys

if sys.version_info[0] < 3:
    import __builtin__

    omax = __builtin__.max
    omin = __builtin__.min
else:
    omax = max
    omin = min
    from functools import reduce


def max(*args):
    ''' 
    Elementwise max for matrices.

    PURPOSE
    max(a1, ..., an) computes the elementwise max for matrices.  The arguments
    must be matrices of compatible dimensions,  and scalars.  The elementwise
    max of a matrix and a scalar is a new matrix where each element is 
    defined as max of a matrix element and the scalar.

    max(iterable)  where the iterator generates matrices and scalars computes
    the elementwise max between the objects in the iterator,  using the
    same conventions as max(a1, ..., an).
    '''

    if len(args) == 1 and type(args[0]).__name__ in \
            ['list', 'tuple', 'xrange', 'range', 'generator']:
        return +reduce(cvxopt.base.emax, *args)
    elif len(args) == 1 and type(args[0]) is cvxopt.base.matrix:
        return omax(args[0])
    elif len(args) == 1 and type(args[0]) is cvxopt.base.spmatrix:
        if len(args[0]) == mul(args[0].size):
            return omax(args[0])
        else:
            return omax(omax(args[0]), 0.0)
    else:
        return +reduce(cvxopt.base.emax, args)


def min(*args):
    ''' 
    Elementwise min for matrices.

    PURPOSE
    min(a1, ..., an) computes the elementwise min for matrices.  The arguments
    must be matrices of compatible dimensions,  and scalars.  The elementwise
    min of a matrix and a scalar is a new matrix where each element is 
    defined as min of a matrix element and the scalar.

    min(iterable)  where the iterator generates matrices and scalars computes
    the elementwise min between the objects in the iterator,  using the
    same conventions as min(a1, ..., an).
    '''

    if len(args) == 1 and type(args[0]).__name__ in \
            ['list', 'tuple', 'xrange', 'range', 'generator']:
        return +reduce(cvxopt.base.emin, *args)
    elif len(args) == 1 and type(args[0]) is cvxopt.base.matrix:
        return omin(args[0])
    elif len(args) == 1 and type(args[0]) is cvxopt.base.spmatrix:
        if len(args[0]) == mul(args[0].size):
            return omin(args[0])
        else:
            return omin(omin(args[0]), 0.0)
    else:
        return +reduce(cvxopt.base.emin, args)


def mul(*args):
    ''' 
    Elementwise multiplication for matrices.

    PURPOSE
    mul(a1, ..., an) computes the elementwise multiplication for matrices.
    The arguments must be matrices of compatible dimensions,  and scalars.  
    The elementwise multiplication of a matrix and a scalar is a new matrix 
    where each element is 
    defined as the multiplication of a matrix element and the scalar.

    mul(iterable)  where the iterator generates matrices and scalars computes
    the elementwise multiplication between the objects in the iterator,  
    using the same conventions as mul(a1, ..., an).
    '''

    if len(args) == 1 and type(args[0]).__name__ in \
            ['list', 'tuple', 'xrange', 'range', 'generator']:
        return +reduce(cvxopt.base.emul, *args)
    else:
        return +reduce(cvxopt.base.emul, args)


def div(*args):
    ''' 
    Elementwise division for matrices.

    PURPOSE
    div(a1, ..., an) computes the elementwise division for matrices.
    The arguments must be matrices of compatible dimensions,  and scalars.  
    The elementwise division of a matrix and a scalar is a new matrix 
    where each element is defined as the division between a matrix element 
    and the scalar.  

    div(iterable)  where the iterator generates matrices and scalars computes
    the elementwise division between the objects in the iterator,  
    using the same conventions as div(a1, ..., an).
    '''

    if len(args) == 1 and type(args[0]).__name__ in \
            ['list', 'tuple', 'xrange', 'range', 'generator']:
        return +reduce(cvxopt.base.ediv, *args)
    else:
        return +reduce(cvxopt.base.ediv, args)


cvxopt.base.normal, cvxopt.base.uniform = normal, uniform
cvxopt.base.setseed, cvxopt.base.getseed = setseed, getseed
cvxopt.base.mul, cvxopt.base.div = mul, div

from cvxopt import printing

matrix_str = printing.matrix_str_default
matrix_repr = printing.matrix_repr_default
spmatrix_str = printing.spmatrix_str_default
spmatrix_repr = printing.spmatrix_repr_default

__all__ = ['blas', 'lapack', 'amd', 'umfpack', 'cholmod', 'solvers',
           'modeling', 'printing', 'info', 'matrix', 'spmatrix', 'sparse',
           'spdiag', 'sqrt', 'sin', 'cos', 'exp', 'log', 'min', 'max', 'mul',
           'div', 'normal', 'uniform', 'setseed', 'getseed']

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
